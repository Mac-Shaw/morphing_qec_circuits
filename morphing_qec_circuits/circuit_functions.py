import numpy as np
import stim
from copy import deepcopy
from timeit import default_timer as timer
from surface_sim.models import CircuitNoiseModel, NoiselessModel


def separate_XZ_dem_from_circuit(circuit, **dem_kwargs):
    detector_X_indices = []
    detector_Z_indices = []
    n_observables = circuit.num_observables
    i = 0
    for instruction in circuit.flattened():
        if instruction.name in ["DETECTOR"]:
            if int(instruction.gate_args_copy()[3]) <= 2:
                detector_X_indices.append(i)
            else:
                detector_Z_indices.append(i)
            i += 1
    dem = circuit.detector_error_model().flattened()
    X_dem = stim.DetectorErrorModel(**dem_kwargs)
    Z_dem = stim.DetectorErrorModel(**dem_kwargs)
    for instruction in dem:
        if instruction.type == "error":
            args = instruction.args_copy()
            targs = instruction.targets_copy()
            X_detector_targs = [
                stim.target_relative_detector_id(detector_X_indices.index(targ.val))
                for targ in targs
                if (targ.is_relative_detector_id() and targ.val in detector_X_indices)
            ]
            Z_detector_targs = [
                stim.target_relative_detector_id(detector_Z_indices.index(targ.val))
                for targ in targs
                if (targ.is_relative_detector_id() and targ.val in detector_Z_indices)
            ]
            X_logical_targs = [
                targ
                for targ in targs
                if (targ.is_logical_observable_id() and targ.val < n_observables // 2)
            ]
            Z_logical_targs = [
                stim.target_logical_observable_id(targ.val - n_observables // 2)
                for targ in targs
                if (targ.is_logical_observable_id() and targ.val >= n_observables // 2)
            ]
            X_instruction = stim.DemInstruction(
                "error", args, X_detector_targs + X_logical_targs
            )
            Z_instruction = stim.DemInstruction(
                "error", args, Z_detector_targs + Z_logical_targs
            )
            X_dem.append(X_instruction)
            Z_dem.append(Z_instruction)
    return {"X": X_dem, "Z": Z_dem}, {"X": detector_X_indices, "Z": detector_Z_indices}


bp_osd_kwargs = {
    "bp_method": "minimum_sum",
    "osd_method": "osd_cs",
    "osd_order": 20,
    "max_iter": 10_000,
}


def sample_logical_flips(
    sampler,
    decoder,
    XZ_detector_indices={},
    target_flips=0,
    batch_size=0,
    max_time=0,
    separate_XZ=False,
):

    n_flips = 0
    n_shots = 0

    if separate_XZ:
        X_decoder = decoder["X"]
        Z_decoder = decoder["Z"]
        detector_X_indices = XZ_detector_indices["X"]
        detector_Z_indices = XZ_detector_indices["Z"]

    start = timer()
    while (n_flips < target_flips) and (timer() < start + max_time):
        defects, log_flips = sampler.sample(shots=batch_size, separate_observables=True)
        n_observables = len(log_flips[0, :])
        if separate_XZ:
            X_predictions = X_decoder.decode_batch(defects[:, detector_X_indices])
            Z_predictions = Z_decoder.decode_batch(defects[:, detector_Z_indices])
            predictions = np.concatenate((X_predictions, Z_predictions), axis=1)
        else:
            predictions = decoder.decode_batch(defects)
        n_flips += np.sum(np.any(predictions != log_flips, axis=1))
        n_shots += batch_size
    return n_flips, n_shots


def observables_to_detectors(circuit):
    new_circuit = stim.Circuit()
    n_observables = circuit.num_observables
    rec_targets = [[] for _ in range(n_observables)]
    for instruction in circuit.flattened():
        if instruction.name == "DETECTOR":
            continue
        elif instruction.name == "OBSERVABLE_INCLUDE":
            arg = round(instruction.gate_args_copy()[0])
            targets = instruction.targets_copy()
            for target in targets:
                rec_targets[arg].append(target.value)
        elif instruction.name in ["M", "MX", "MY", "MR", "MRX", "MRY", "MPP"]:
            to_add = 1 if instruction.name == "MPP" else len(instruction.targets_copy())
            for i in range(len(rec_targets)):
                for j in range(len(rec_targets[i])):
                    rec_targets[i][j] += -to_add
            new_circuit.append(instruction)
        else:
            new_circuit.append(instruction)
    for i in range(n_observables):  # range should be equal to the number of observables
        new_circuit.append(
            "DETECTOR", [stim.target_rec(target) for target in rec_targets[i]]
        )
    return new_circuit


def filter_circuit_detectors(
    circuit, detector_list=None, coordinates_condition=lambda coordinates: True
):
    new_circuit = stim.Circuit()
    if detector_list == None:
        detector_list = range(circuit.num_detectors)
    i = 0
    for instruction in circuit.flattened():
        if instruction.name == "DETECTOR":
            if i in detector_list and coordinates_condition(
                instruction.gate_args_copy()
            ):
                new_circuit.append(instruction)
                i += 1
            else:
                i += 1
        else:
            new_circuit.append(instruction)
    return new_circuit


def filter_dem_detectors(
    dem, detector_list=None, coordinates_condition=lambda coordinates: True
):
    declarations = stim.DetectorErrorModel()
    for dem_instruction in dem.flattened():
        if dem_instruction.type == "detector":
            declarations.append(dem_instruction)
    if detector_list == None:
        detector_list = range(dem.num_detectors)
    new_detector_list = []
    for dem_instruction in dem.flattened():
        if dem_instruction.type == "detector":
            if dem_instruction.targets_copy()[0].val in detector_list:
                if coordinates_condition(
                    np.array(dem_instruction.args_copy()).astype(int)
                ):
                    new_detector_list.append(dem_instruction.targets_copy()[0].val)
    new_dem = stim.DetectorErrorModel()
    for dem_instruction in dem.flattened():
        if dem_instruction.type == "error":
            new_targs = []
            targs = dem_instruction.targets_copy()
            for targ in targs:
                if targ.is_logical_observable_id or targ.val in new_detector_list:
                    new_targs.append(targ)
            new_instruction = stim.DemInstruction(
                "error", dem_instruction.args_copy(), new_targs
            )
            new_dem.append(new_instruction)
        else:
            new_dem.append(dem_instruction)
    return new_dem


def filter_detectors(object, **kwargs):
    if type(object) == stim._stim_sse2.Circuit:
        return filter_circuit_detectors(object, **kwargs)
    elif type(object) == stim._stim_sse2.DetectorErrorModel:
        return filter_dem_detectors(object, **kwargs)

def qec_circuit_to_XZ_memory_experiment(
        qec_circuits,
        coordinate_circuit,
        detector_circuits,
        logical_operator_basis,
        noise_setup,
        T,
        noise_model_generator=lambda noise_setup, qubit_dictionary: CircuitNoiseModel(
            noise_setup, qubit_dictionary
            ),
        ):
    """
    This is just a direct copy/paste from helper_functions.py copied at 16:17 28/08/2025.
    
    A bit hacky just for my purposes right now. Assumes the qec_detectors are given explicitly, but not the observable instructions.
    qec_circuit should not have any noise on it. It should end in a tick.
    logical_operator_basis is a dictionary of "X" and "Z"-type operators each of which should be a list of stim.PauliString objects over ALL of the qubits.
    """
    if type(qec_circuits) != list:
        qec_circuits = [qec_circuits]
    if type(detector_circuits) != list:
        detector_circuits = [detector_circuits]
    n_qubits = []
    for qec_circuit in qec_circuits:
        if type(qec_circuit) == str:
            qec_circuit = stim.Circuit(qec_circuit)
        if type(qec_circuit) != stim.Circuit:
            raise TypeError(f"Each element of qec_circuits must be str or stim.Circuit, but is instead type {type(qec_circuit)}")
        n_qubits.append(qec_circuit.num_qubits)
    n_qubits = max(n_qubits)
    if type(coordinate_circuit) == str:
        coordinate_circuit = stim.Circuit(coordinate_circuit)
        if type(qec_circuit) != stim.Circuit:
            raise TypeError(f"coordinate_circuit must be str or stim.Circuit, but is instead type {type(coordinate_circuit)}")
    for detector_circuit in detector_circuits:
        if type(detector_circuit) == str:
            detector_circuit = stim.Circuit(detector_circuit)
        if type(qec_circuit) != stim.Circuit:
            raise TypeError(f"Each element of detector_circuits must be str or stim.Circuit, but is instead type {type(detector_circuits)}")
        
    if T < 1:
        raise ValueError("T is too small for the mall, make him bigger")
    
    n_logicals = len(logical_operator_basis["X"])

    qubit_dictionary = {str(i): i for i in range(n_qubits)}
    all_qubits_dictionary = {str(i): i for i in range(n_qubits + n_logicals)}
    noise_model = noise_model_generator(noise_setup, qubit_dictionary)
    noiseless_model = NoiselessModel(all_qubits_dictionary)
    tick = noiseless_model.tick()
    
    noisy_qec_circuits = [stim.Circuit() for _ in range(len(qec_circuits))]
    reset_lists = [[] for _ in range(len(qec_circuits))]
    unitary_qec_circuits = [stim.Circuit() for _ in range(len(qec_circuits))]
    for j in range(len(qec_circuits)):
        for instruction in qec_circuits[j]:
            if instruction.name  == "R":
                reset_lists[j] += [i.qubit_value for i in instruction.targets_copy()]
                noisy_qec_circuits[j] += noise_model.reset([str(i.qubit_value) for i in instruction.targets_copy()])
            elif instruction.name == "RX":
                reset_lists[j] += [i.qubit_value for i in instruction.targets_copy()]
                noisy_qec_circuits[j] += noise_model.reset_x([str(i.qubit_value) for i in instruction.targets_copy()])
            elif instruction.name == "RY":
                reset_lists[j] += [i.qubit_value for i in instruction.targets_copy()]
                noisy_qec_circuits[j] += noise_model.reset_y([str(i.qubit_value) for i in instruction.targets_copy()])
            elif instruction.name == "M":
                noisy_qec_circuits[j] += noise_model.measure([str(i.qubit_value) for i in instruction.targets_copy()])
            elif instruction.name == "MX":
                noisy_qec_circuits[j] += noise_model.measure_x([str(i.qubit_value) for i in instruction.targets_copy()])
            elif instruction.name == "MY":
                noisy_qec_circuits[j] += noise_model.measure_y([str(i.qubit_value) for i in instruction.targets_copy()])
            elif instruction.name == "H":
                noisy_qec_circuits[j] += noise_model.hadamard([str(i.qubit_value) for i in instruction.targets_copy()])
                unitary_qec_circuits[j].append(instruction)
            elif instruction.name == "S":
                noisy_qec_circuits[j] += noise_model.s_gate([str(i.qubit_value) for i in instruction.targets_copy()])
                unitary_qec_circuits[j].append(instruction)
            elif instruction.name == "S_DAG":
                noisy_qec_circuits[j] += noise_model.s_dag_gate([str(i.qubit_value) for i in instruction.targets_copy()])
                unitary_qec_circuits[j].append(instruction)
            elif instruction.name == "XCX":
                noisy_qec_circuits[j] += noise_model.xcx([str(i.qubit_value) for i in instruction.targets_copy()])
                unitary_qec_circuits[j].append(instruction)
            elif instruction.name == "XCY":
                noisy_qec_circuits[j] += noise_model.xcy([str(i.qubit_value) for i in instruction.targets_copy()])
                unitary_qec_circuits[j].append(instruction)
            elif instruction.name == "XCZ":
                noisy_qec_circuits[j] += noise_model.xcz([str(i.qubit_value) for i in instruction.targets_copy()])
                unitary_qec_circuits[j].append(instruction)
            elif instruction.name == "YCX":
                noisy_qec_circuits[j] += noise_model.ycx([str(i.qubit_value) for i in instruction.targets_copy()])
                unitary_qec_circuits[j].append(instruction)
            elif instruction.name == "YCY":
                noisy_qec_circuits[j] += noise_model.ycy([str(i.qubit_value) for i in instruction.targets_copy()])
                unitary_qec_circuits[j].append(instruction)
            elif instruction.name == "YCZ":
                noisy_qec_circuits[j] += noise_model.ycz([str(i.qubit_value) for i in instruction.targets_copy()])
                unitary_qec_circuits[j].append(instruction)
            elif instruction.name == "CX":
                noisy_qec_circuits[j] += noise_model.cnot([str(i.qubit_value) for i in instruction.targets_copy()])
                unitary_qec_circuits[j].append(instruction)
            elif instruction.name == "CY":
                noisy_qec_circuits[j] += noise_model.cy([str(i.qubit_value) for i in instruction.targets_copy()])
                unitary_qec_circuits[j].append(instruction)
            elif instruction.name == "CZ":
                noisy_qec_circuits[j] += noise_model.cphase([str(i.qubit_value) for i in instruction.targets_copy()])
                unitary_qec_circuits[j].append(instruction)
            elif instruction.name == "TICK":
                noisy_qec_circuits[j] += noise_model.tick()
            else:
                raise NotImplementedError(f"A qec_circuit has an instruction {instruction.name} that I wasn't expecting!")
    
    backpropagated_logicals = deepcopy(logical_operator_basis)
    for i in range(T+2*len(qec_circuits)-1,-1,-1):
        for pauli in ["X", "Z"]:
            for j in range(n_logicals):
                post_reset_operator = backpropagated_logicals[pauli][j].before(unitary_qec_circuits[i%len(qec_circuits)])
                for reset_qubit in reset_lists[i%len(qec_circuits)]:
                    post_reset_operator[reset_qubit] = 0
                backpropagated_logicals[pauli][j] = post_reset_operator
                
    max_x = -np.inf
    min_y = np.inf
    max_y = -np.inf
    for instruction in coordinate_circuit:
        coords = instruction.gate_args_copy()
        max_x = max(max_x, coords[0])
        min_y = min(min_y, coords[1])
        max_y = max(max_y, coords[1])
    
    delta_y = max_y - min_y + 1
    
    extra_coordinate_circuit = stim.Circuit()
    for i in range(n_logicals):
        extra_coordinate_circuit.append("QUBIT_COORDS", i + n_qubits, [max_x+1-((max_x+min_y)%1)+(i//delta_y), min_y+(i%delta_y)])
            
    circ = coordinate_circuit
    circ += extra_coordinate_circuit
    
    # The noiseless part of the circuit. To be safe, after the Bell state is prepared, all the qec_circuits are run once each.
    
    circ += noiseless_model.reset_x(
        [str(i) for i in range(n_qubits, n_qubits + n_logicals)]
    )
    circ += noiseless_model.reset_z([str(i) for i in range(n_qubits)])
    
    circ += tick
    for i in range(n_logicals):
        circ.append(stim.CircuitInstruction("MPP", stim.target_combined_paulis(backpropagated_logicals["Z"][i])))
        circ.append(stim.CircuitInstruction("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], [i + n_logicals]))
        circ += tick
        
    for i in range(n_logicals):
        for j in range(len(backpropagated_logicals["X"][i])):
            qubit_Pauli_op = backpropagated_logicals["X"][i][j]
            if qubit_Pauli_op != 0:
                Pauli_str = ["I", "X", "Y", "Z"][qubit_Pauli_op]
                circ.append(stim.CircuitInstruction("C" + Pauli_str, [n_qubits + i, j]))
        circ += tick
    
    for i in range(len(qec_circuits)):
        circ += qec_circuits[i]

    # perfect Bell state prepared, now run T cycles

    repeated_circ = stim.Circuit()
    for i in range(len(qec_circuits)):
        repeated_circ += noisy_qec_circuits[i]
        repeated_circ += detector_circuits[i]

    circ += repeated_circ * (T//len(qec_circuits))
    
    for i in range(T%len(qec_circuits)):
        circ += noisy_qec_circuits[i]
        circ += detector_circuits[i]
    
    # final noiseless QEC rounds. Again to be safe, all the qec_circuits are run once each.
    # This can sometimes add some detectors that have no support on the noisy part of the circuit,
    #    this is ok though because it doesn't meaningfully change the dem.

    for i in range(len(qec_circuits)):
        circ += qec_circuits[(i+T)%len(qec_circuits)]
        circ += detector_circuits[(i+T)%len(qec_circuits)]

    for i in range(n_logicals):
        for j in range(len(logical_operator_basis["X"][i])):
            qubit_Pauli_op = logical_operator_basis["X"][i][j]
            if qubit_Pauli_op != 0:
                Pauli_str = ["I", "X", "Y", "Z"][qubit_Pauli_op]
                circ.append(stim.CircuitInstruction("C" + Pauli_str, [n_qubits + i, j]))
        circ += tick
    
    circ += noiseless_model.measure_x(
        [str(i) for i in range(n_qubits, n_qubits + n_logicals)]
    )
    for i in range(n_logicals):
        circ.append(
            "OBSERVABLE_INCLUDE", [stim.target_rec(i - n_logicals)], [i]
        )
    circ += tick

    for i in range(n_logicals):
        circ.append(stim.CircuitInstruction("MPP", stim.target_combined_paulis(logical_operator_basis["Z"][i])))
        circ.append(stim.CircuitInstruction("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], [i + n_logicals]))
        circ += tick
    
    return circ
