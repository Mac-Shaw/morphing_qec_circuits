import numpy as np
import stim
from copy import deepcopy
from timeit import default_timer as timer


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
    "max_bp_iters": 10_000,
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
            to_add = len(instruction.targets_copy())
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
