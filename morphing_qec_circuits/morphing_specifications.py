import numpy as np
import stim
from copy import deepcopy
import galois  # for matrix calculations over a finite field
from surface_sim.models import CircuitNoiseModel, NoiselessModel
from .distance_functions import nkd_params


class MorphingSpecifications:
    """
    A class containing all the features required to define a morphing circuit.

    ATTRIBUTES:

    mid_cycle_stabilisers represents all the stabilisers such that mid_cycle_stabilisers[pauli] contains all mid-cycle stabilisers
    stabiliser_contraction_rounds lists the rounds during which each corresponding mid cycle stabiliser is contracting
    gate_lists represents the CNOT gates used in each contraction circuit, such that gate_lists[c_round][layer] is a list of [ctrl, targ] CNOT gates that occur in the layer = layer of the contraction circuit in round = c_round.
    measurement_lists represents the measurements after each contraction circuit, such that measurement_lists[pauli][c_round] contains a list of measurements that occur after the contraction circuit round = c_round in measurement basis = pauli.
    mid_cycle_logicals represents the logical operators of the mid-cycle code, such that mid_cycle_logicals[pauli][i] lists the qubits in the support of the logical operator \bar{pauli}_{i}.
    redundancies represents the linear combinations of mid_cycle_stabilisers that product to the identity.  of indices that correspond  linear combinations of mid_cycle_stabilisers that product to the identity.
    coordinates_list is a list of complex numbers representing coordinates for each qubit.
    end_cycle_stabilisers represents the support of each mid_cycle_stabiliser on the data qubits at the end of each round, such that end_cycle_stabilisers[pauli][c_round][k] lists the support at the end of the round c_round of the mid-cycle stabiliser mid_cycle_stabiliser[pauli][k].
    stabiliser_measurement_indices represents the support of each mid-cycle stabiliser on measurements at the end of each round, such that stabiliser_measurement_indices[pauli][c_round][k] lists the indices i of the measurements (given explicitly by measurement_lists[pauli][c_round][i]) during the round c_round that are supported by the mid-cycle stabiliser mid_cycle_stabiliser[pauli][k].
    end_cycle_logicals represents the support of each mid-cycle logical on the data qubits at the end of each round, such that end_cycle_logicals[pauli][c_round][k] lists the support of at the end of the round c_round of the mid-cycle logical mid_cycle_logical[pauli][k].
    logical_measurement_indices represents the support of each mid-cycle logical on measurements at the end of each round, such that logical_measurement_indices[pauli][c_round][k] lists the indices i of the measurements (given explicitly by measurement_lists[pauli][c_round][i]) during the round c_round that are supported by the mid-cycle logical mid_cycle_logical[pauli][k].
    """

    def __init__(
        self,
        mid_cycle_stabilisers,
        stabiliser_contraction_rounds,
        gate_lists,
        measurement_lists,
        mid_cycle_logicals={},
        redundancies={},
        coordinates_list=[],
        end_cycle_stabilisers={},
        stabiliser_measurement_indices={},
        end_cycle_logicals={},
        logical_measurement_indices={},
        stabiliser_annotations={}
    ):
        self.mid_cycle_stabilisers = mid_cycle_stabilisers
        self.stabiliser_contraction_rounds = stabiliser_contraction_rounds
        self.gate_lists = gate_lists
        self.measurement_lists = measurement_lists
        self.mid_cycle_logicals = mid_cycle_logicals
        self.redundancies = redundancies
        self.coordinates_list = coordinates_list
        self.end_cycle_stabilisers = end_cycle_stabilisers
        self.stabiliser_measurement_indices = stabiliser_measurement_indices
        self.end_cycle_logicals = end_cycle_logicals
        self.logical_measurement_indices = logical_measurement_indices
        self.stabiliser_annotations = stabiliser_annotations

    def n_qubits(self):
        """
        Returns the number of qubits in the mid-cycle code.
        """
        return 1 + max(
            [
                qubit
                for pauli in ["X", "Z"]
                for stabiliser in self.mid_cycle_stabilisers[pauli]
                for qubit in stabiliser
            ]
        )

    def n_rounds(self):
        """
        Returns the number of contraction rounds of the morphing circuit.
        """
        num_rounds = [
            len(rounds_list)
            for rounds_list in [
                self.gate_lists,
                self.measurement_lists["X"],
                self.measurement_lists["Z"],
            ]
        ]
        if num_rounds.count(num_rounds[0]) == len(num_rounds):
            return num_rounds[0]
        else:
            raise ValueError(
                """The number of rounds is inconsistent between the gate and measurement lists."""
            )

    def n_redundancies(self):
        """
        Returns dictionary the number of redundancies between the X and Z stabilisers.
        """
        h = self.parity_check_matrices()
        n_redundancies = {pauli: len(h[pauli].T.null_space()) for pauli in ["X", "Z"]}
        return n_redundancies

    def n_stabilisers(self):
        """
        Returns a dictionary of the number of X and Z type stabilisers.
        """
        return {pauli: len(self.mid_cycle_stabilisers[pauli]) for pauli in ["X", "Z"]}

    def n_logicals(self):
        """
        Returns the number of logical operators in the circuit.
        """
        n_redundancies = self.n_redundancies()
        n_total_stabilisers = self.n_stabilisers()
        return (
            self.n_qubits()
            - sum(n_total_stabilisers.values())
            + sum(n_redundancies.values())
        )

    def mid_cycle_nkd_params(
        self, check_x_and_z=True, separate_x_and_z=False, quiet=False
    ):
        hs = self.parity_check_matrices()
        return nkd_params(
            hs,
            check_x_and_z=check_x_and_z,
            separate_x_and_z=separate_x_and_z,
            quiet=quiet,
        )

    def end_cycle_nkd_params(
        self,
        check_all_end_cycle_codes=True,
        check_x_and_z=True,
        separate_x_and_z=False,
        quiet=False,
    ):
        hs = self.end_cycle_parity_check_matrices()
        if check_all_end_cycle_codes:
            nkd_params_list = []
            for i in range(self.n_rounds()):
                nkd_params_list.append(
                    nkd_params(
                        hs[i],
                        check_x_and_z=check_x_and_z,
                        separate_x_and_z=separate_x_and_z,
                        quiet=quiet,
                    )
                )
            return nkd_params_list
        else:
            return nkd_params(
                hs[0],
                check_x_and_z=check_x_and_z,
                separate_x_and_z=separate_x_and_z,
                quiet=quiet,
            )

    def parity_check_matrices(self):
        """
        Returns a dictionary h such that each item h[pauli] is a parity-check matrix representing the stabilisers of type pauli.
        """
        h_int = {pauli: [] for pauli in ["X", "Z"]}
        for pauli in ["X", "Z"]:
            for stabiliser in self.mid_cycle_stabilisers[pauli]:
                new_row = np.zeros((self.n_qubits())).astype(int)
                for qubit in stabiliser:
                    new_row[qubit] = 1
                h_int[pauli].append(new_row)

        return {pauli: galois.GF(2)(h_int[pauli]) for pauli in ["X", "Z"]}

    def end_cycle_parity_check_matrices(self):
        n_qubits = self.n_qubits()
        n_rounds = self.n_rounds()
        hs = [{pauli: [] for pauli in ["X", "Z"]} for i in range(n_rounds)]
        if self.end_cycle_stabilisers == {}:
            self.compile_end_cycle_stabilisers_and_stabiliser_measurement_indices()
        for i in range(n_rounds):
            left_over_qubits = [
                j
                for j in range(n_qubits)
                if j
                not in self.measurement_lists["X"][i] + self.measurement_lists["Z"][i]
            ]
            for pauli in ["X", "Z"]:
                h_int = []
                for stabiliser in self.end_cycle_stabilisers[pauli][i]:
                    new_row = np.zeros((len(left_over_qubits))).astype(int)
                    for qubit in stabiliser:
                        if qubit in left_over_qubits:
                            new_row[left_over_qubits.index(qubit)] = 1
                    if new_row.any():
                        h_int.append(new_row)
                hs[i][pauli] = galois.GF(2)(h_int)
        return hs

    def compile_mid_cycle_logicals(self):
        """
        Calculates the mid-cycle logical operators of the mid-cycle code based on the mid-cycle stabilisers and assigns these to self.mid_cycle_logicals.
        """
        h = self.parity_check_matrices()
        h_kernel = {pauli: h[pauli].null_space() for pauli in ["X", "Z"]}
        stabilisers_to_logicals = {pauli: h[pauli].row_reduce() for pauli in ["X", "Z"]}

        mid_cycle_logicals = {pauli: [] for pauli in ["X", "Z"]}
        for pauli in ["X", "Z"]:
            other_pauli = "Z" if pauli == "X" else "X"
            for i in range(h_kernel[other_pauli].shape[0]):
                new_vectors = np.concatenate(
                    (stabilisers_to_logicals[pauli], [h_kernel[other_pauli][i]]), axis=0
                )
                if np.linalg.matrix_rank(new_vectors) != np.linalg.matrix_rank(
                    stabilisers_to_logicals[pauli]
                ):
                    stabilisers_to_logicals[pauli] = new_vectors
                    mid_cycle_logicals[pauli].append(
                        np.nonzero(h_kernel[other_pauli][i])[0].tolist()
                    )

        self.mid_cycle_logicals = mid_cycle_logicals

    def compile_redundancies(self):
        """
        Calculates the redundancies between the stabilisers of the mid-cycle code based on the mid-cycle stabilisers and assigns these to self.mid_cycle_redundancies.
        (Note that these redundancies do not change throughout the circuit since the identity operator is preserved by all unitary gates)
        """
        h = self.parity_check_matrices()
        hT_kernel = {pauli: h[pauli].T.null_space() for pauli in ["X", "Z"]}
        n_redundancies = {pauli: len(hT_kernel[pauli]) for pauli in ["X", "Z"]}
        # n_rounds = self.n_rounds()

        # mid_cycle_stabilisers = self.mid_cycle_stabilisers
        # stabiliser_round_offsets = {pauli: [len(mid_cycle_stabilisers[pauli][i]) for i in range(n_rounds)] for pauli in ["X", "Z"]}

        mid_cycle_redundancies = {pauli: [] for pauli in ["X", "Z"]}
        for pauli in ["X", "Z"]:
            for redundancy in hT_kernel[pauli]:
                # to_append = [[] for _ in range(n_rounds)]
                redundant_stabilisers = np.nonzero(redundancy)[0]
                """old stuff I'm just keeping in case I've stuffed something up
                for stabiliser_index in redundant_stabilisers:
                    stabiliser_index_to_append = stabiliser_index
                    for c_round in range(n_rounds):
                        if stabiliser_index_to_append < stabiliser_round_offsets[pauli][c_round]:
                            to_append[c_round].append(stabiliser_index_to_append)
                            break
                        elif c_round == n_rounds - 1:
                            raise ValueError("stabiliser_index was not allocated a c_round")
                        else:
                            stabiliser_index_to_append += -stabiliser_round_offsets[pauli][c_round]"""
                mid_cycle_redundancies[pauli].append(redundant_stabilisers)

        self.redundancies = mid_cycle_redundancies

    def mid_cycle_to_pre_measurement(self, mid_cycle_operator, pauli, c_round):
        """
        Manually propagates a multi-qubit Pauli operator (either a product of X's or Z's) from the mid-cycle code to the pre-measurement code in the round c_round.

        INPUTS

        mid_cycle_support lists the qubits the operator has support on
        pauli is "X" or "Z", depending on whether the operator is a product of Pauli X's or Z's
        c_round is an integer from 0 to self.n_rounds() describing the round in which the operator is being propagated.
        """
        if pauli not in ["X", "Z"]:
            raise ValueError("""pauli should be either the string 'X' or 'Z'. """)

        pre_measurement_operator = deepcopy(mid_cycle_operator)
        for gate_layer in self.gate_lists[c_round]:
            for gate in gate_layer:
                # gate[0] is the control of the CNOT and gate[1] is the target.
                # When propagating an X operator through a CNOT gate, nothing happens if the ctrl of the CNOT (the "check_qubit") does not overlap with
                #    the operator. If it does overlap, then the operator is multiplied by X on the targ of the CNOT (the "expansion_qubit").
                # Vice-versa for if the basis is Z.
                if pauli == "X":
                    check_qubit = gate[0]
                    expansion_qubit = gate[1]
                else:
                    check_qubit = gate[1]
                    expansion_qubit = gate[0]
                if check_qubit in pre_measurement_operator:
                    if expansion_qubit in pre_measurement_operator:
                        pre_measurement_operator.remove(expansion_qubit)
                    else:
                        pre_measurement_operator.append(expansion_qubit)
        return pre_measurement_operator

    def pre_measurement_to_end_cycle_and_measurement_indices(
        self, pre_measurement_operator, pauli, c_round
    ):
        """
        Manually propagates a multi-qubit Pauli operator (either a product of X's or Z's) from the pre-measurement code in the round c_round to the end-cycle code, including a list of indices representing the measurements in measurement_lists[c_round] that are included in the pre-measurement support of the operator.

        INPUTS

        pre_measurement_support lists the qubits the operator has support on
        pauli is "X" or "Z", depending on whether the operator is a product of Pauli X's or Z's
        c_round is an integer from 0 to self.n_rounds() describing the round in which the operator is being propagated.
        """
        end_cycle_operator = deepcopy(pre_measurement_operator)
        measurement_indices = []
        this_measurement_list = self.measurement_lists[pauli][c_round]
        for i in range(len(this_measurement_list)):
            # checks the measurements in the expanded c_round that the expanded detector has support over and adds their measurement index to the
            #   list of measurements in the detector.
            measured_qubit = this_measurement_list[i]
            if measured_qubit in pre_measurement_operator:
                measurement_indices.append(i)
                end_cycle_operator.remove(measured_qubit)
        return end_cycle_operator, measurement_indices

    def mid_cycle_to_end_cycle_and_measurement_indices(
        self, mid_cycle_operator, pauli, c_round
    ):
        """ """
        pre_measurement_operator = self.mid_cycle_to_pre_measurement(
            mid_cycle_operator, pauli, c_round
        )
        return self.pre_measurement_to_end_cycle_and_measurement_indices(
            pre_measurement_operator, pauli, c_round
        )

    def compile_end_cycle_stabilisers_and_stabiliser_measurement_indices(self):
        """ """
        n_rounds = self.n_rounds()
        end_cycle_stabilisers = {
            pauli: [[] for _ in range(n_rounds)] for pauli in ["X", "Z"]
        }
        stabiliser_measurement_indices = {
            pauli: [[] for _ in range(n_rounds)] for pauli in ["X", "Z"]
        }
        for pauli in ["X", "Z"]:
            for mid_cycle_stabiliser in self.mid_cycle_stabilisers[pauli]:
                for i in range(n_rounds):
                    end_cycle_stabiliser, measurement_indices = (
                        self.mid_cycle_to_end_cycle_and_measurement_indices(
                            mid_cycle_stabiliser, pauli, i
                        )
                    )
                    end_cycle_stabilisers[pauli][i].append(end_cycle_stabiliser)
                    stabiliser_measurement_indices[pauli][i].append(measurement_indices)
        self.end_cycle_stabilisers = end_cycle_stabilisers
        self.stabiliser_measurement_indices = stabiliser_measurement_indices

    def compile_end_cycle_logicals_and_logical_measurement_indices(self):
        """ """
        n_rounds = self.n_rounds()
        end_cycle_logicals = {
            pauli: [[] for _ in range(n_rounds)] for pauli in ["X", "Z"]
        }
        logical_measurement_indices = {
            pauli: [[] for _ in range(n_rounds)] for pauli in ["X", "Z"]
        }
        for pauli in ["X", "Z"]:
            for mid_cycle_logical in self.mid_cycle_logicals[pauli]:
                for i in range(n_rounds):
                    end_cycle_logical, measurement_indices = (
                        self.mid_cycle_to_end_cycle_and_measurement_indices(
                            mid_cycle_logical, pauli, i
                        )
                    )
                    end_cycle_logicals[pauli][i].append(end_cycle_logical)
                    logical_measurement_indices[pauli][i].append(measurement_indices)
        self.end_cycle_logicals = end_cycle_logicals
        self.logical_measurement_indices = logical_measurement_indices

    def compile_all(self, check_compiled=False):
        """ """
        if (not check_compiled) or self.mid_cycle_logicals == {}:
            self.compile_mid_cycle_logicals()
        if (not check_compiled) or self.redundancies == {}:
            self.compile_redundancies()
        if (
            (not check_compiled)
            or self.end_cycle_stabilisers == {}
            or self.stabiliser_measurement_indices == {}
        ):
            self.compile_end_cycle_stabilisers_and_stabiliser_measurement_indices()
        if (
            (not check_compiled)
            or self.end_cycle_logicals == {}
            or self.logical_measurement_indices == {}
        ):
            self.compile_end_cycle_logicals_and_logical_measurement_indices()

    def get_morphing_XZ_memory_circuit(
        self,
        noise_setup,
        T,
        noise_model_generator=lambda noise_setup, qubit_dictionary: CircuitNoiseModel(
            noise_setup, qubit_dictionary
        ),
    ):
        """
        Seems to work?
        """
        if T < 1:
            raise ValueError("T is too smol for the mol, make him bigger")

        n_qubits = self.n_qubits()
        n_logicals = self.n_logicals()
        n_rounds = self.n_rounds()
        n_stabilisers = self.n_stabilisers()
        n_layers = [len(self.gate_lists[i]) for i in range(n_rounds)]

        self.compile_all(check_compiled=True)

        final_round = (T - 1) % n_rounds

        # The circuit starts in the end-cycle code \tilde{C}_{-1} and finishes in the end-cycle code \tilde{C}_{T}
        # The circuit starts with the expanding gates of c_round -1, then the contracting gates of c_round 0.
        # The contraction gates of final_round will just have been completed at the end of the circuit.

        qubit_dictionary = {str(i): i for i in range(n_qubits)}
        all_qubits_dictionary = {str(i): i for i in range(n_qubits + n_logicals)}
        noise_model = noise_model_generator(noise_setup, qubit_dictionary)
        noiseless_model = NoiselessModel(all_qubits_dictionary)
        tick = noiseless_model.tick()

        detector_circuits = self.mid_experiment_detector_circuits()
        observable_circuits = self.mid_experiment_logical_observable_circuits()

        circ = self.coordinate_circuit(n_extra_qubits=n_logicals)

        circ += noiseless_model.reset_x(
            [str(i) for i in range(n_qubits, n_qubits + n_logicals)]
        )
        circ += noiseless_model.reset_z([str(i) for i in range(n_qubits)])

        for i in range(n_logicals):
            circ += tick
            for qubit in self.end_cycle_logicals["X"][0][i]:
                circ += noiseless_model.cnot([str(n_qubits + i), str(qubit)])

        for i in range(1, n_rounds):
            circ += self.morphing_QEC_round(noiseless_model, i - 1 / 2)
            circ += observable_circuits[i]

        # perfect Bell state prepared, now run T cycles

        n_repeats = T // n_rounds
        n_left_over_rounds = T % n_rounds

        repeated_circ = stim.Circuit()
        for i in range(n_rounds):
            repeated_circ += self.morphing_QEC_round(noise_model, i - 1 / 2)
            repeated_circ += detector_circuits[i]
            repeated_circ += observable_circuits[i]

        circ += repeated_circ * n_repeats

        for i in range(n_left_over_rounds):
            circ += self.morphing_QEC_round(noise_model, i - 1 / 2)
            circ += detector_circuits[i]
            circ += observable_circuits[i]

        # perfect Bell measurements now

        for i in range(n_left_over_rounds, n_left_over_rounds + n_rounds - 1):
            circ += self.morphing_QEC_round(noiseless_model, i - 1 / 2)
            circ += detector_circuits[i % n_rounds]
            circ += observable_circuits[i % n_rounds]

        final_round = (n_left_over_rounds - 2) % n_rounds

        for i in range(n_logicals):
            circ += tick
            for qubit in self.end_cycle_logicals["X"][final_round][i]:
                circ += noiseless_model.cnot([str(n_qubits + i), str(qubit)])

        circ += tick
        circ += noiseless_model.measure_x(
            [str(i) for i in range(n_qubits, n_qubits + n_logicals)]
        )
        circ += noiseless_model.measure_z([str(i) for i in range(n_qubits)])
        for i in range(n_logicals):
            circ.append(
                "OBSERVABLE_INCLUDE", [stim.target_rec(i - n_qubits - n_logicals)], i
            )
        for i in range(n_logicals):
            circ.append(
                "OBSERVABLE_INCLUDE",
                [
                    stim.target_rec(j - n_qubits)
                    for j in self.end_cycle_logicals["Z"][final_round][i]
                ],
                i + n_logicals,
            )
        return circ

    def get_morphing_XZ_stability_circuit(
        self,
        noise_setup,
        T,
        noise_model_generator=lambda noise_setup, qubit_dictionary: CircuitNoiseModel(
            noise_setup, qubit_dictionary
        ),
    ):
        """ """

        n_qubits = self.n_qubits()
        n_redundancies = self.n_redundancies()
        n_rounds = self.n_rounds()
        n_stabilisers = self.n_stabilisers()
        n_layers = [len(self.gate_lists[i]) for i in range(n_rounds)]

        if T < n_rounds:
            raise ValueError("T is too smol for the mol, make him bigger")

        self.compile_all(check_compiled=True)

        final_round = (T - 1) % n_rounds

        # The circuit starts in the end-cycle code \tilde{C}_{-1} and finishes in the end-cycle code \tilde{C}_{T}
        # The circuit starts with the expanding gates of c_round -1, then the contracting gates of c_round 0.
        # The contraction gates of final_round will just have been completed at the end of the circuit.

        qubit_dictionary = {str(i): i for i in range(n_qubits)}
        noise_model = noise_model_generator(noise_setup, qubit_dictionary)
        tick = noise_model.tick()

        detector_circuits_i = self.initial_detector_circuits()
        detector_circuits = self.mid_experiment_detector_circuits()
        observable_circuits = self.initial_redundancy_observable_circuits()

        circ = self.coordinate_circuit(n_extra_qubits=0)

        circ += noise_model.reset_y([str(i) for i in range(n_qubits)])

        # first (n_rounds - 1) rounds need to use initial detectors (not mid-circuit detectors)
        # first (n_rounds - 1) rounds have observables

        for i in range(n_rounds - 1):
            circ += self.morphing_QEC_round(noise_model, i - 1 / 2)
            circ += detector_circuits_i[i]
            circ += observable_circuits[i]

        circ += self.morphing_QEC_round(noise_model, n_rounds - 1 - 1 / 2)
        circ += detector_circuits[-1]

        n_repeats = (T // n_rounds) - 1
        n_left_over_rounds = T % n_rounds

        repeated_circ = stim.Circuit()
        for i in range(n_rounds):
            repeated_circ += self.morphing_QEC_round(noise_model, i - 1 / 2)
            repeated_circ += detector_circuits[i]

        circ += repeated_circ * n_repeats

        for i in range(n_left_over_rounds):
            circ += self.morphing_QEC_round(noise_model, i - 1 / 2)
            circ += detector_circuits[i]

        circ += tick
        circ += noise_model.measure_y([str(i) for i in range(n_qubits)])
        return circ

    def coordinate_circuit(self, n_extra_qubits=0):

        circuit_str = ""
        coordinates_list = self.coordinates_list

        min_x = 0
        max_x = 0
        min_y = 0
        max_y = 0
        for i in range(len(coordinates_list)):
            coord = coordinates_list[i]
            min_x = min(round(coord.real), min_x)
            max_x = max(round(coord.real), max_x)
            min_y = min(round(coord.imag), min_y)
            max_y = max(round(coord.imag), max_y)
        for i in range(len(coordinates_list)):
            coord = coordinates_list[i]
            circuit_str += f"QUBIT_COORDS({-min_x + round(coord.real)}, {(max_y - round(coord.imag))}) {i}\n"
            # the minus makes sure y always is pointing up.
            # Also should be such that all coords are always positive (to copy/paste into crumble if necessary)
        for j in range(n_extra_qubits):
            circuit_str += f"QUBIT_COORDS({-min_x + max_x + 1 + (j // (-min_y+max_y+1))}, {-min_y + max_y - (j % (-min_y+max_y+1))}) {len(coordinates_list) + j}\n"

        return stim.Circuit(circuit_str)

    def morphing_QEC_round(self, noise_model, QEC_round):
        """
        QEC_round should be a half-integer such that the first half of the round is the inverse of the QEC_round - 1/2 contraction circuit and the second half is
            the QEC_round + 1/2 contraction circuit.
        still to be properly bug tested.
        """
        n_qubits = self.n_qubits()
        n_rounds = self.n_rounds()
        tick = noise_model.tick()
        previous_round = round(QEC_round - 1 / 2) % n_rounds
        following_round = round(QEC_round + 1 / 2) % n_rounds
        circ = stim.Circuit()
        circ += tick
        reset_lists = {}
        for pauli in ["X", "Z"]:
            reset_lists[pauli] = self.measurement_lists[pauli][previous_round]
        reset_lists["idle"] = [
            i
            for i in range(n_qubits)
            if (not i in reset_lists["X"]) and (not i in reset_lists["Z"])
        ]
        circ += noise_model.reset_x([str(i) for i in reset_lists["X"]])
        circ += noise_model.reset_z([str(i) for i in reset_lists["Z"]])
        circ += noise_model.idle([str(i) for i in reset_lists["idle"]])
        circ += tick
        expanding_gates = self.gate_lists[previous_round]
        for j in range(len(expanding_gates) - 1, -1, -1):
            cnot_list = [i for gate in expanding_gates[j] for i in gate]
            circ += noise_model.cnot([str(i) for i in cnot_list])
            circ += noise_model.idle(
                [str(i) for i in range(n_qubits) if not (i in cnot_list)]
            )
            circ += tick
        contracting_gates = self.gate_lists[following_round]
        for j in range(len(contracting_gates)):
            cnot_list = [i for gate in contracting_gates[j] for i in gate]
            circ += noise_model.cnot([str(i) for i in cnot_list])
            circ += noise_model.idle(
                [str(i) for i in range(n_qubits) if not (i in cnot_list)]
            )
            circ += tick
        meas_lists = {}
        for pauli in ["X", "Z"]:
            meas_lists[pauli] = self.measurement_lists[pauli][following_round]
        meas_lists["idle"] = [
            i
            for i in range(n_qubits)
            if (not i in meas_lists["X"]) and (not i in meas_lists["Z"])
        ]
        circ += noise_model.measure_x([str(i) for i in meas_lists["X"]])
        circ += noise_model.measure_z([str(i) for i in meas_lists["Z"]])
        circ += noise_model.idle([str(i) for i in meas_lists["idle"]])
        return circ

    def mid_experiment_detector_circuits(self):
        """
        assumes you are in the "middle" of the experiment; that is, more than n_rounds after the initialisation.
        """
        n_rounds = self.n_rounds()
        detector_circuits = [stim.Circuit() for _ in range(n_rounds)]
        for pauli in ["X", "Z"]:
            for i in range(len(self.mid_cycle_stabilisers[pauli])):
                contraction_rounds = self.stabiliser_contraction_rounds[pauli][i]
                for j in range(len(contraction_rounds)):
                    rec_targets = []
                    current_contraction_round = contraction_rounds[j]
                    past_contraction_round = contraction_rounds[j - 1]
                    n_included_rounds = (
                        (current_contraction_round - past_contraction_round - 1)
                        % n_rounds
                    ) + 1
                    offset = len(self.measurement_lists["Z"][current_contraction_round])
                    offset += {
                        "X": len(
                            self.measurement_lists["X"][current_contraction_round]
                        ),
                        "Z": 0,
                    }[pauli]
                    for k in range(n_included_rounds):
                        this_round = (current_contraction_round - k) % n_rounds
                        rec_targets += [
                            stim.target_rec(l - offset)
                            for l in self.stabiliser_measurement_indices[pauli][
                                this_round
                            ][i]
                        ]
                        offset += len(self.measurement_lists["Z"][this_round - 1])
                        offset += len(
                            self.measurement_lists["X"][
                                this_round - {"X": 1, "Z": 0}[pauli]
                            ]
                        )
                    if self.stabiliser_annotations == {}:
                        detector_annotation = (0, 0, 0, {"X": 0, "Z": 3}[pauli])
                    else:
                        detector_annotation = self.stabiliser_annotations[pauli][i]
                    detector_circuits[current_contraction_round].append(
                        "DETECTOR", rec_targets, detector_annotation
                    )
        return detector_circuits

    def initial_detector_circuits(self):
        """
        defines the detector circuits for the BULK detectors (NOT boundary detectors) in the first n_rounds after the initialisation.
        This is used in the stability experiment because in general there may be some bulk detectors in the first n_rounds because some stabilisers may be contracting
        in more than one round.
        """
        n_rounds = self.n_rounds()
        detector_circuits = [stim.Circuit() for _ in range(n_rounds)]
        for pauli in ["X", "Z"]:
            for i in range(len(self.mid_cycle_stabilisers[pauli])):
                contraction_rounds = self.stabiliser_contraction_rounds[pauli][i]
                if len(contraction_rounds) > 1:
                    for j in range(1, len(contraction_rounds)):
                        rec_targets = []
                        current_contraction_round = contraction_rounds[j]
                        past_contraction_round = contraction_rounds[j - 1]
                        n_included_rounds = (
                            (current_contraction_round - past_contraction_round - 1)
                            % n_rounds
                        ) + 1
                        offset = len(
                            self.measurement_lists["Z"][current_contraction_round]
                        )
                        offset += {
                            "X": len(
                                self.measurement_lists["X"][current_contraction_round]
                            ),
                            "Z": 0,
                        }[pauli]
                        for k in range(n_included_rounds):
                            this_round = (current_contraction_round - k) % n_rounds
                            rec_targets += [
                                stim.target_rec(l - offset)
                                for l in self.stabiliser_measurement_indices[pauli][
                                    this_round
                                ][i]
                            ]
                            offset += len(self.measurement_lists["Z"][this_round - 1])
                            offset += len(
                                self.measurement_lists["X"][
                                    this_round - {"X": 1, "Z": 0}[pauli]
                                ]
                            )
                        if self.stabiliser_annotations == {}:
                            detector_annotation = (0, 0, 0, {"X": 0, "Z": 3}[pauli])
                        else:
                            detector_annotation = self.stabiliser_annotations[pauli][i]
                        detector_circuits[current_contraction_round].append(
                            "DETECTOR", rec_targets, detector_annotation
                        )
        return detector_circuits

    def mid_experiment_logical_observable_circuits(self):
        """
        assumes you are in the "middle" of the experiment; that is, more than n_rounds after the initialisation.
        this is for when your observable is a logical operator (as opposed to a stabiliser redundancy, in which case use initial_redundancy_observable_circuits below)
        """
        n_rounds = self.n_rounds()
        n_logicals = self.n_logicals()
        observable_circuits = [stim.Circuit() for _ in range(n_rounds)]
        for pauli in ["X", "Z"]:
            for i in range(n_logicals):
                for current_contraction_round in range(n_rounds):
                    offset = len(self.measurement_lists["Z"][current_contraction_round])
                    offset += {
                        "X": len(
                            self.measurement_lists["X"][current_contraction_round]
                        ),
                        "Z": 0,
                    }[pauli]
                    indices_to_include = self.logical_measurement_indices[pauli][
                        current_contraction_round
                    ][i]
                    targets_to_include = [
                        stim.target_rec(indices_to_include[j] - offset)
                        for j in range(len(indices_to_include))
                    ]
                    observable_circuits[current_contraction_round].append(
                        "OBSERVABLE_INCLUDE",
                        targets_to_include,
                        i + {"X": 0, "Z": n_logicals}[pauli],
                    )
        return observable_circuits

    def initial_redundancy_observable_circuits(self):
        """
        uses the initial (-1) end-cycle code of the experiment to define the redundancies that are included in each observable.
        """
        n_rounds = self.n_rounds()
        n_redundancies = self.n_redundancies()
        targets_to_include = [
            [[] for _ in range(n_redundancies["X"] + n_redundancies["Z"])]
            for _ in range(n_rounds - 1)
        ]
        redundancy_index = 0
        for pauli in ["X", "Z"]:
            for i in range(n_redundancies[pauli]):
                redundancy = self.redundancies[pauli][i]
                for j in range(len(redundancy)):
                    stabiliser_index = redundancy[j]
                    if (
                        n_rounds - 1
                        not in self.stabiliser_contraction_rounds[pauli][
                            stabiliser_index
                        ]
                    ):
                        for current_contraction_round in range(
                            self.stabiliser_contraction_rounds[pauli][stabiliser_index][
                                0
                            ]
                            + 1
                        ):
                            indices_to_include = self.stabiliser_measurement_indices[
                                pauli
                            ][current_contraction_round][stabiliser_index]
                            offset = len(
                                self.measurement_lists["Z"][current_contraction_round]
                            )
                            offset += {
                                "X": len(
                                    self.measurement_lists["X"][
                                        current_contraction_round
                                    ]
                                ),
                                "Z": 0,
                            }[pauli]
                            targets_to_include[current_contraction_round][
                                redundancy_index
                            ] += [
                                stim.target_rec(index - offset)
                                for index in indices_to_include
                            ]
                redundancy_index += 1
        observable_circuits = [stim.Circuit() for _ in range(n_rounds - 1)]
        for i in range(n_redundancies["X"] + n_redundancies["Z"]):
            for current_contraction_round in range(n_rounds - 1):
                observable_circuits[current_contraction_round].append(
                    "OBSERVABLE_INCLUDE",
                    targets_to_include[current_contraction_round][i],
                    i,
                )
        return observable_circuits
