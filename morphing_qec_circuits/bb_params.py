import numpy as np
from copy import deepcopy
import galois  # for matrix calculations over a finite field
from .morphing_specifications import MorphingSpecifications
from .distance_functions import nkd_params

F2 = galois.GF(2)


class BBParams:
    def __init__(self, l, m, a_list, b_list):
        self.l = l
        self.m = m
        self.a_list = a_list
        self.b_list = b_list

    def nkd_params(self, quiet=False):
        hs = self.parity_check_matrices()
        return nkd_params(hs, check_x_and_z=False, quiet=quiet)

    def valid_standard_morphing_parameters(self, homomorphism_str):
        """
        Checks if a given Bivariate Bicycle code specification and homorphism define a valid morphing circuit.

        INPUTS

        l and m are positive integers such that the bicyclic group defining the BB code is Z_l times Z_m.
        a_list and b_list are lists of group elements defining the BB code. Each group element x^alpha y^beta is represented as a list of two integers [alpha,beta].
        homomorphism_str must be "x", "y" or "xy" depending on the choice of homomorphism (see Eq 1 of arXiv:2407.16336)
        """
        if homomorphism_str == "x":
            if self.l % 2 == 1:
                return False
            homomorphism = lambda group_element: group_element[0] % 2
        elif homomorphism_str == "y":
            if self.m % 2 == 1:
                return False
            homomorphism = lambda group_element: group_element[1] % 2
        elif homomorphism_str == "xy":
            if (self.l % 2 == 1) or (self.m % 2 == 1):
                return False
            homomorphism = (
                lambda group_element: (group_element[0] + group_element[1]) % 2
            )
        else:
            raise ValueError(
                "What a rotten piece of luck old chap, homomorphism_str needs to be x, y or xy for me to continue!"
            )

        if all([self.l > 0, self.m > 0, len(self.a_list) > 1, len(self.b_list) > 1]):
            for c_list in [self.a_list, self.b_list]:
                homomorphism_list = [homomorphism(c_el) for c_el in c_list]
                if (homomorphism_list.count(0) == 1) and (
                    homomorphism_list.count(1) == len(c_list) - 1
                ):
                    continue
                elif (homomorphism_list.count(1) == 1) and (
                    homomorphism_list.count(0) == len(c_list) - 1
                ):
                    continue
                else:
                    return False
            return True
        else:
            return False

    def valid_three_round_morphing_parameters(self, homomorphism_str):
        """
        Checks if a given Bivariate Bicycle code specification and homorphism define a valid morphing circuit.

        INPUTS

        l and m are positive integers such that the bicyclic group defining the BB code is Z_l times Z_m.
        a_list and b_list are lists of group elements defining the BB code. Each group element x^alpha y^beta is represented as a list of two integers [alpha,beta].
        homomorphism_str must be "x", "y", "xy" or "x2y" depending on the choice of homomorphism
        """
        if homomorphism_str == "x":
            if self.l % 3 != 0:
                return False
            homomorphism = lambda group_element: group_element[0] % 3
        elif homomorphism_str == "y":
            if self.m % 3 != 0:
                return False
            homomorphism = lambda group_element: group_element[1] % 3
        elif homomorphism_str == "xy":
            if (self.l % 3 != 0) or (self.m % 3 != 0):
                return False
            homomorphism = (
                lambda group_element: (group_element[0] + group_element[1]) % 3
            )
        elif homomorphism_str == "x2y":
            if (self.l % 3 != 0) or (self.m % 3 != 0):
                return False
            homomorphism = (
                lambda group_element: (2 * group_element[0] + group_element[1]) % 3
            )
        else:
            raise ValueError(
                "What a rotten piece of luck old chap, homomorphism_str needs to be x, y, xy or x2y for me to continue!"
            )

        if all([self.l > 0, self.m > 0, len(self.a_list) == 3, len(self.b_list) == 3]):
            for c_list in [self.a_list, self.b_list]:
                homomorphism_list = [homomorphism(c_el) for c_el in c_list]
                if all([(homomorphism_list.count(i) == 1) for i in range(3)]):
                    continue
                else:
                    return False
            return True

    def prepare_for_standard_morphing(self, homomorphism_str):
        if homomorphism_str == "x":
            if self.l % 2 == 1:
                raise ValueError(
                    "The x homomorphism is incompatible with the chosen value of l"
                )
            homomorphism = lambda group_element: group_element[0] % 2
        elif homomorphism_str == "y":
            if self.m % 2 == 1:
                raise ValueError(
                    "The y homomorphism is incompatible with the chosen value of m"
                )
            homomorphism = lambda group_element: group_element[1] % 2
        elif homomorphism_str == "xy":
            if (self.l % 2 == 1) or (self.m % 2 == 1):
                raise ValueError(
                    "The xy homomorphism is incompatible with the chosen values of l and m"
                )
            homomorphism = (
                lambda group_element: (group_element[0] + group_element[1]) % 2
            )
        else:
            raise ValueError(
                "What a rotten piece of luck old chap, homomorphism_str needs to be x, y or xy for me to continue!"
            )

        if all([self.l > 0, self.m > 0, len(self.a_list) > 1, len(self.b_list) > 1]):
            for c_list in [self.a_list, self.b_list]:
                homomorphism_list = [homomorphism(c_el) for c_el in c_list]
                if (homomorphism_list.count(0) == 1) and (
                    homomorphism_list.count(1) == len(c_list) - 1
                ):
                    special_value = 0
                elif (homomorphism_list.count(1) == 1) and (
                    homomorphism_list.count(0) == len(c_list) - 1
                ):
                    special_value = 1
                else:
                    raise ValueError(
                        "The morphing specifications are not suitable for standard morphing!"
                    )
                to_swap = homomorphism_list.index(special_value)
                c_list[0], c_list[to_swap] = c_list[to_swap], c_list[0]
        else:
            raise ValueError(
                "The morphing specifications are not suitable for standard morphing!"
            )

    def prepare_for_three_round_morphing(self, homomorphism_str):
        if homomorphism_str == "x":
            if self.l % 3 != 0:
                return False
            homomorphism = lambda group_element: group_element[0] % 3
        elif homomorphism_str == "y":
            if self.m % 3 != 0:
                return False
            homomorphism = lambda group_element: group_element[1] % 3
        elif homomorphism_str == "xy":
            if (self.l % 3 != 0) or (self.m % 3 != 0):
                return False
            homomorphism = (
                lambda group_element: (group_element[0] + group_element[1]) % 3
            )
        elif homomorphism_str == "x2y":
            if (self.l % 3 != 0) or (self.m % 3 != 0):
                return False
            homomorphism = (
                lambda group_element: (2 * group_element[0] + group_element[1]) % 3
            )
        else:
            raise ValueError(
                "What a rotten piece of luck old chap, homomorphism_str needs to be x, y, xy or x2y for me to continue!"
            )

        if all([self.l > 0, self.m > 0, len(self.a_list) == 3, len(self.b_list) == 3]):
            for c_list in [self.a_list, self.b_list]:
                homomorphism_list = [homomorphism(c_el) for c_el in c_list]
                if all([(homomorphism_list.count(i) == 1) for i in range(3)]):
                    new_list = []
                    for i in range(3):
                        new_list.append(c_list[homomorphism_list.index(i)])
                    c_list[:] = new_list[:]
                else:
                    raise ValueError(
                        "The morphing specifications are not suitable for three-round morphing!"
                    )
        else:
            raise ValueError(
                "The morphing specifications are not suitable for three-round morphing!"
            )

    def parity_check_matrices(self):
        l = self.l
        m = self.m
        a_list = self.a_list
        b_list = self.b_list

        n_a = len(a_list)
        n_b = len(b_list)
        x_matrix = F2(np.diag(np.ones((l - 1)), 1).astype(int))
        x_matrix[l - 1, 0] = 1
        y_matrix = F2(np.diag(np.ones((m - 1)), 1).astype(int))
        y_matrix[m - 1, 0] = 1
        a_matrices = [
            np.kron(
                np.linalg.matrix_power(x_matrix, a[0]),
                np.linalg.matrix_power(y_matrix, a[1]),
            )
            for a in a_list
        ]
        b_matrices = [
            np.kron(
                np.linalg.matrix_power(x_matrix, b[0]),
                np.linalg.matrix_power(y_matrix, b[1]),
            )
            for b in b_list
        ]
        a_matrix = F2(np.zeros((l * m, l * m)).astype(int))
        b_matrix = F2(np.zeros((l * m, l * m)).astype(int))
        for i in range(n_a):
            a_matrix += a_matrices[i]
        for i in range(n_b):
            b_matrix += b_matrices[i]

        h = {
            "X": np.concatenate((a_matrix, b_matrix), axis=1),
            "Z": np.concatenate((b_matrix.T, a_matrix.T), axis=1),
        }

        return h

    def get_stabilisers_and_contraction_rounds(self, contraction_round_functions=None):
        l = self.l
        m = self.m
        a_list = self.a_list
        b_list = self.b_list

        h = self.parity_check_matrices()

        mid_cycle_stabilisers = {pauli: [] for pauli in ["X", "Z"]}
        stabiliser_contraction_rounds = (
            {pauli: [] for pauli in ["X", "Z"]}
            if contraction_round_functions != None
            else None
        )
        for i in range(l):
            for j in range(m):
                mid_cycle_stabilisers["X"].append(
                    np.nonzero(h["X"][m * i + j])[0].tolist()
                )
                mid_cycle_stabilisers["Z"].append(
                    np.nonzero(h["Z"][m * i + j])[0].tolist()
                )
                if contraction_round_functions != None:
                    x_round = contraction_round_functions["X"](i, j)
                    stabiliser_contraction_rounds["X"].append([x_round])
                    z_round = contraction_round_functions["Z"](i, j)
                    stabiliser_contraction_rounds["Z"].append([z_round])
        return mid_cycle_stabilisers, stabiliser_contraction_rounds

    def get_real_coordinates_list(self):
        coordinates_list = []
        # last two coordinates i,j represent the group element x^i y^j, first coordinate represents left (0) or right (1).
        for lr in range(2):
            for i in range(self.l):
                for j in range(self.m):
                    coordinates_list.append([lr, i, j])
        return coordinates_list

    def get_standard_morphing_specification(self, homomorphism_str):
        """
        Returns a MorphingSpecifications object representing the standard morphing circuit for a BB code (as in Table IV of arXiv:2407.16336). Works for any BB code satisfying valid_BB_morphing_parameters

        INPUTS

        l and m are positive integers such that the bicyclic group defining the BB code is Z_l times Z_m.
        a_list and b_list are lists of group elements defining the BB code. Each group element x^alpha y^beta is represented as a list of two integers [alpha,beta]. These must be provided in the correct order for the morphing circuit.
        homomorphism_str must be "x", "y" or "xy" depending on the choice of homomorphism (see Eq 1 of [ST24])
        """
        if homomorphism_str == "x":
            homomorphism = lambda group_element: group_element[0] % 2
        elif homomorphism_str == "y":
            homomorphism = lambda group_element: group_element[1] % 2
        elif homomorphism_str == "xy":
            homomorphism = (
                lambda group_element: (group_element[0] + group_element[1]) % 2
            )
        else:
            raise ValueError(
                "What a rotten piece of luck old chap, homomorphism_str needs to be x, y or xy for me to continue!"
            )

        if not self.valid_standard_morphing_parameters(homomorphism_str):
            raise ValueError(
                "The inputted midcycle parameters do not form a valid middle-out circuit."
            )

        old_a_list = deepcopy(self.a_list)
        old_b_list = deepcopy(self.b_list)

        self.prepare_for_standard_morphing(homomorphism_str)

        l = self.l
        m = self.m
        a_list = self.a_list
        b_list = self.b_list

        n_a = len(a_list)
        n_b = len(b_list)

        coordinates_list = self.get_real_coordinates_list()

        contraction_round_functions = {
            "X": lambda i, j: (homomorphism([i, j]) + homomorphism(a_list[0])) % 2,
            "Z": lambda i, j: (homomorphism([i, j]) + homomorphism(b_list[0]) + 1) % 2,
        }

        mid_cycle_stabilisers, stabiliser_contraction_rounds = (
            self.get_stabilisers_and_contraction_rounds(
                contraction_round_functions=contraction_round_functions
            )
        )

        gate_lists = [[[] for layer in range(max([n_a, n_b]))] for c_round in range(2)]
        measurement_lists = {
            pauli: [[] for c_round in range(2)] for pauli in ["X", "Z"]
        }

        for i in range(l):
            for j in range(m):
                round_g = homomorphism(
                    [i, j]
                )  # this is the round in which the left qubit labelled by [i,j] corresponds to the element g in Table IV
                round_h = (homomorphism([i, j]) + 1) % 2  # same but for h in Table IV
                this_left_qubit = coordinates_list.index([0, i, j])
                right_g_qubits = [
                    coordinates_list.index(
                        [
                            1,
                            (-a_list[0][0] + bel[0] + i) % l,
                            (-a_list[0][1] + bel[1] + j) % m,
                        ]
                    )
                    for bel in b_list
                ]
                right_h_qubits = [
                    coordinates_list.index(
                        [
                            1,
                            (-ael[0] + b_list[0][0] + i) % l,
                            (-ael[1] + b_list[0][1] + j) % m,
                        ]
                    )
                    for ael in a_list
                ]
                for k in range(1, n_b):
                    gate_lists[round_g][k - 1].append(
                        [this_left_qubit, right_g_qubits[k]]
                    )
                for k in range(1, n_a):
                    gate_lists[round_h][k - 1].append(
                        [right_h_qubits[k], this_left_qubit]
                    )
                gate_lists[round_g][-1].append([right_g_qubits[0], this_left_qubit])
                gate_lists[round_h][-1].append([this_left_qubit, right_h_qubits[0]])
                measurement_lists["X"][round_g].append(right_g_qubits[0])
                measurement_lists["Z"][round_h].append(right_h_qubits[0])

        complex_coordinates_list = [
            2 * coordinates[1] + 2j * coordinates[2] + (1 + 1j) * coordinates[0]
            for coordinates in coordinates_list
        ]

        specifications = MorphingSpecifications(
            mid_cycle_stabilisers,
            stabiliser_contraction_rounds,
            gate_lists,
            measurement_lists,
            coordinates_list=complex_coordinates_list,
        )

        self.a_list = old_a_list
        self.b_list = old_b_list

        return specifications

    def get_three_round_morphing_specification(self, homomorphism_str):
        """
        BUGS
        Not yet bug tested

        Returns a MorphingSpecifications object representing the three-round morphing circuit for a BB code. Works for any weight-(3+3) BB code satisfying valid_three_round_BB_morphing_parameters

        INPUTS

        l and m are positive integers such that the bicyclic group defining the BB code is Z_l times Z_m.
        a_list and b_list are length-three lists of group elements defining the BB code. Each group element x^alpha y^beta is represented as a list of two integers [alpha,beta]. These must be provided in the correct order for the morphing circuit such that homomorphism(a_list[i]) == i and homomorphism(b_list[i]) == i.
        homomorphism_str must be "x", "y", "xy" or "x2y" depending on the choice of homomorphism
        """
        if homomorphism_str == "x":
            homomorphism = lambda group_element: group_element[0] % 3
        elif homomorphism_str == "y":
            homomorphism = lambda group_element: group_element[1] % 3
        elif homomorphism_str == "xy":
            homomorphism = (
                lambda group_element: (group_element[0] + group_element[1]) % 3
            )
        elif homomorphism_str == "x2y":
            homomorphism = (
                lambda group_element: (2 * group_element[0] + group_element[1]) % 3
            )
        else:
            raise ValueError(
                "What a rotten piece of luck old chap, homomorphism_str needs to be x, y, xy or x2y for me to continue!"
            )

        old_a_list = deepcopy(self.a_list)
        old_b_list = deepcopy(self.b_list)

        self.prepare_for_three_round_morphing(homomorphism_str)

        l = self.l
        m = self.m
        a_list = self.a_list
        b_list = self.b_list

        if not self.valid_three_round_morphing_parameters(homomorphism_str):
            raise ValueError(
                "The inputted midcycle parameters do not form a valid middle-out circuit."
            )

        n_a = len(a_list)
        n_b = len(b_list)

        coordinates_list = self.get_real_coordinates_list()

        contraction_round_functions = {
            "X": lambda i, j: homomorphism([i, j]),
            "Z": lambda i, j: homomorphism([i, j]),
        }

        mid_cycle_stabilisers, stabiliser_contraction_rounds = (
            self.get_stabilisers_and_contraction_rounds(
                contraction_round_functions=contraction_round_functions
            )
        )

        gate_lists = [[[] for layer in range(3)] for c_round in range(3)]
        measurement_lists = {
            pauli: [[] for c_round in range(3)] for pauli in ["X", "Z"]
        }

        for i in range(l):
            for j in range(m):
                c_round = homomorphism([i, j])
                left_x_qubits = [
                    coordinates_list.index([0, (ael[0] + i) % l, (ael[1] + j) % m])
                    for ael in a_list
                ]
                right_x_qubits = [
                    coordinates_list.index([1, (bel[0] + i) % l, (bel[1] + j) % m])
                    for bel in b_list
                ]
                z_indices = [
                    (a_list[0][0] + b_list[0][0] + i) % l,
                    (a_list[0][1] + b_list[0][1] + j) % m,
                ]
                right_z_qubits = [
                    coordinates_list.index(
                        [1, (z_indices[0] - ael[0]) % l, (z_indices[1] - ael[1]) % m]
                    )
                    for ael in a_list
                ]
                # Note that right_z_qubits[0] = right_x_qubits[0]
                gate_lists[c_round][0].append([left_x_qubits[0], right_x_qubits[0]])
                gate_lists[c_round][0].append([left_x_qubits[1], right_x_qubits[2]])
                gate_lists[c_round][0].append([left_x_qubits[2], right_x_qubits[1]])
                gate_lists[c_round][1].append([left_x_qubits[0], left_x_qubits[1]])
                gate_lists[c_round][1].append([right_z_qubits[1], right_z_qubits[0]])
                gate_lists[c_round][2].append([left_x_qubits[0], left_x_qubits[2]])
                gate_lists[c_round][2].append([right_z_qubits[2], right_z_qubits[0]])
                measurement_lists["X"][c_round].append(left_x_qubits[0])
                measurement_lists["Z"][c_round].append(right_z_qubits[0])

        complex_coordinates_list = [
            2 * coordinates[1] + 2j * coordinates[2] + (1 + 1j) * coordinates[0]
            for coordinates in coordinates_list
        ]

        specifications = MorphingSpecifications(
            mid_cycle_stabilisers,
            stabiliser_contraction_rounds,
            gate_lists,
            measurement_lists,
            coordinates_list=complex_coordinates_list,
        )

        self.a_list = old_a_list
        self.b_list = old_b_list

        return specifications

    def get_swap2_morphing_specification(self, homomorphism_str):
        """
        BUGS
        NOT YET TESTED
        Also just quick theory double check that the homomorphism condition is indeed the same for this circuit even for |A|,|B| > 3.

        Returns a MorphingSpecifications object representing the standard morphing circuit for a BB code (as in Table VIII of arXiv:2407.16336). Works for any BB code satisfying valid_BB_morphing_parameters

        INPUTS

        l and m are positive integers such that the bicyclic group defining the BB code is Z_l times Z_m.
        a_list and b_list are lists of group elements defining the BB code. Each group element x^alpha y^beta is represented as a list of two integers [alpha,beta]. These must be provided in the correct order for the morphing circuit.
        homomorphism_str must be "x", "y" or "xy" depending on the choice of homomorphism (see Eq 1 of arXiv:2407.16336)
        """
        if homomorphism_str == "x":
            homomorphism = lambda group_element: group_element[0] % 2
        elif homomorphism_str == "y":
            homomorphism = lambda group_element: group_element[1] % 2
        elif homomorphism_str == "xy":
            homomorphism = (
                lambda group_element: (group_element[0] + group_element[1]) % 2
            )
        else:
            raise ValueError(
                "What a rotten piece of luck old chap, homomorphism_str needs to be x, y or xy for me to continue!"
            )

        if not self.valid_standard_morphing_parameters(homomorphism_str):
            raise ValueError(
                "The inputted midcycle parameters do not form a valid middle-out circuit."
            )

        old_a_list = deepcopy(self.a_list)
        old_b_list = deepcopy(self.b_list)

        self.prepare_for_standard_morphing(homomorphism_str)

        l = self.l
        m = self.m
        a_list = self.a_list
        b_list = self.b_list

        n_a = len(a_list)
        n_b = len(b_list)

        coordinates_list = self.get_real_coordinates_list()

        contraction_round_functions = {
            "X": lambda i, j: (homomorphism([i, j]) + homomorphism(a_list[0])) % 2,
            "Z": lambda i, j: (homomorphism([i, j]) + homomorphism(b_list[0]) + 1) % 2,
        }

        mid_cycle_stabilisers, stabiliser_contraction_rounds = (
            self.get_stabilisers_and_contraction_rounds(
                contraction_round_functions=contraction_round_functions
            )
        )

        gate_lists = [[[] for layer in range(max([n_a, n_b]))] for c_round in range(2)]
        measurement_lists = {
            pauli: [[] for c_round in range(2)] for pauli in ["X", "Z"]
        }

        for i in range(l):
            for j in range(m):
                round_g = homomorphism(
                    [i, j]
                )  # this is the round in which the left qubit labelled by [i,j] corresponds to the element g in Table IV
                round_h = (homomorphism([i, j]) + 1) % 2  # same but for h in Table IV
                this_left_qubit = coordinates_list.index([0, i, j])
                right_g_qubits = [
                    coordinates_list.index(
                        [
                            1,
                            (-a_list[0][0] + bel[0] + i) % l,
                            (-a_list[0][1] + bel[1] + j) % m,
                        ]
                    )
                    for bel in b_list
                ]
                right_h_qubits = [
                    coordinates_list.index(
                        [
                            1,
                            (-ael[0] + b_list[0][0] + i) % l,
                            (-ael[1] + b_list[0][1] + j) % m,
                        ]
                    )
                    for ael in a_list
                ]
                right_nucleus = coordinates_list.index(
                    [
                        1,
                        (-a_list[1][0] + b_list[1][0] + i) % l,
                        (-a_list[1][1] + b_list[1][1] + j) % m,
                    ]
                )
                for k in range(2, n_b):
                    gate_lists[round_g][k - 2].append(
                        [this_left_qubit, right_g_qubits[k]]
                    )
                for k in range(2, n_a):
                    gate_lists[round_h][k - 2].append(
                        [right_h_qubits[k], this_left_qubit]
                    )
                gate_lists[round_g][-2].append([right_g_qubits[1], this_left_qubit])
                gate_lists[round_h][-2].append([this_left_qubit, right_h_qubits[1]])
                gate_lists[round_g][-1].append([this_left_qubit, right_nucleus])
                gate_lists[round_h][-1].append([right_nucleus, this_left_qubit])
                measurement_lists["Z"][round_g].append(right_nucleus)
                measurement_lists["X"][round_h].append(right_nucleus)

        complex_coordinates_list = [
            2 * coordinates[1] + 2j * coordinates[2] + (1 + 1j) * coordinates[0]
            for coordinates in coordinates_list
        ]

        specifications = MorphingSpecifications(
            mid_cycle_stabilisers,
            stabiliser_contraction_rounds,
            gate_lists,
            measurement_lists,
            coordinates_list=complex_coordinates_list,
        )

        self.a_list = old_a_list
        self.b_list = old_b_list

        return specifications

    def get_IBM_XZ_memory_circuit(
        self,
        noise_setup,
        T,
        noise_model_generator=lambda noise_setup, qubit_dictionary: CircuitNoiseModel(
            noise_setup, qubit_dictionary
        ),
    ):
        """
        a bit messier than the other functions since it very closely follows my old code. But seems to work.
        """

        l = self.l
        m = self.m
        a_list = self.a_list
        b_list = self.b_list

        if T < 2:
            raise ValueError("T is too smol for the mol, make him bigger")

        if len(a_list) != 3 or len(b_list) != 3:
            raise Exception("a_list and b_list must have length 3")

        n_qubits = 4 * l * m

        coords_list = [
            (label, i, j)
            for label in ["L", "R", "X", "Z"]
            for i in range(l)
            for j in range(m)
        ]

        # round indices line up mod 7 with the rounds in Fig 5 of the IBM paper arxiv:2308.07915
        gate_lists = [[] for _ in range(8)]
        measure_lists = {pauli: [[] for _ in range(8)] for pauli in ["X", "Z"]}
        reset_lists = {pauli: [[] for _ in range(8)] for pauli in ["X", "Z"]}

        for i in range(l):
            for j in range(m):
                x_ancilla = coords_list.index(("X", i, j))
                z_ancilla = coords_list.index(("Z", i, j))

                left_x_qubits = [
                    coords_list.index(("L", (ael[0] + i) % l, (ael[1] + j) % m))
                    for ael in a_list
                ]
                right_x_qubits = [
                    coords_list.index(("R", (bel[0] + i) % l, (bel[1] + j) % m))
                    for bel in b_list
                ]
                left_z_qubits = [
                    coords_list.index(("L", (-bel[0] + i) % l, (-bel[1] + j) % m))
                    for bel in b_list
                ]
                right_z_qubits = [
                    coords_list.index(("R", (-ael[0] + i) % l, (-ael[1] + j) % m))
                    for ael in a_list
                ]

                measure_lists["X"][0].append(str(x_ancilla))
                reset_lists["X"][1].append(str(x_ancilla))
                measure_lists["Z"][7].append(str(z_ancilla))
                reset_lists["Z"][0].append(str(z_ancilla))

                gate_lists[1].append(str(right_z_qubits[0]))
                gate_lists[1].append(str(z_ancilla))
                gate_lists[2].append(str(right_z_qubits[2]))
                gate_lists[2].append(str(z_ancilla))
                gate_lists[3].append(str(left_z_qubits[0]))
                gate_lists[3].append(str(z_ancilla))
                gate_lists[4].append(str(left_z_qubits[1]))
                gate_lists[4].append(str(z_ancilla))
                gate_lists[5].append(str(left_z_qubits[2]))
                gate_lists[5].append(str(z_ancilla))
                gate_lists[6].append(str(right_z_qubits[1]))
                gate_lists[6].append(str(z_ancilla))

                gate_lists[2].append(str(x_ancilla))
                gate_lists[2].append(str(left_x_qubits[1]))
                gate_lists[3].append(str(x_ancilla))
                gate_lists[3].append(str(right_x_qubits[1]))
                gate_lists[4].append(str(x_ancilla))
                gate_lists[4].append(str(right_x_qubits[0]))
                gate_lists[5].append(str(x_ancilla))
                gate_lists[5].append(str(right_x_qubits[2]))
                gate_lists[6].append(str(x_ancilla))
                gate_lists[6].append(str(left_x_qubits[0]))
                gate_lists[7].append(str(x_ancilla))
                gate_lists[7].append(str(left_x_qubits[2]))

        x_matrix = F2(np.diag(np.ones((l - 1)), 1).astype(int))
        x_matrix[l - 1, 0] = 1
        y_matrix = F2(np.diag(np.ones((m - 1)), 1).astype(int))
        y_matrix[m - 1, 0] = 1
        amatrices = [
            np.kron(
                np.linalg.matrix_power(x_matrix, ael[0]),
                np.linalg.matrix_power(y_matrix, ael[1]),
            )
            for ael in a_list
        ]
        bmatrices = [
            np.kron(
                np.linalg.matrix_power(x_matrix, bel[0]),
                np.linalg.matrix_power(y_matrix, bel[1]),
            )
            for bel in b_list
        ]
        amatrix = F2(np.zeros((l * m, l * m)).astype(int))
        bmatrix = F2(np.zeros((l * m, l * m)).astype(int))
        for i in range(len(a_list)):
            amatrix += amatrices[i]
            bmatrix += bmatrices[i]
        h = {
            "X": np.concatenate((amatrix, bmatrix), axis=1),
            "Z": np.concatenate((bmatrix.T, amatrix.T), axis=1),
        }

        stabilisers = {pauli: [] for pauli in ["X", "Z"]}
        for i in range(l):
            for j in range(m):
                stabilisers["X"].append(np.nonzero(h["X"][m * i + j])[0].tolist())
                stabilisers["Z"].append(np.nonzero(h["Z"][m * i + j])[0].tolist())

        hker = {pauli: h[pauli].null_space() for pauli in ["X", "Z"]}
        stabs_to_logicals = {pauli: h[pauli].row_reduce() for pauli in ["X", "Z"]}
        observables = {pauli: [] for pauli in ["X", "Z"]}

        for pauli in ["X", "Z"]:
            other_pauli = "Z" if pauli == "X" else "X"
            for i in range(hker[other_pauli].shape[0]):
                new_vecs = np.concatenate(
                    (stabs_to_logicals[pauli], [hker[other_pauli][i]]), axis=0
                )
                if np.linalg.matrix_rank(new_vecs) != np.linalg.matrix_rank(
                    stabs_to_logicals[pauli]
                ):
                    stabs_to_logicals[pauli] = new_vecs
                    observables[pauli].append(
                        np.nonzero(hker[other_pauli][i])[0].tolist()
                    )
        n_logicals = len(observables["X"])

        qubit_dictionary = {str(i): i for i in range(n_qubits)}
        all_qubits_dictionary = {str(i): i for i in range(n_qubits + n_logicals)}
        noise_model = noise_model_generator(noise_setup, qubit_dictionary)
        noiseless_model = NoiselessModel(all_qubits_dictionary)
        tick = noiseless_model.tick()

        labels = ["L", "R", "X", "Z"]
        qubit_strs = {
            labels[j]: [str(i) for i in range(j * l * m, (j + 1) * l * m)]
            for j in range(4)
        }

        offsets = {"L": 0, "R": 1 + 1j, "X": 1, "Z": 1j}
        coordinates_list = []
        for i in range(n_qubits):
            offset = offsets[coords_list[i][0]]
            coordinates_list.append(
                round(2 * coords_list[i][1] + np.real(offset))
                + 1j * round(2 * coords_list[i][2] + np.imag(offset))
            )

        circ = coordinate_circuit(coordinates_list, n_extra_qubits=n_logicals)

        # prepare an error-free Bell state between each logical qubit and each perfect dummy qubit

        circ += noiseless_model.reset_x(
            [str(i) for i in range(4 * l * m, 4 * l * m + n_logicals)]
        )
        circ += noiseless_model.reset(qubit_strs["L"] + qubit_strs["R"])
        circ += tick

        for i in range(n_logicals):
            for qubit in observables["X"][i]:
                circ += noiseless_model.cnot([str(n_qubits + i), str(qubit)])
            circ += tick

        projection_str = ""
        for pauli in ["Z", "X"]:
            for stabiliser in stabilisers[pauli]:
                projection_str += "\nTICK\n"
                projection_str += "MPP"
                projection_str += " "
                for i in range(len(stabiliser)):
                    if i > 0:
                        projection_str += "*"
                    projection_str += pauli + str(stabiliser[i])
        circ += stim.Circuit(projection_str)
        circ += tick

        # perfect state prepared, now run the T rounds of noisy QEC.

        circ += noise_model.reset(reset_lists["Z"][0])
        circ += noise_model.idle(
            [str(i) for i in range(4 * l * m) if str(i) not in reset_lists["Z"][0]]
        )

        for i in [1, 2, 3, 4, 5, 6, 7, 0]:
            circ += tick
            circ += noise_model.cnot(gate_lists[i])
            circ += noise_model.measure_x(measure_lists["X"][i])
            circ += noise_model.measure(measure_lists["Z"][i])
            circ += noise_model.reset_x(reset_lists["X"][i])
            circ += noise_model.reset(reset_lists["Z"][i])
            circ += noise_model.idle(
                [
                    str(j)
                    for j in range(4 * l * m)
                    if str(j)
                    not in (
                        gate_lists[i]
                        + measure_lists["X"][i]
                        + measure_lists["Z"][i]
                        + reset_lists["X"][i]
                        + reset_lists["Z"][i]
                    )
                ]
            )
        for i in range(l * m):
            circ.append(
                "DETECTOR",
                [stim.target_rec(-4 * l * m + i), stim.target_rec(-2 * l * m + i)],
                (0, 0, 0, 3),
            )
        for i in range(l * m, 2 * l * m):
            circ.append(
                "DETECTOR",
                [stim.target_rec(-4 * l * m + i), stim.target_rec(-2 * l * m + i)],
                (0, 0, 0, 0),
            )

        if T > 2:
            repeat_circ = stim.Circuit()
            for i in [1, 2, 3, 4, 5, 6, 7, 0]:
                repeat_circ += tick
                repeat_circ += noise_model.cnot(gate_lists[i])
                repeat_circ += noise_model.measure_x(measure_lists["X"][i])
                repeat_circ += noise_model.measure(measure_lists["Z"][i])
                repeat_circ += noise_model.reset_x(reset_lists["X"][i])
                repeat_circ += noise_model.reset(reset_lists["Z"][i])
                repeat_circ += noise_model.idle(
                    [
                        str(j)
                        for j in range(4 * l * m)
                        if str(j)
                        not in (
                            gate_lists[i]
                            + measure_lists["X"][i]
                            + measure_lists["Z"][i]
                            + reset_lists["X"][i]
                            + reset_lists["Z"][i]
                        )
                    ]
                )
            for i in range(l * m):
                repeat_circ.append(
                    "DETECTOR",
                    [stim.target_rec(-4 * l * m + i), stim.target_rec(-2 * l * m + i)],
                    (0, 0, 0, 3),
                )
            for i in range(l * m, 2 * l * m):
                repeat_circ.append(
                    "DETECTOR",
                    [stim.target_rec(-4 * l * m + i), stim.target_rec(-2 * l * m + i)],
                    (0, 0, 0, 0),
                )
            circ += repeat_circ * (T - 2)

        for i in [1, 2, 3, 4, 5, 6, 7]:
            circ += tick
            circ += noise_model.cnot(gate_lists[i])
            circ += noise_model.measure_x(measure_lists["X"][i])
            circ += noise_model.measure(measure_lists["Z"][i])
            circ += noise_model.reset_x(reset_lists["X"][i])
            circ += noise_model.reset(reset_lists["Z"][i])
            circ += noise_model.idle(
                [
                    str(j)
                    for j in range(4 * l * m)
                    if str(j)
                    not in (
                        gate_lists[i]
                        + measure_lists["X"][i]
                        + measure_lists["Z"][i]
                        + reset_lists["X"][i]
                        + reset_lists["Z"][i]
                    )
                ]
            )

        # no idling noise in this round to transition to the noise-free Bell measurements
        circ += tick
        circ += noise_model.cnot(gate_lists[0])
        circ += noise_model.measure_x(measure_lists["X"][0])
        circ += noise_model.measure(measure_lists["Z"][0])
        circ += noise_model.reset_x(reset_lists["X"][0])
        circ += noise_model.reset(reset_lists["Z"][0])

        for i in range(l * m):
            circ.append(
                "DETECTOR",
                [stim.target_rec(-4 * l * m + i), stim.target_rec(-2 * l * m + i)],
                (0, 0, 0, 3),
            )
        for i in range(l * m, 2 * l * m):
            circ.append(
                "DETECTOR",
                [stim.target_rec(-4 * l * m + i), stim.target_rec(-2 * l * m + i)],
                (0, 0, 0, 0),
            )

        circ += stim.Circuit(projection_str)
        for i in range(l * m):
            circ.append(
                "DETECTOR",
                [stim.target_rec(-4 * l * m + i), stim.target_rec(-2 * l * m + i)],
                (0, 0, 0, 3),
            )
        for i in range(l * m, 2 * l * m):
            circ.append(
                "DETECTOR",
                [stim.target_rec(-4 * l * m + i), stim.target_rec(-2 * l * m + i)],
                (0, 0, 0, 0),
            )

        # perfect measurement of the observables
        circ += tick
        for i in range(n_logicals):
            for qubit in observables["X"][i]:
                circ += noiseless_model.cnot([str(n_qubits + i), str(qubit)])
            circ += tick

        circ += noiseless_model.measure_x(
            [str(i) for i in range(4 * l * m, 4 * l * m + n_logicals)]
        )
        circ += noiseless_model.measure([str(i) for i in range(2 * l * m)])

        for i in range(n_logicals):
            circ.append(
                "OBSERVABLE_INCLUDE", [stim.target_rec(-2 * l * m - n_logicals + i)], i
            )
        for i in range(n_logicals):
            circ.append(
                "OBSERVABLE_INCLUDE",
                [stim.target_rec(-2 * l * m + qubit) for qubit in observables["Z"][i]],
                n_logicals + i,
            )
        return circ