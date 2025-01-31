import numpy as np
from copy import deepcopy
import galois  # for matrix calculations over a finite field
from .bb_params import BBParams
from .morphing_specifications import MorphingSpecifications

F2 = galois.GF(2)


class ToricCodeParams(BBParams):
    def __init__(self, d):
        self.l = d
        self.m = d
        self.a_list = [[0, 0], [1, 0]]
        self.b_list = [[0, 0], [0, d - 1]]

    def get_toric_standard_morphing_specification(self):
        return self.get_standard_morphing_specification("xy")

    def get_toric_scrambled_morphing_specification(
        self, scrambled_offsets={"X": [-3, -1], "Z": [-1, -3]}
    ):

        homomorphism = lambda group_el: (group_el[0] + group_el[1]) % 2
        specifications = self.get_standard_morphing_specification("xy")

        if self.l % 2 == 1:
            raise ValueError("What an oddity! d must be even to proceed.")

        check_mod = {}
        for pauli in ["X", "Z"]:
            if len(scrambled_offsets[pauli]) != 2:
                raise ValueError(
                    "scrambled_offsets should be a dictionary with keys X and Z and values that are length two lists of odd integers."
                )
            for el in scrambled_offsets[pauli]:
                if (type(el) != int) or (el % 2 != 1):
                    raise ValueError(
                        "scrambled_offsets should be a dictionary with keys X and Z and values that are length two lists of odd integers."
                    )
            check_mod[pauli] = (
                (scrambled_offsets[pauli][0] + scrambled_offsets[pauli][1]) % 4
            ) // 2
        if check_mod["X"] != check_mod["Z"]:
            print("the X and Z scrambled_offsets are overlapping")

        rescaled_offsets = {}
        for pauli in ["X", "Z"]:
            rescaled_offsets[pauli] = [
                (scrambled_offsets[pauli][i] - 1) // 2 for i in range(2)
            ]

        real_coordinates_list = [
            [round(z.real) % 2, round(z.real) // 2, round(z.imag) // 2]
            for z in specifications.coordinates_list
        ]

        gates_to_append = [[] for _ in range(2)]
        for i in range(self.l):
            for j in range(self.m):
                x_round = (homomorphism([i, j]) + 1) % 2
                z_round = (
                    homomorphism([i, j])
                    + homomorphism(self.a_list[0])
                    + homomorphism(self.b_list[0])
                ) % 2
                left_x_qubits_1 = real_coordinates_list.index(
                    [
                        0,
                        (self.a_list[1][0] + i) % self.l,
                        (self.a_list[1][1] + j) % self.m,
                    ]
                )
                left_z_qubits_1 = real_coordinates_list.index(
                    [
                        0,
                        (-self.b_list[1][0] + i) % self.l,
                        (-self.b_list[1][1] + j) % self.m,
                    ]
                )
                x_cooked = real_coordinates_list.index(
                    [
                        1,
                        (rescaled_offsets["X"][0] + self.a_list[1][0] + i) % self.l,
                        (rescaled_offsets["X"][1] + self.a_list[1][1] + j) % self.m,
                    ]
                )
                z_cooked = real_coordinates_list.index(
                    [
                        1,
                        (rescaled_offsets["Z"][0] - self.b_list[1][0] + i) % self.l,
                        (rescaled_offsets["Z"][1] - self.b_list[1][1] + j) % self.m,
                    ]
                )
                gates_to_append[x_round].append([x_cooked, left_x_qubits_1])
                gates_to_append[z_round].append([left_z_qubits_1, z_cooked])

        for i in range(2):
            specifications.gate_lists[i].append(gates_to_append[i])

        return specifications

    def get_toric_hex_scrambled_morphing_specification(self):
        specifications = self.get_standard_morphing_specification("xy")

        gates_to_append = [[[] for _ in range(2)] for _ in range(2)]
        for contraction_round in range(2):
            for cnot_round in range(2):
                for gate in specifications.gate_lists[contraction_round][cnot_round]:
                    gates_to_append[contraction_round][cnot_round].append(
                        [gate[1], gate[0]]
                    )

        for contraction_round in range(2):
            for cnot_round in range(2):
                specifications.gate_lists[contraction_round].append(
                    gates_to_append[contraction_round][cnot_round]
                )

        return specifications
