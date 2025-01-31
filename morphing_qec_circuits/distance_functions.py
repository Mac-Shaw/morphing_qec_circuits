import numpy as np
from copy import deepcopy
import galois  # for matrix calculations over a finite field

import gurobipy as gp
from gurobipy import GRB

F2 = galois.GF(2)


def dx(hs, quiet=False):
    n_qubits = hs["X"].shape[1]
    if 0 in hs["X"].shape:
        n_qubits = hs["Z"].shape[1]
    n_logicals = (
        n_qubits - np.linalg.matrix_rank(hs["X"]) - np.linalg.matrix_rank(hs["Z"])
    )
    n_z_stabs = hs["Z"].shape[0]

    kerhx = hs["X"].null_space()
    if 0 in hs["X"].shape:
        kerhx = F2(np.eye(n_qubits).astype(int))
    temp = hs["Z"].row_reduce()
    imhzt = temp[temp.any((1))].copy()
    if 0 in imhzt.shape:
        z_logicals = kerhx
    else:
        stabs_to_logicals = imhzt.copy()
        z_logicals = F2([[]])
        for i in range(kerhx.shape[0]):
            new_vecs = np.append(stabs_to_logicals, [kerhx[i]], axis=0)
            if np.linalg.matrix_rank(new_vecs) != np.linalg.matrix_rank(
                stabs_to_logicals
            ):
                stabs_to_logicals = new_vecs.copy()
                if 0 in z_logicals.shape:
                    z_logicals = np.array([kerhx[i]]).copy()
                else:
                    z_logicals = np.append(z_logicals, [kerhx[i]], axis=0)
    dist = np.inf
    hz = np.array(hs["Z"]).astype(np.int32)
    zero_array = np.zeros((n_z_stabs)).astype(np.int32)
    for i in range(z_logicals.shape[0]):
        m = gp.Model()
        if quiet:
            m.Params.OutputFlag = 0
            m.Params.LogToConsole = 0
        var = m.addMVar(shape=n_qubits, vtype=GRB.BINARY, name="var")
        if not (0 in hz.shape):
            dummy1 = m.addMVar(shape=n_z_stabs, vtype=GRB.INTEGER, name="dummy1")
            stab_constraint = m.addConstr(
                hz @ var - 2 * dummy1 == zero_array, "stab_constraint"
            )
        dummy2 = m.addVar(vtype=GRB.INTEGER, name="dummy2")
        logical_constraint = m.addConstr(
            z_logicals[i] @ var - 2 * dummy2 == 1, "logical_constraint"
        )
        m.setObjective(var.sum(), GRB.MINIMIZE)
        m.optimize()
        if m.ObjVal < dist:
            dist = m.ObjVal
    return round(dist)


def dz(hs, quiet=False):
    swapped_hs = {"X": hs["Z"], "Z": hs["X"]}
    return dx(swapped_hs, quiet=quiet)


def distance(hs, check_x_and_z=True, separate_x_and_z=False, quiet=False):
    x_distance = dx(hs, quiet=quiet)
    if check_x_and_z:
        z_distance = dz(hs, quiet=quiet)
        if separate_x_and_z:
            return {"X": x_distance, "Z": z_distance}
        else:
            return min(x_distance, z_distance)
    else:
        return x_distance


def nkd_params(hs, check_x_and_z=True, separate_x_and_z=False, quiet=False):
    n_qubits = hs["X"].shape[1]
    if 0 in hs["X"].shape:
        n_qubits = hs["Z"].shape[1]
    n_logicals = (
        n_qubits - np.linalg.matrix_rank(hs["X"]) - np.linalg.matrix_rank(hs["Z"])
    )
    if n_logicals <= 10**(-10):
        return (n_qubits, n_logicals, 0)
    return (
        n_qubits,
        n_logicals,
        distance(
            hs,
            check_x_and_z=check_x_and_z,
            separate_x_and_z=separate_x_and_z,
            quiet=quiet,
        ),
    )
