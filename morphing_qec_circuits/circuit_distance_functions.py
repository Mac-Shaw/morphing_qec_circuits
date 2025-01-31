import numpy as np
import stim
import random
from collections import Counter
from .circuit_functions import filter_detectors


def include_observable_as_detector(dem, observables_to_include):
    new_detector_index = dem.num_detectors
    new_dem = stim.DetectorErrorModel()
    for deminstruction in dem:
        if deminstruction.type == "error":
            observable_flips = [
                targ.val
                for targ in deminstruction.targets_copy()
                if targ.is_logical_observable_id()
            ]
            to_include = False
            for observable_flip in observable_flips:
                if observables_to_include[observable_flip]:
                    to_include = not to_include
            new_targets = [
                targ
                for targ in deminstruction.targets_copy()
                if targ.is_relative_detector_id()
            ]
            if to_include:
                new_targets.append(stim.target_relative_detector_id(new_detector_index))
            new_dem.append(
                stim.DemInstruction(
                    deminstruction.type, deminstruction.args_copy(), new_targets
                )
            )
        else:
            new_dem.append(deminstruction)
    return new_dem


def circuit_distance_ub(
    circuit, initialise_decoder, basis="both", n_shots=None, output_statistics=False, output_corrections=False
):
    if basis == "both":
        dem = circuit.detector_error_model().flattened()
        k = dem.num_observables
    elif basis in ["X", "Z"]:
        dem = filter_detectors(
            circuit.detector_error_model(),
            coordinates_condition=(lambda coordinates: coordinates[3] == {"X": 0, "Z": 3}[basis]),
        ).flattened()
        k = dem.num_observables // 2
    else:
        raise Exception("basis must be X, Z, or both")
    n_detectors = dem.num_detectors

    random_shots = True
    if n_shots == None:
        n_shots = 2**k - 1
        random_shots = False
    elif n_shots >= 2**k - 1:
        n_shots = 2**k - 1
        random_shots = False

    d_estimates = [0 for i in range(n_shots)]
    if output_corrections:
        error_corrections = []
    for i in range(1, n_shots + 1):
        if random_shots:
            rand_int = random.randrange(1, 2**k)
            observables_to_include = [bool(int(x)) for x in bin(rand_int)[2:].zfill(k)]
        else:
            observables_to_include = [bool(int(x)) for x in bin(i)[2:].zfill(k)]
        if basis == "X":
            observables_to_include += [False] * k
        elif basis == "Z":
            observables_to_include = ([False] * k) + observables_to_include
        new_dem = include_observable_as_detector(dem, observables_to_include)
        decoder = initialise_decoder(new_dem)
        defects = np.array([False] * n_detectors + [True])
        logical_fault_prediction = decoder.decode_to_faults_array(defects)
        if output_corrections:
            error_correction = [dem[i] for i, x in enumerate(logical_fault_prediction) if x == 1]
            error_corrections.append(error_correction)
        d_estimates[i - 1] = sum(logical_fault_prediction)

    d_estimate = min(d_estimates)
    if output_statistics:
        to_output = Counter(d_estimates)
    else:
        to_output = d_estimate

    if output_corrections:
        return to_output, dem, error_corrections
    else:
        return to_output
