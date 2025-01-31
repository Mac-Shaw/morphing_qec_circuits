import stim
from surface_sim.models import NoiselessModel
from surface_sim.setup import Setup


class DistanceNoiseModel(NoiselessModel):
    def __init__(
        self, setup: Setup, qubit_inds: dict[str, int], noise_basis: str
    ) -> None:
        super().__init__(qubit_inds)
        self._setup = setup
        if noise_basis in ["X", "Z"]:
            self._noise_basis = noise_basis
        else:
            raise Exception(
                '''Noise_basis should be X or Z.
                (If your code/circuit is not CSS, then you should use a different noise model when calculating the circuit-level distance)'''
            )

    def tick(self) -> stim.Circuit:
        circ = stim.Circuit()
        circ.append("TICK")
        for qubit, ind in self._qubit_inds.items():
            prob = self.param("sq_error_prob", qubit)
            circ.append(self._noise_basis + "_ERROR", [ind], [prob])
        return circ
