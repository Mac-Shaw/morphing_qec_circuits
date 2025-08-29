from surface_sim.setup import Setup

def SI1000_variable_measurement_setup(p, x):
    setup = SI1000VariableMeasurement()
    setup.set_var_param("prob", p)
    setup.set_var_param("meas_factor", x)
    return setup

class SI1000VariableMeasurement(Setup):
    def __init__(self) -> None:
        """Initialises a ``Setup`` class for the modification of the SI1000 circuit-level noise
        as intended for use in the More Morphing Circuits paper. This includes a parameter to
        scale the measurement error probability up or down compared to the SI1000 noise model.
        
        SI1000 circuit-level noise is described in:
        C. Gidney, M. Newman, A. Fowler, and M. Broughton.
        A Fault-Tolerant honeycomb memory. Quantum, 5:605, Dec. 2021.

        **IMPORTANT**

        1. It should be loaded with the ``SI1000NoiseModel`` model. It should not be loaded
        with ``CircuitNoiseModel`` because the noise model stacks noise channels
        for qubits that are not being measured on top of their respective
        noise gate channels (e.g. idling).

        2. This noise model assumes that qubits are reset after measurements.
        In this sense, it does not add classical measurement errors (also known as
        assignment errors). It also assumes that ``model.tick()`` is called
        in-between gate layers.

        3. It contains two variable parameters ``"prob"`` and ``"meas_factor"`` that must be set before
        building any circuit.
        """
        setup_dict = dict(
            name="SI1000 noise setup with variable measurement error rate",
            description="Setup for the SI1000 noise model that can be used for any distance, with variable measurement noise.",
            setup=[
                dict(
                    sq_error_prob="{prob} / 10",
                    tq_error_prob="{prob}",
                    meas_error_prob="{prob} * {meas_factor} * 5",
                    reset_error_prob="{prob} * 2",
                    idle_error_prob="{prob} / 10",
                    extra_idle_meas_or_reset_error_prob="{prob} * 2",
                    assign_error_flag=False,
                    assign_error_prob=0,
                ),
            ],
        )
        super().__init__(setup_dict)
        return
