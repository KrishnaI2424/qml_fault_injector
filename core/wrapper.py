import pennylane as qml
import functools
from .recorder import ResultRecorder

class FaultInjectionWrapper:
    """
    Intercepts a QNode at call time, injects faults, and records results.
    Modelled after XALT's LD_PRELOAD: transparent to the model definition,
    active only at execution boundaries.
    """

    def __init__(self, qnode, fault_config, recorder):
        self.qnode = qnode
        self.fault_config = fault_config
        self.recorder = recorder
        functools.update_wrapper(self, qnode)  # Preserve QNode metadata

    def __call__(self, weights, *args, **kwargs):
      faulted_weights = self.fault_config.apply_classical_faults(weights)
      active_qnode    = self.fault_config.wrap_circuit(self.qnode)
      result          = active_qnode(faulted_weights, *args, **kwargs)
      self.recorder.log(self.fault_config, weights, faulted_weights, result)
      return result

def inject(qnode, fault_config, recorder):
    """Entry point — mirrors how XALT wraps execution transparently."""
    return FaultInjectionWrapper(qnode, fault_config, recorder)
