import numpy as np
import pennylane as qml
from dataclasses import dataclass
from typing import List, Optional
import sys
@dataclass
class FaultConfig:
    name: str = "baseline"
    bit_flip_p: float = 0.0
    phase_flip_p: float=0.0
    target_wires: Optional[List[int]] = None

    def apply_classical_faults(self, weights):
        """No classical faults in this minimal version — pass through cleanly."""
        return weights.copy()

    def wrap_circuit(self, qnode):

        original_func = qnode.func
        target_wires = self.target_wires if self.target_wires is not None else [0]
        bit_flip_p = self.bit_flip_p
        phase_flip_p=self.phase_flip_p
        n_wires=len(qnode.device.wires)
        mixed_dev = qml.device("default.mixed", wires=n_wires) 

        mod          = sys.modules[qnode.func.__module__]
        _state_prep  = getattr(mod, "state_preparation")
        _layer       = getattr(mod, "layer")


        def faulted_func(weights, x):   
          _state_prep(x)


          for wire in target_wires:
            if bit_flip_p > 0.0: qml.BitFlip(bit_flip_p, wires=wire)
            if phase_flip_p >0.0: qml.PhaseFlip(phase_flip_p, wires=wire)
          
          for layer_weights in weights:
            _layer(layer_weights)
          return qml.expval(qml.PauliZ(0))
        return qml.QNode(faulted_func, mixed_dev, diff_method=None)
