import numpy as np
import pennylane as qml
from dataclasses import dataclass
from typing import List, Optional
import sys, random, itertools

@dataclass
class FaultConfig:
    name: str = "baseline"
    bit_flip_p: float = 0.0
    phase_flip_p: float=0.0
    target_wires: Optional[List[int]] = None

    mode: str="exact"
    n_trials: int=1000
    device_name: str="default.mixed"
    seed:        Optional[int] = None

    def apply_classical_faults(self, weights):
        """No classical faults in this minimal version — pass through cleanly."""
        return weights.copy()

    def wrap_circuit(self, qnode):
        """Return a callable that mimics qnode(weights, x) with noise applied.

        In "exact" mode this returns a qml.QNode.
        In "stochastic" mode this returns a plain Python function that
        averages over n_trials calls to a noiseless QNode.
        """
        target_wires = self.target_wires if self.target_wires is not None else [0]
        n_wires      = len(qnode.device.wires)

        # Pull state_preparation and layer from the user's model module
        mod          = sys.modules[qnode.func.__module__]
        _state_prep  = getattr(mod, "state_preparation")
        _layer       = getattr(mod, "layer")

        if self.mode == "exact":
            return self._wrap_exact(_state_prep, _layer, target_wires, n_wires)
        elif self.mode == "stochastic":
            return self._wrap_stochastic(_state_prep, _layer, target_wires, n_wires)
        else:
            raise ValueError(f"Unknown mode: {self.mode!r}. Use 'exact' or 'stochastic'.")

    def _wrap_stochastic(self, _state_prep, _layer, target_wires, n_wires):
        # Pure-state device — supports lightning.gpu, lightning.qubit,
        # default.qubit, etc.  No noise channels are queued, only X/Z gates.
        pure_dev     = qml.device(self.device_name, wires=n_wires)
        bit_flip_p   = self.bit_flip_p
        phase_flip_p = self.phase_flip_p
        n_trials     = self.n_trials

        # Build a list of (wire, gate_class, p) tuples for active channels.
        # Each entry is one independent Pauli channel to enumerate over.
        events = []
        for wire in target_wires:
            if bit_flip_p > 0.0:
                events.append((wire, qml.PauliX, bit_flip_p))
            if phase_flip_p > 0.0:
                events.append((wire, qml.PauliZ, phase_flip_p))

        n_events   = len(events)
        n_patterns = 2 ** n_events

        # No active faults — return a clean QNode (one call, no averaging).
        if n_events == 0:
            def clean_func(weights, x):
                _state_prep(x)
                for layer_weights in weights:
                    _layer(layer_weights)
                return qml.expval(qml.PauliZ(0))
            return qml.QNode(clean_func, pure_dev, diff_method=None)

        # ── Pauli enumeration: exact result with 2^n_events circuit calls ──
        # Cheaper than sampling when 2^n_events <= n_trials, and gives the
        # exact density-matrix expectation with zero statistical noise.
        if n_patterns <= n_trials:
            # Pre-build one QNode per pattern.  Each pattern is a tuple of
            # 0/1 indicating whether each event fires.
            patterns = []
            for bits in itertools.product([0, 1], repeat=n_events):
                applied = []
                weight  = 1.0
                for fire, (wire, gate_cls, p) in zip(bits, events):
                    if fire:
                        applied.append((wire, gate_cls))
                        weight *= p
                    else:
                        weight *= (1 - p)

                # Bind `applied` into the closure via a default argument so
                # each QNode keeps its own pattern (avoids the late-binding
                # bug where every QNode would see the last pattern).
                def make_pattern_func(applied=applied):
                    def faulted(weights, x):
                        _state_prep(x)
                        for wire, gate_cls in applied:
                            gate_cls(wires=wire)
                        for layer_weights in weights:
                            _layer(layer_weights)
                        return qml.expval(qml.PauliZ(0))
                    return faulted

                qn = qml.QNode(make_pattern_func(), pure_dev, diff_method=None)
                patterns.append((qn, weight))

            def enumerated_call(weights, x):
                return sum(w * qn(weights, x) for qn, w in patterns)

            return enumerated_call

        # ── Sampling fallback: random Pauli insertion, averaged over trials ──
        # Used when 2^n_events > n_trials makes enumeration too expensive.
        if self.seed is not None:
            random.seed(self.seed)

        def faulted_func(weights, x):
            _state_prep(x)
            for wire, gate_cls, p in events:
                if random.random() < p:
                    gate_cls(wires=wire)
            for layer_weights in weights:
                _layer(layer_weights)
            return qml.expval(qml.PauliZ(0))

        single_trial_qnode = qml.QNode(faulted_func, pure_dev, diff_method=None)

        def averaged_call(weights, x):
            total = 0.0
            for _ in range(n_trials):
                total += single_trial_qnode(weights, x)
            return total / n_trials
        return averaged_call

    def _wrap_exact(self, _state_prep, _layer, target_wires, n_wires):
        mixed_dev    = qml.device("default.mixed", wires=n_wires)
        bit_flip_p   = self.bit_flip_p
        phase_flip_p = self.phase_flip_p

        def faulted_func(weights, x):
            _state_prep(x)
            for wire in target_wires:
                if bit_flip_p   > 0.0: qml.BitFlip(bit_flip_p,   wires=wire)
                if phase_flip_p > 0.0: qml.PhaseFlip(phase_flip_p, wires=wire)
            for layer_weights in weights:
                _layer(layer_weights)
            return qml.expval(qml.PauliZ(0))

        return qml.QNode(faulted_func, mixed_dev, diff_method=None)
