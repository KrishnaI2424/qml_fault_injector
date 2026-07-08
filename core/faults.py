import numpy as np
import pennylane as qml
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable
import itertools
import warnings


# Map each logical fault channel to (exact density-matrix op, stochastic Pauli op).
# Adding a new channel here is the only change needed to support it everywhere.
_CHANNELS = {
    "bit_flip":   (qml.BitFlip,   qml.PauliX),
    "phase_flip": (qml.PhaseFlip, qml.PauliZ),
}


@dataclass
class FaultConfig:
    """Configuration for a fault-injection run.

    This version is model-agnostic: it injects faults into *any* QNode by
    rewriting the quantum tape that the QNode produces, rather than by
    reconstructing the circuit from assumed helper functions.
    """
    name: str = "baseline"
    bit_flip_p: float = 0.0
    phase_flip_p: float = 0.0
    target_wires: Optional[List[int]] = None

    mode: str = "exact"               # "exact" | "stochastic"
    n_trials: int = 1000
    device_name: str = "default.qubit"   # device used for the stochastic path
    seed: Optional[int] = None

    # Where to insert faults in the operation stream:
    #   None        -> at the end, just before measurement (readout fault)
    #   int k       -> after the k-th operation (e.g. k = #state-prep ops
    #                  reproduces the "fault on the encoded input" behaviour
    #                  of the original implementation)
    insert_after: Optional[int] = None

    def _active_channels(self) -> Dict[str, float]:
        return {
            name: p for name, p in (
                ("bit_flip",   self.bit_flip_p),
                ("phase_flip", self.phase_flip_p),
            ) if p > 0.0
        }

    def apply_classical_faults(self, weights):
        """No classical faults in this version — pass through cleanly."""
        return np.asarray(weights).copy()

    # ------------------------------------------------------------------ #
    #  Public entry point
    # ------------------------------------------------------------------ #
    def wrap_circuit(self, qnode):
        """Return a callable with the same signature as ``qnode`` but with
        faults injected. Works for any QNode regardless of its internal
        structure, argument signature, number of wires, or measurements.
        """
        target_wires = self._resolve_target_wires(qnode)
        channels = self._active_channels()

        if self.mode == "exact":
            return self._wrap_exact(qnode, target_wires, channels)
        elif self.mode == "stochastic":
            return self._wrap_stochastic(qnode, target_wires, channels)
        else:
            raise ValueError(
                f"Unknown mode: {self.mode!r}. Use 'exact' or 'stochastic'."
            )

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    def _resolve_target_wires(self, qnode):
        if self.target_wires is not None:
            return list(self.target_wires)
        # Default: the first wire on the device.
        dev_wires = list(qnode.device.wires)
        return [dev_wires[0]] if dev_wires else [0]

    @staticmethod
    def _rebuild_on_device(qnode, device):
        """Return a copy of the qnode running on a different device."""
        return qml.QNode(qnode.func, device, diff_method=None)

    def _insertion_index(self, n_ops):
        if self.insert_after is None:
            return n_ops
        return max(0, min(int(self.insert_after), n_ops))

    # ------------------------------------------------------------------ #
    #  Exact path: density-matrix simulation on default.mixed
    # ------------------------------------------------------------------ #
    def _wrap_exact(self, qnode, target_wires, channels):
        n_wires = len(qnode.device.wires)
        mixed_dev = qml.device("default.mixed", wires=n_wires)
        insert_idx_fn = self._insertion_index

        @qml.transform
        def _inject(tape):
            ops = list(tape.operations)
            present = set(tape.wires)
            idx = insert_idx_fn(len(ops))

            fault_ops = []
            for w in target_wires:
                if w not in present:
                    warnings.warn(
                        f"target wire {w} is not used by the circuit; "
                        f"no fault injected there.", stacklevel=2,
                    )
                    continue
                for name, p in channels.items():
                    exact_op = _CHANNELS[name][0]
                    fault_ops.append(exact_op(p, wires=w))

            new_ops = ops[:idx] + fault_ops + ops[idx:]
            new_tape = type(tape)(new_ops, tape.measurements, shots=tape.shots)
            return [new_tape], lambda res: res[0]

        rebuilt = self._rebuild_on_device(qnode, mixed_dev)
        if not channels:
            return rebuilt          # baseline: no faults to insert
        return _inject(rebuilt)

    # ------------------------------------------------------------------ #
    #  Stochastic path: random Pauli insertion, averaged over n_trials
    # ------------------------------------------------------------------ #

    def _wrap_stochastic(self, qnode, target_wires, channels):
      n_wires = len(qnode.device.wires)
      pure_dev = qml.device(self.device_name, wires=n_wires)
      rebuilt = self._rebuild_on_device(qnode, pure_dev)
      if not channels:
        return rebuilt

      n_trials = self.n_trials
      seed = self.seed
      insert_idx_fn = self._insertion_index

      def _pattern_transform(fired_ops):
          # fired_ops: list of (wire, pauli_class) inserted deterministically
          @qml.transform
          def _inject(tape):
              ops = list(tape.operations)
              idx = insert_idx_fn(len(ops))
              faults = [cls(wires=w) for (w, cls) in fired_ops]
              new_ops = ops[:idx] + faults + ops[idx:]
              return [type(tape)(new_ops, tape.measurements, shots=tape.shots)], lambda r: r[0]
          return _inject

      def _build_events(present):
          events = []
          for w in target_wires:
              if w not in present:
                  warnings.warn(f"target wire {w} not used by circuit; skipped.", stacklevel=2)
                  continue
              for name in channels:
                  events.append((w, _CHANNELS[name][1], channels[name]))
          return events

      def _make_enumerated(events):
          patterns = []  # (transformed_qnode, probability_weight)
          for bits in itertools.product([0, 1], repeat=len(events)):
              fired, weight = [], 1.0
              for bit, (w, cls, p) in zip(bits, events):
                  if bit:
                      fired.append((w, cls)); weight *= p
                  else:
                      weight *= (1.0 - p)
              patterns.append((_pattern_transform(fired)(rebuilt), weight))

          def call(*a, **k):
              total = None
              for qn, wt in patterns:
                  r = np.asarray(qn(*a, **k), dtype=float) * wt
                  total = r if total is None else total + r
              return total.item() if total.ndim == 0 else total
          return call

      def _make_sampled(events):
          rng = np.random.default_rng(seed)
          def call(*a, **k):
              total = None
              for _ in range(n_trials):
                  fired = [(w, cls) for (w, cls, p) in events if rng.random() < p]
                  qn = _pattern_transform(fired)(rebuilt)
                  r = np.asarray(qn(*a, **k), dtype=float)
                  total = r if total is None else total + r
              avg = total / n_trials
              return avg.item() if avg.ndim == 0 else avg
          return call

        # Lazily decide enumerate-vs-sample on first call, once we can see the
        # tape and know which target wires are actually present.
      state = {"caller": None}

      def dispatch(*a, **k):
          if state["caller"] is None:
              tape = qml.workflow.construct_tape(rebuilt)(*a, **k)
              events = _build_events(set(tape.wires))
              if 2 ** len(events) <= n_trials:
                  state["caller"] = _make_enumerated(events)
              else:
                  state["caller"] = _make_sampled(events)
          return state["caller"](*a, **k)

      return dispatch
