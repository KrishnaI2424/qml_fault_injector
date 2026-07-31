import numpy as np
import pennylane as qml
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable, Tuple
import functools
import itertools
import warnings


# Single-qubit Paulis by letter, used to materialize stochastic realizations.
_PAULIS = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}

# Map each logical fault channel to (exact density-matrix op, stochastic
# decomposition). The decomposition lists ``(pauli_letter, conditional_prob)``
# outcomes *given the channel fired* (conditional probs sum to 1): a channel
# with total probability p applies Pauli P_i with probability
# p * conditional_prob_i. This matches PennyLane's channel conventions —
# e.g. ``qml.DepolarizingChannel(p)`` applies each of X, Y, Z with
# probability p/3 and leaves the state untouched with probability 1 - p.
# Adding a new channel here is the only change needed to support it everywhere.
_CHANNELS = {
    "bit_flip":     (qml.BitFlip,             (("X", 1.0),)),
    "phase_flip":   (qml.PhaseFlip,           (("Z", 1.0),)),
    "depolarizing": (qml.DepolarizingChannel, (("X", 1 / 3), ("Y", 1 / 3), ("Z", 1 / 3))),
}


def _sample_pauli(rng, name):
    """Draw one Pauli letter from a fired channel's conditional decomposition."""
    decomp = _CHANNELS[name][1]
    if len(decomp) == 1:
        return decomp[0][0]
    letters, probs = zip(*decomp)
    return letters[rng.choice(len(letters), p=np.asarray(probs) / sum(probs))]


@dataclass
class FaultRealization:
    """One sampled Pauli fault pattern, held fixed for one optimizer step.

    ``fired`` lists which channel from ``_CHANNELS`` fired on which of the
    config's target wires, and which Pauli it collapsed to:
    e.g. ``[("bit_flip", 0, "X"), ("depolarizing", 3, "Y")]``. The Pauli is
    recorded explicitly because multi-outcome channels (depolarizing) cannot
    be reconstructed from channel name + wire alone.
    """
    step_index: int
    fired: List[Tuple[str, int, str]]

    def pauli_ops(self):
        """Materialise the realized faults as non-parameterized Pauli gates."""
        return [_PAULIS[pauli](wires=w) for _name, w, pauli in self.fired]

    def as_dict(self):
        """JSON-safe representation for logging."""
        return {
            "step_index": self.step_index,
            "fired": [[name, int(w), pauli] for name, w, pauli in self.fired],
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
    depolarizing_p: float = 0.0   # total channel prob; each Pauli fires with p/3
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
                ("bit_flip",     self.bit_flip_p),
                ("phase_flip",   self.phase_flip_p),
                ("depolarizing", self.depolarizing_p),
            ) if p > 0.0
        }

    @classmethod
    def no_faults(cls, name="baseline", **kwargs):
        """A fault-free config.

        ``training_session()`` on it produces a stepper that injects nothing —
        behaviorally identical to a bare ``opt.step`` loop (verified by
        ``test_zero_probability_matches_unwrapped_baseline_exactly``). Use it as
        the no-fault default so plain and faulted training share one abstraction.

        Probabilities are already zero by default, so this is a self-documenting
        convenience. Extra kwargs (e.g. ``target_wires``, ``insert_after``) pass
        through for callers who want a placeholder they can later flip on.
        """
        return cls(name=name, bit_flip_p=0.0, phase_flip_p=0.0,
                   depolarizing_p=0.0, **kwargs)

    def apply_classical_faults(self, weights):
        """No classical faults in this version — pass through cleanly.

        The input is returned untouched: converting via ``np.asarray(...).copy()``
        would strip autograd tracking from trainable tensors (pennylane.numpy,
        torch, jax). Any future perturbation must use ``qml.math`` operations so
        trainable tensor types are preserved.
        """
        return weights

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

    def training_session(self, qnode, device_name=None):
        """Return a :class:`FaultTrainingSession` for per-step fault injection.

        Unlike ``wrap_circuit`` (which targets inference on trained models),
        the session draws ONE fault realization per optimizer step and holds
        it fixed across every QNode evaluation within that step — all batch
        points and parameter-shift evaluations see the same Pauli pattern.

        ``device_name=None`` keeps the QNode's own device; pass e.g.
        ``"default.mixed"`` to train on a different simulator.
        """
        return FaultTrainingSession(self, qnode, device_name=device_name)

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
          # fired_ops: list of (wire, pauli_letter) inserted deterministically
          @qml.transform
          def _inject(tape):
              ops = list(tape.operations)
              idx = insert_idx_fn(len(ops))
              faults = [_PAULIS[pauli](wires=w) for (w, pauli) in fired_ops]
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
                  events.append((w, name, channels[name]))
          return events

      def _n_patterns(events):
          # Each event contributes (1 + #decomposition outcomes) branches:
          # "did not fire" plus one branch per Pauli it can collapse to.
          n = 1
          for (_w, name, _p) in events:
              n *= 1 + len(_CHANNELS[name][1])
          return n

      def _make_enumerated(events):
          # Per-event outcome list: None ("did not fire", weight 1-p) plus one
          # (wire, pauli) outcome per decomposition entry, weighted by the
          # channel's total probability times the conditional prob — p/3 per
          # Pauli for depolarizing, matching qml.DepolarizingChannel exactly.
          outcome_sets = []
          for (w, name, p) in events:
              outcomes = [(None, 1.0 - p)]
              outcomes += [((w, pauli), p * cond) for pauli, cond in _CHANNELS[name][1]]
              outcome_sets.append(outcomes)

          patterns = []  # (transformed_qnode, probability_weight)
          for combo in itertools.product(*outcome_sets):
              fired, weight = [], 1.0
              for outcome, wt in combo:
                  weight *= wt
                  if outcome is not None:
                      fired.append(outcome)
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
                  fired = [(w, _sample_pauli(rng, name))
                           for (w, name, p) in events if rng.random() < p]
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
              if _n_patterns(events) <= n_trials:
                  state["caller"] = _make_enumerated(events)
              else:
                  state["caller"] = _make_sampled(events)
          return state["caller"](*a, **k)

      return dispatch


class FaultTrainingSession:
    """Per-step fixed-sample fault injection during training (autograd path).

    One :class:`FaultRealization` is drawn per optimizer step (``new_step``)
    and applied at QNode call time by splicing the realized Pauli gates into
    the tape as non-parameterized constants, so every evaluation within one
    step — all batch points and all parameter-shift/backprop evaluations —
    sees the same fixed pattern.

    Contract for ``step``: the user's cost function accepts the wrapped QNode
    as its FIRST parameter (``cost(qnode, *trainable_and_data_args)``) rather
    than closing over a module-level circuit.

    TODO: JAX/jit mask-parameterization backend. Under ``jax.jit`` the tape
    cannot be re-spliced per step; the plan is to parameterize the fault layer
    with a static-shape 0/1 mask array (one slot per target-wire/channel
    event) baked into the circuit, updating only the mask values each step.
    Not implemented yet — constructing a session from a JAX-interface QNode
    raises ``NotImplementedError``.
    """

    def __init__(self, config, qnode, device_name=None):
        interface = str(getattr(qnode, "interface", "auto"))
        if "jax" in interface:
            raise NotImplementedError(
                "FaultTrainingSession only supports the autograd path for now; "
                "the JAX/jit mask-parameterization backend is not implemented yet."
            )

        self.config = config
        self.recorder = None  # set below; import here would be circular at module load

        # Rebuild the QNode onto the training device exactly once, preserving
        # the original interface, diff_method, and wire labels.
        dev_name = device_name or qnode.device.name
        dev = qml.device(dev_name, wires=list(qnode.device.wires))
        rebuilt = qml.QNode(
            qnode.func, dev,
            interface=qnode.interface,
            diff_method=qnode.diff_method,
        )

        self._target_wires = config._resolve_target_wires(rebuilt)
        # Same event structure as _wrap_stochastic._build_events, resolved
        # against the config rather than the tape (sampling happens before any
        # circuit call). Wires absent from the tape are skipped at splice time.
        self._events = [
            (w, name, p)
            for w in self._target_wires
            for name, p in config._active_channels().items()
        ]
        self._rng = np.random.default_rng(config.seed)
        self._step_count = 0
        self._current: Optional[FaultRealization] = None

        session = self
        insert_idx_fn = config._insertion_index

        @qml.transform
        def _apply_realization(tape):
            ops = list(tape.operations)
            present = set(tape.wires)
            idx = insert_idx_fn(len(ops))
            fault_ops = []
            if session._current is not None:
                # The realization carries the concrete Pauli letter: a
                # depolarizing fault cannot be re-derived from name + wire.
                for _name, w, pauli in session._current.fired:
                    if w not in present:
                        warnings.warn(
                            f"target wire {w} is not used by the circuit; "
                            f"no fault injected there.", stacklevel=2,
                        )
                        continue
                    fault_ops.append(_PAULIS[pauli](wires=w))
            new_ops = ops[:idx] + fault_ops + ops[idx:]
            new_tape = type(tape)(new_ops, tape.measurements, shots=tape.shots)
            return [new_tape], lambda res: res[0]

        self.qnode = _apply_realization(rebuilt)

        from .recorder import ResultRecorder
        self.recorder = ResultRecorder()

    def new_step(self):
        """Draw a fresh fault realization from the config RNG.

        Each event fires with its channel's total probability; a fired event
        then collapses to one Pauli from the channel's conditional
        decomposition (always X for bit_flip, Z for phase_flip; uniform over
        X/Y/Z for depolarizing — i.e. p/3 each, PennyLane's convention).
        """
        fired = [(name, w, _sample_pauli(self._rng, name))
                 for (w, name, p) in self._events if self._rng.random() < p]
        self._current = FaultRealization(step_index=self._step_count, fired=fired)
        self._step_count += 1
        return self._current

    def step(self, opt, cost, *args, **kwargs):
        """Draw a new realization, then delegate to the optimizer's step.

        ``cost`` must take the wrapped QNode as its first parameter; it is
        bound here so the optimizer differentiates only the remaining args.
        Returns exactly what ``opt.step`` would. Uses ``step_and_cost`` when
        the optimizer provides it, so the pre-update loss is logged without
        extra circuit evaluations.
        """
        realization = self.new_step()
        bound_cost = functools.partial(cost, self.qnode)
        if hasattr(opt, "step_and_cost"):
            new_args, loss = opt.step_and_cost(bound_cost, *args, **kwargs)
        else:
            new_args, loss = opt.step(bound_cost, *args, **kwargs), None
        self.recorder.log_step(self.config, realization, loss,
                               args[0] if args else None)
        return new_args
