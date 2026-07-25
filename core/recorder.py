import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml

from .faults import _CHANNELS


def _active_channels(cfg):
    """Active fault channels of a config, tolerating older configs."""
    if hasattr(cfg, "_active_channels"):
        return cfg._active_channels()
    return {
        name: p for name, p in (
            ("bit_flip",   getattr(cfg, "bit_flip_p",   0.0)),
            ("phase_flip", getattr(cfg, "phase_flip_p", 0.0)),
        ) if p > 0.0
    }


def _insert_label(cfg):
    """Human-readable description of where faults are inserted."""
    k = getattr(cfg, "insert_after", None)
    return "before measurement" if k is None else f"after op #{k}"


class ResultRecorder:
    """Simple recorder to log injection results."""
    def __init__(self):
        self.history = []

    def log(self, fault_config, weights, faulted_weights, result):
        record = {
            'config': fault_config,
            'original_weights': weights,
            'faulted_weights': faulted_weights,
            'result': result
        }
        self.history.append(record)
        print("Result recorded.")

    def log_step(self, fault_config, realization, loss, weights):
        """Log one record per optimizer step of a training session.

        Records are a superset of ``log()``'s shape, so ``summary()``,
        ``plot_results()`` and ``affected_parameters()`` work on them
        unchanged (``result`` holds the step's loss). Weights are unwrapped
        to plain NumPy for JSON safety. Deliberately silent — one print per
        step would drown a training run.
        """
        unwrapped = qml.math.unwrap(weights)
        record = {
            'config': fault_config,
            'original_weights': unwrapped,
            'faulted_weights': unwrapped,
            'result': None if loss is None else float(loss),
            'step': realization.step_index,
            'realization': realization.as_dict(),
        }
        self.history.append(record)

    def affected_parameters(self, index=-1):
        rec = self.history[index]
        cfg = rec["config"]
        w0  = np.asarray(rec["original_weights"])

        return {
            "name":            cfg.name,
            "mode":            getattr(cfg, "mode", "exact"),
            "target_wires":    cfg.target_wires,   # None -> resolved per-device at run time
            "active_channels": _active_channels(cfg),
            "insert_point":    _insert_label(cfg),
            "weight_shape":    w0.shape,
            "result":          rec["result"],
        }

    def print_affected_parameters(self, index=-1):
        """Pretty-print the fault summary for one record."""
        info = self.affected_parameters(index)
        wires = info["target_wires"]
        print(f"── Run: {info['name']} ──")
        print(f"  Mode            : {info['mode']}")
        print(f"  Target wires    : "
              + (str(wires) if wires is not None else "auto (first device wire)"))
        print(f"  Active channels : "
              + (" ".join(f"{k}(p={v})" for k, v in info["active_channels"].items())
                 if info["active_channels"] else "none"))
        print(f"  Insert point    : {info['insert_point']}")
        print(f"  Weight shape    : {info['weight_shape']}")
        print(f"  Result          : {info['result']}")

    def summary(self):
        """Print a one-line summary for every logged run."""
        if not self.history:
            print("No runs recorded.")
            return
        print(f"{'#':>3}  {'name':<14} {'mode':<11} {'wires':<10} {'channels':<28} {'result':>12}")
        print("-" * 84)
        for i, rec in enumerate(self.history):
            cfg = rec["config"]
            chans = ", ".join(
                f"{k}={v}" for k, v in _active_channels(cfg).items()
            ) or "none"
            wires = str(cfg.target_wires) if cfg.target_wires is not None else "auto"
            mode  = getattr(cfg, "mode", "exact")
            res   = rec["result"]
            res_s = (f"{float(res):.4f}" if np.ndim(res) == 0
                     else f"shape{np.shape(res)}")
            print(f"{i:>3}  {cfg.name:<14} {mode:<11} {wires:<10} {chans:<28} {res_s:>12}")

    def plot_wire_impact(self, figsize=(8, 4), save_path=None):
        """
        Bar chart showing how many runs touched each wire across all
        recorded faults — useful for slides describing experimental
        coverage.
        """
        if not self.history:
            print("No runs to plot.")
            return

        wire_counts = {}
        for rec in self.history:
            cfg = rec["config"]
            if not _active_channels(cfg):
                continue  # baseline run, no faults
            wires = cfg.target_wires if cfg.target_wires is not None else [0]
            for w in wires:
                wire_counts[w] = wire_counts.get(w, 0) + 1

        if not wire_counts:
            print("No faulted runs to plot.")
            return

        fig, ax = plt.subplots(figsize=figsize)
        wires = sorted(wire_counts)
        ax.bar([f"wire {w}" for w in wires], [wire_counts[w] for w in wires], color="#55A868")
        ax.set_ylabel("# faulted runs targeting this wire")
        ax.set_title("Wire coverage across logged runs")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved figure to {save_path}")
        plt.show()
        return fig

    def plot_results(self, figsize=(8, 4), save_path=None):
        """
        Bar chart of the scalar result of every logged run, so faulted runs
        can be compared against the baseline at a glance. Runs with
        non-scalar results are skipped.
        """
        if not self.history:
            print("No runs to plot.")
            return

        names, values = [], []
        for i, rec in enumerate(self.history):
            if np.ndim(rec["result"]) != 0:
                continue
            names.append(f"{i}: {rec['config'].name}")
            values.append(float(rec["result"]))

        if not names:
            print("No scalar results to plot.")
            return

        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(names, values, color="#4C72B0")
        ax.set_ylabel("result")
        ax.set_title("Result per logged run")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved figure to {save_path}")
        plt.show()
        return fig

    def draw_circuit(self, qnode, *args, index=-1, save_path=None, **kwargs):
        """
        Draw the faulted circuit for one logged run.

        Rebuilds the tape the same way FaultConfig does — construct the
        QNode's tape, then splice the exact channel ops in at the configured
        insertion point — so the drawing matches what actually executed.
        Extra ``*args``/``**kwargs`` are the circuit inputs after the weights
        (e.g. the data sample ``x``).
        """
        rec    = self.history[index]
        cfg    = rec["config"]
        params = rec["faulted_weights"]

        tape = qml.workflow.construct_tape(qnode)(params, *args, **kwargs)
        ops     = list(tape.operations)
        present = set(tape.wires)

        channels = _active_channels(cfg)
        if hasattr(cfg, "_resolve_target_wires"):
            target_wires = cfg._resolve_target_wires(qnode)
        else:
            target_wires = cfg.target_wires if cfg.target_wires is not None else [0]
        idx = cfg._insertion_index(len(ops)) if hasattr(cfg, "_insertion_index") else len(ops)

        fault_ops = [
            _CHANNELS[name][0](p, wires=w)
            for w in target_wires if w in present
            for name, p in channels.items()
        ]
        faulted_tape = type(tape)(ops[:idx] + fault_ops + ops[idx:],
                                  tape.measurements, shots=tape.shots)

        fig, ax = qml.drawer.tape_mpl(faulted_tape)
        chan_s = ", ".join(f"{k}(p={v})" for k, v in channels.items()) or "no faults"
        ax.set_title(f"Faulted circuit — {cfg.name}  [{chan_s}, {_insert_label(cfg)}]")

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved figure to {save_path}")
        plt.show()
        return fig, ax
