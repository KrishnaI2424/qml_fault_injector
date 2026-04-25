import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
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
    def affected_parameters(self, index=-1):
        rec  = self.history[index]
        cfg  = rec["config"]
        w0   = np.asarray(rec["original_weights"])
        wf   = np.asarray(rec["faulted_weights"])

        diff = ~np.isclose(w0, wf, atol=1e-12)
        n_changed = int(diff.sum())

        active_channels = {
            ch: p for ch, p in [
                ("bit_flip",   getattr(cfg, "bit_flip_p",   0.0)),
                ("phase_flip", getattr(cfg, "phase_flip_p", 0.0)),
                ("depolar",    getattr(cfg, "depolar_p",    0.0)),
            ] if p > 0.0
        }

        return {
            "name":              cfg.name,
            "target_wires":      cfg.target_wires if cfg.target_wires is not None else [0],
            "active_channels":   active_channels,
            "weight_shape":      w0.shape,
            "weights_changed":   n_changed,
            "weights_total":     w0.size,
            "result":            rec["result"],
        }

    def print_affected_parameters(self, index=-1):
        """Pretty-print the affected-parameter summary for one record."""
        info = self.affected_parameters(index)
        print(f"── Run: {info['name']} ──")
        print(f"  Target wires    : {info['target_wires']}")
        print(f"  Active channels : "
              + (" ".join(f"{k}(p={v})" for k, v in info["active_channels"].items())
                 if info["active_channels"] else "none"))
        print(f"  Weight shape    : {info['weight_shape']}")
        print(f"  Weights changed : {info['weights_changed']} / {info['weights_total']}")
        print(f"  Result          : {info['result']}")

    def summary(self):
        """Print a one-line summary for every logged run."""
        if not self.history:
            print("No runs recorded.")
            return
        print(f"{'#':>3}  {'name':<14} {'wires':<10} {'channels':<32} {'result':>10}")
        print("-" * 75)
        for i, rec in enumerate(self.history):
            cfg = rec["config"]
            chans = ", ".join(f"{k}={v}" for k, v in [
                ("bf", cfg.bit_flip_p), ("pf", cfg.phase_flip_p), ("dp", cfg.depolar_p),
            ] if v > 0.0) or "none"
            wires = str(cfg.target_wires if cfg.target_wires is not None else [0])
            res   = rec["result"]
            res_s = f"{float(res):.4f}" if np.isscalar(res) or np.ndim(res) == 0 else f"shape{np.shape(res)}"
            print(f"{i:>3}  {cfg.name:<14} {wires:<10} {chans:<32} {res_s:>10}")


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
            cfg   = rec["config"]
            wires = cfg.target_wires if cfg.target_wires is not None else [0]
            # Skip the baseline run (no faults active)
            if not any([cfg.bit_flip_p, cfg.phase_flip_p]):
                continue
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


    
    def draw_circuit(self, qnode, x, index=-1, save_path=None):
        import pennylane as qml
 
        rec    = self.history[index]
        cfg    = rec["config"]
        params = rec["faulted_weights"]
 
        # Reconstruct the faulted circuit inline so draw_mpl sees the
        # noise channels exactly as they were injected at run time.
        target_wires = cfg.target_wires if cfg.target_wires is not None else [0]
        bit_flip_p   = getattr(cfg, "bit_flip_p",   0.0)
        phase_flip_p = getattr(cfg, "phase_flip_p", 0.0)
        original_func = qnode.func
 
        def faulted_func(params, x):
            for wire in target_wires:
                if bit_flip_p   > 0.0: qml.BitFlip(bit_flip_p,   wires=wire)
                if phase_flip_p > 0.0: qml.PhaseFlip(phase_flip_p, wires=wire)
            return original_func(params, x)
 
        drawable = qml.QNode(faulted_func, qnode.device)
        fig, ax  = qml.draw_mpl(drawable)(params, x)
 
        ax.set_title(
            f"Faulted circuit — {cfg.name}  "
            + ", ".join(f"{k}(p={v})" for k, v in [
                ("BitFlip",   bit_flip_p),
                ("PhaseFlip", phase_flip_p)
            ] if v > 0.0)
        )
 
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved figure to {save_path}")
        plt.show()
        return fig, a
