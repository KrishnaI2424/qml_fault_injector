# qml_fault_injector

A PennyLane-based wrapper that injects quantum faults into a QML model to
study fault tolerance, at HPC scale.

## Overview

`qml_fault_injector` wraps a QNode with a `FaultConfig` that specifies fault
type, probability, target wires, and simulation device. Running the wrapped
node — with trained weights and biases — produces the appropriately faulted
circuit output, with no assumptions about the circuit's internal structure.

What sets this apart is support for evaluating fault tolerance at HPC scale:
unlike density-matrix simulation, which is capped at a handful of qubits by
memory, the stochastic backend runs on pure-state simulators, including
`lightning.qubit` and `lightning.gpu`.

## Backends

Selected via `FaultConfig(mode=...)`:

- **`exact`** — density-matrix simulation on `default.mixed`, using
  PennyLane's native `qml.BitFlip` / `qml.PhaseFlip` channels. Ground truth,
  but memory-limited (scales as 2^2N).
- **`stochastic`** — pure-state simulation (`lightning.qubit`,
  `lightning.gpu`, or any pure-state device). These simulators don't support
  mixed-channel operations natively, so faults are approximated by randomly
  inserting Pauli gates and averaging over trials. When the number of fault
  "events" is small, every pattern is enumerated exactly (zero sampling
  error); otherwise `n_trials` realizations are Monte Carlo sampled. This
  approximation is unbiased and is validated against the `exact` backend —
  accuracy is best when wire/target counts are small enough for that
  reference to be checked directly.

## Installation

```bash
pip install -e .            # CPU: pennylane, numpy, pennylane-lightning
pip install -e ".[gpu]"     # + pennylane-lightning-gpu (needs a CUDA toolkit)
```

## Quick start

```python
import pennylane as qml
from qml_fault_injector import FaultConfig

dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev)
def circuit(weights, x):
    ...
    return qml.expval(qml.PauliZ(0))

cfg = FaultConfig(bit_flip_p=0.05, target_wires=[0], mode="exact")
faulted_circuit = cfg.wrap_circuit(circuit)

faulted_circuit(weights, x)   # same call signature as circuit, faults included
```

`inject(qnode, fault_config, recorder)` does the same wrapping but logs every
call into a `ResultRecorder`:

```python
from qml_fault_injector import inject, FaultConfig, ResultRecorder

recorder = ResultRecorder()
faulted  = inject(circuit, FaultConfig(bit_flip_p=0.05, target_wires=[0]), recorder)
faulted(weights, x)
recorder.summary()
```

For training *through* faults — one fixed fault pattern per optimizer step,
with correct gradient semantics — see [TRAINING.md](../TRAINING.md).

## Results

Trial data is logged in `ResultRecorder`, which stores and displays outcomes
from injection runs: `.summary()`, `.plot_results()`, `.plot_wire_impact()`,
`.draw_circuit()`, `.affected_parameters()`.

## Future Plans

Completing HPC-scale features, including a JSONL output for persistence
across SLURM array tasks. Designed around NCSA DeltaAI.
