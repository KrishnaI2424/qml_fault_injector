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
