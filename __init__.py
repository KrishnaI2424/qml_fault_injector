# qml_fault_injector/__init__.py

__version__ = "0.1.0"
__author__  = "Krishna Iyer"

# from qml_fault_injector import inject
from .core.wrapper import inject
from .core.faults  import FaultConfig
from .core.recorder import ResultRecorder
