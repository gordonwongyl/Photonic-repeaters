import numpy as np
from importlib import reload
import tgs
import rgs 


reload(tgs)
reload(rgs)

V_DELAY = 2e8 # m/s

def delay_line_tgs_ancilla(branch_param: np.ndarray, time: tgs.Time) -> float:
    return 0.

def delay_line_tgs_feedback(branch_param: np.ndarray, time: tgs.Time) -> float:
    return 0.

def feedback_line_tgs_feedback(branch_param: np.ndarray, time: tgs.Time) -> float:
    return 0.

def delay_line_rgs_ancilla(branch_param: np.ndarray, n: int, time: tgs.Time) -> float:
    return 0.

def delay_line_rgs_feedback(branch_param: np.ndarray, n: int, time: tgs.Time) -> float:
    return 0.

def feedback_line_rgs_feedback(branch_param: np.ndarray, n: int, time: tgs.Time) -> float:
    return 0.


