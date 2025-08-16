from typing import Optional, List
import numpy as np


class ControlParams:
    def __init__(
        self,
        g: float = 9.8,
        m_c: float = 1.0,
        m_p: float = 0.1,
        l: float = 0.5,
        Q: Optional[List[float]] = None,
        Qf: Optional[List[float]] = None,
        R: Optional[List[float]] = None,
        *args,
        **kwargs,
    ):
        M = m_c + m_p
        D = l * (4.0 / 3.0 - m_p / M)
        alpha = (m_p / M) / (4.0 / 3.0 - m_p / M)
        self.A = np.array(
            [
                [0, 1,          0, 0],
                [0, 0, -alpha * g, 0],
                [0, 0,          0, 1],
                [0, 0,      g / D, 0],
            ],
            dtype=float)

        self.B = np.array(
            [
                [              0],
                [(1 + alpha) / M],
                [              0],
                [   -1 / (D * M)],
            ],
            dtype=float)
        self.Q = np.diag(Q) if Q is not None else None
        self.Qf = np.diag(Qf) if Qf is not None else None
        self.R = np.array([R]) if R is not None else None
