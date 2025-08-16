import numpy as np
from typing import Tuple, Optional
from scipy.linalg import expm, solve_continuous_are, solve_discrete_are


class BaseLQR:
    def u(self, x: np.ndarray) -> float:
        """
        Compute the continuous-valued control signal u = -Kx.
        This must be called after the controller has solved for the gain matrix K.
        """
        if not hasattr(self, "K"):
            raise RuntimeError("LQR gain K is not computed. Call solve() first.")
        return float(-(self.K @ x))

    def action(self, x: np.ndarray, deadband: float = 0.0) -> int:
        """
        Map the continuous control value u to the discrete CartPole action space.

        Parameters
        ----------
        x : np.ndarray
            State vector [x, x_dot, theta, theta_dot].
        deadband : float
            Optional deadband around zero to avoid control chattering.

        Returns
        -------
        int
            Discrete action: 0 = push left, 1 = push right.
        """
        u = self.u(x)
        if abs(u) <= deadband:
            return 1  # Default to pushing right within deadband
        return 1 if u > 0 else 0


class ContinuousLQR(BaseLQR):
    """Continuous-time LQR controller for linearized CartPole dynamics."""

    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray):
        self.A, self.B, self.Q, self.R = A, B, Q, R
        self.K: Optional[np.ndarray] = None

    def solve(self) -> np.ndarray:
        """Solve the continuous-time algebraic Riccati equation (CARE) and compute the LQR gain matrix."""
        P = solve_continuous_are(self.A, self.B, self.Q, self.R)
        self.K = np.linalg.solve(self.R, self.B.T @ P)  # Equivalent to R^{-1} B^T P
        return self.K


def c2d(A: np.ndarray, B: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a continuous-time state-space model to discrete-time using Zero-Order Hold (ZOH).

    Parameters
    ----------
    A : np.ndarray
        Continuous-time system matrix.
    B : np.ndarray
        Continuous-time input matrix.
    dt : float
        Sampling time step.

    Returns
    -------
    Ad : np.ndarray
        Discrete-time system matrix.
    Bd : np.ndarray
        Discrete-time input matrix.
    """
    n = A.shape[0]
    m = B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    Md = expm(M * dt)
    Ad = Md[:n, :n]
    Bd = Md[:n, n:]
    return Ad, Bd


class DiscreteLQR(BaseLQR):
    """Discrete-time LQR controller for linearized CartPole dynamics."""

    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, dt: float):
        self.dt = dt
        self.Ad, self.Bd = c2d(A, B, dt)
        self.Q, self.R = Q, R
        self.K: Optional[np.ndarray] = None

    def solve(self) -> np.ndarray:
        """Solve the discrete-time algebraic Riccati equation (DARE) and compute the LQR gain matrix."""
        X = solve_discrete_are(self.Ad, self.Bd, self.Q, self.R)
        self.K = np.linalg.inv(self.Bd.T @ X @ self.Bd + self.R) @ (self.Bd.T @ X @ self.Ad)
        return self.K
