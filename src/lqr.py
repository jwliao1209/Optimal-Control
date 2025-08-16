import numpy as np
from typing import Tuple, Optional
from numpy.linalg import inv
from scipy.linalg import eig, expm, ordqz
# from scipy.linalg import solve_discrete_are, solve_continuous_are


def solve_continuous_are(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """
    Solve the continuous-time algebraic Riccati equation (CARE)
    P = A^T P + P A - P B R^{-1} B^T P + Q
    via a symplectic pencil and eigenvalue decomposition.
    """

    # Construct Hamiltonian matrix
    H = np.block(
        [
            [A, -B @ inv(R) @ B.T],
            [-Q,             -A.T],
        ]
    )

    # Compute eigenvalue decomposition
    eigvals, eigvecs = eig(H)

    # Choose stable eigenvalues
    select = eigvals.real < 0
    eigvecs_stable = eigvecs[:, select]

    # Split into X, Y
    n = A.shape[0]
    X = eigvecs_stable[:n, :]
    Y = eigvecs_stable[n:, :]

    # Solve for P
    P = Y @ np.linalg.inv(X)
    P = P.real
    return P


def solve_discrete_are(A, B, Q, R):
    """
    Solve the discrete-time algebraic Riccati equation (DARE)
    X = A^T X A - (A^T X B + S) (R + B^T X B)^{-1} (B^T X A + S^T) + Q
    via a symplectic pencil and ordered QZ (generalized Schur) decomposition.

    If S is provided (cross term), we first apply the standard completion-of-squares
    reduction to an equivalent problem without cross term:
        Ã = A - B R^{-1} S^T
        Q̃ = Q - S R^{-1} S^T
        R̃ = R
        B̃ = B
    Then we solve the S=0 case for (Ã,B̃,Q̃,R̃).

    Returns
    -------
    X : ndarray (n,n)  -- stabilizing solution (symmetric if symmetrize=True)
    K : ndarray (m,n)  -- corresponding optimal feedback gain K = (R + B^T X B)^{-1}(B^T X A + S^T)
    """
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    Q = np.atleast_2d(Q)
    R = np.atleast_2d(R)
    n, m = B.shape

    S = np.zeros((n, m))

    # Complete the square to remove cross term:
    Rinv = inv(R)
    A_tilde = A - B @ Rinv @ S.T
    Q_tilde = Q - S @ Rinv @ S.T
    # Now solve the S = 0 DARE for (A_tilde, B, Q_tilde, R)

    # Build the symplectic pencil (M - λ N) of size 2n × 2n:
    BRB = B @ Rinv @ B.T
    M = np.block(
        [
            [A_tilde,                 np.zeros((n, n))],
            [-Q_tilde,                np.eye(n)       ],
        ]
    )
    N = np.block(
        [
            [np.eye(n),               BRB      ],
            [np.zeros((n, n)),        A_tilde.T],
        ]
    )

    # Generalized Schur (QZ) with ordering: inside unit circle first ('iuc')
    # qz returns AA, BB, Q, Z s.t. Q^T M Z = AA, Q^T N Z = BB (real case) up to conventions.
    # ordqz reorders to put desired eigenvalues (|alpha/beta|<1) leading.
    AA, BB, alpha, beta, Qqz, Zqz = ordqz(M, N, sort='iuc')

    # Extract the right Schur vectors (columns of Zqz) spanning stable invariant subspace:
    Z1 = Zqz[:n, :n]    # U1
    Z2 = Zqz[n:, :n]    # U2

    # X = U2 U1^{-1}
    X = Z2 @ inv(Z1)
    return X


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
