import math
import numpy as np
from typing import Tuple, Optional, List


def to_scalar(x) -> float:
    """Convert a 0-d / 1x1 ndarray-like to a Python float."""
    return np.asarray(x, dtype=float).reshape(()).item()


def to_vec(x) -> np.ndarray:
    """Convert input to a 1-D vector with shape (n,)."""
    return np.asarray(x, dtype=float).reshape(-1)


def row1(x) -> np.ndarray:
    """Convert input to a single-row array with shape (1, n)."""
    return np.asarray(x, dtype=float).reshape(1, -1)


def cartpole_f_cont(x: np.ndarray, u: float, g: float, m_c: float, m_p: float, l: float) -> np.ndarray:
    """
    Continuous-time dynamics for the standard CartPole:
    x = [x_pos, x_dot, theta, theta_dot].
    Uses the underactuated model with input force u applied to the cart.
    """
    _, x_dot, th, th_dot = x
    s, c = math.sin(th), math.cos(th)
    M = m_c + m_p
    temp  = (u + m_p * l * th_dot**2 * s) / M
    denom = l * (4.0/3.0 - (m_p/M) * c*c)
    th_dd = (g * s - c * temp) / denom
    x_dd  = temp - (m_p * l * th_dd * c) / M
    return np.array([x_dot, x_dd, th_dot, th_dd], dtype=float)


def rk4_step(x: np.ndarray, u: float, dt: float, g: float, m_c: float, m_p: float, l: float) -> np.ndarray:
    """Fourth-order Runge–Kutta integrator for one time step."""
    k1 = cartpole_f_cont(x, u, g, m_c, m_p, l)
    k2 = cartpole_f_cont(x + 0.5 * dt * k1, u, g, m_c, m_p, l)
    k3 = cartpole_f_cont(x + 0.5 * dt * k2, u, g, m_c, m_p, l)
    k4 = cartpole_f_cont(x + dt * k3, u, g, m_c, m_p, l)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def fd_jacobian_fd(f_d, x: np.ndarray, u: float, dt: float, eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finite-difference Jacobians of a given discrete dynamics f_d:
    x_{k+1} = f_d(x_k, u_k, dt).
    Returns A = df/dx, B = df/du at (x, u).
    """
    n = x.size
    A = np.zeros((n, n))
    fx = f_d(x, u, dt)
    for i in range(n):
        dx = np.zeros_like(x)
        dx[i] = eps
        A[:, i] = (f_d(x + dx, u, dt) - f_d(x - dx, u, dt)) / (2 * eps)
    du = eps
    B = ((f_d(x, u + du, dt) - f_d(x, u - du, dt)) / (2 * du)).reshape(-1, 1)
    return A, B


class ILQRController:
    """
    iLQR controller with modularized planning steps:
    - rollout
    - trajectory linearization
    - backward pass (compute K_seq, k_seq)
    - forward line search
    - final backward pass to store gains
    """

    def __init__(
        self,
        g: float,
        m_c: float,
        m_p: float,
        l: float,
        Q: np.ndarray,
        R: float,
        Qf: np.ndarray,
        dt: float,
        max_iter: int = 100,
        tol: float = 1e-6,
        reg_min: float = 1e-6,
        reg_max: float = 1e10,
        alphas: list = None,
        act_limit: float = 10.0,
    ):
        self.g, self.m_c, self.m_p, self.l = g, m_c, m_p, l
        self.Q, self.Rs, self.Qf = Q, float(R), Qf
        self.dt = dt
        self.max_iter = max_iter
        self.tol = tol
        self.reg_min = reg_min
        self.reg_max = reg_max
        self.alphas = alphas if alphas is not None else [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]
        self.act_limit = float(act_limit)

        # Planned trajectories/gains (filled by plan())
        self.X_ref: Optional[np.ndarray] = None
        self.U_ff:  Optional[np.ndarray] = None
        self.K_seq: Optional[List[np.ndarray]] = None  # list of (1,n)
        self.k_seq: Optional[List[float]] = None       # list of scalars
        self._sigma: float = 0.0  # for PWM mode if you use it elsewhere

    def f_discrete(self, x: np.ndarray, u: float, dt: float) -> np.ndarray:
        return rk4_step(x, u, dt, self.g, self.m_c, self.m_p, self.l)

    def stage_cost(self, x: np.ndarray, u: float) -> float:
        return float(x @ self.Q @ x + self.Rs * (float(u) ** 2))

    def terminal_cost(self, x: np.ndarray) -> float:
        return float(x @ self.Qf @ x)

    def plan(self, x0: np.ndarray, U_init: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[float], float]:
        """
        Perform iLQR with Levenberg–Marquardt regularization and backtracking line-search.
        Returns (X_opt, U_opt, K_seq, k_seq, J).
        """
        x0 = to_vec(x0)
        U = np.asarray(U_init, dtype=float).reshape(-1, 1)
        N = U.shape[0]
        n = x0.size

        # 1) initial rollout
        X, J = self._initial_rollout(x0, U)

        lam = 1.0
        for _ in range(self.max_iter):
            # 2) linearize dynamics along current trajectory
            A_seq, B_seq = self._linearize_traj(X, U)

            # 3) backward pass (compute time-varying gains)
            bp_ok, K_seq, k_seq, Vx, Vxx = self._backward_pass(X, U, A_seq, B_seq, lam)
            if not bp_ok:
                # increase regularization and retry
                lam = min(lam * 10.0, self.reg_max)
                if lam >= self.reg_max:
                    break
                continue

            # 4) forward line search
            X_new, U_new, J_new, accepted = self._forward_line_search(x0, X, U, K_seq, k_seq, J, self.alphas)

            if accepted:
                X, U, J = X_new, U_new, J_new
                lam = max(lam / 10.0, self.reg_min)
            else:
                lam = min(lam * 10.0, self.reg_max)

            if abs(J - J_new) < self.tol:  # (equivalently abs(J_old - J) < tol)
                break

        # 5) final backward pass to store gains for closed-loop use
        K_seq, k_seq = self._final_backward_for_gains(X, U)

        # store and return
        self.X_ref, self.U_ff, self.K_seq, self.k_seq = X, U, K_seq, k_seq
        return X, U, K_seq, k_seq, float(J)

    # Step 1: rollout
    def _initial_rollout(self, x0: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, float]:
        """Forward simulate with given U to obtain (X, J)."""
        N = U.shape[0]
        n = x0.size
        X = np.zeros((N + 1, n), dtype=float); X[0] = x0
        J = 0.0
        for k in range(N):
            X[k + 1] = self.f_discrete(X[k], float(U[k, 0]), self.dt)
            J += self.stage_cost(X[k], float(U[k, 0]))
        J += self.terminal_cost(X[-1])
        return X, float(J)

    # Step 2: linearize
    def _linearize_traj(self, X: np.ndarray, U: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Compute (A_seq, B_seq) via finite differences along (X, U)."""
        N = U.shape[0]
        A_seq, B_seq = [], []
        for k in range(N):
            A, B = fd_jacobian_fd(self.f_discrete, X[k], float(U[k, 0]), self.dt)
            A_seq.append(A); B_seq.append(B)
        return A_seq, B_seq

    # Step 3: backward pass
    def _backward_pass(
        self,
        X: np.ndarray,
        U: np.ndarray,
        A_seq: List[np.ndarray],
        B_seq: List[np.ndarray],
        lam: float,
        ) -> Tuple[bool, List[np.ndarray], List[float], np.ndarray, np.ndarray]:
        """
        Compute time-varying gains (K_seq, k_seq) with current regularization lam.
        Returns (ok, K_seq, k_seq, Vx, Vxx).
        """
        N = U.shape[0]
        n = X.shape[1]

        Vx  = to_vec(self.Qf @ X[-1])
        Vxx = self.Qf.copy()
        K_seq: List[np.ndarray] = [None] * N
        k_seq: List[float] = [None] * N

        for k in reversed(range(N)):
            A, B = A_seq[k], B_seq[k]
            xk, uk = X[k], float(U[k, 0])

            # stage derivatives
            lx  = to_vec(self.Q @ xk)
            lu  = self.Rs * uk
            lxx = self.Q
            luu = self.Rs
            lux = np.zeros((1, n))

            # Q-function quadratic terms
            Qx  = lx + A.T @ Vx
            Qu  = lu + to_scalar(B.T @ Vx)
            Qxx = lxx + A.T @ Vxx @ A
            Qux = lux + B.T @ Vxx @ A
            Quu = luu + to_scalar(B.T @ Vxx @ B)

            Quu_reg = Quu + lam
            if Quu_reg <= 0:
                return False, None, None, None, None

            # gains
            kff = - Quu_reg**-1 * Qu              # scalar
            Kfb = - (1.0 / Quu_reg) * Qux         # (1, n)

            k_seq[k] = float(kff)
            K_seq[k] = row1(Kfb)

            # Value function recursion (symmetrized)
            k_vec   = Kfb.ravel()
            qux_vec = Qux.ravel()
            Vx  = Qx + k_vec * (Quu * kff + Qu) + qux_vec * kff
            Vxx = Qxx + Quu * (Kfb.T @ Kfb) + (Kfb.T @ Qux) + (Qux.T @ Kfb)
            Vxx = 0.5 * (Vxx + Vxx.T)

        return True, K_seq, k_seq, Vx, Vxx

    # Step 4: forward line search
    def _forward_line_search(
        self,
        x0: np.ndarray,
        X: np.ndarray,
        U: np.ndarray,
        K_seq: List[np.ndarray],
        k_seq: List[float],
        J_old: float,
        alphas: list,
    ) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """
        Try candidate step sizes alpha; accept the first that decreases cost.
        Returns (X_new, U_new, J_new, accepted).
        """
        N = U.shape[0]
        X_best, U_best, J_best = None, None, np.inf
        accepted = False

        for alpha in alphas:
            X_new = np.zeros_like(X); X_new[0] = x0
            U_new = np.zeros_like(U)
            J_new = 0.0

            for k in range(N):
                dx = to_vec(X_new[k] - X[k])
                du = k_seq[k] + to_scalar(K_seq[k] @ dx)
                u  = float(U[k, 0] + alpha * du)
                u  = self._clip_u(u)  # respect actuator limits
                X_new[k + 1] = self.f_discrete(X_new[k], u, self.dt)
                U_new[k, 0]  = u
                J_new += self.stage_cost(X_new[k], u)
            J_new += self.terminal_cost(X_new[-1])

            if J_new < J_old:
                X_best, U_best, J_best = X_new, U_new, float(J_new)
                accepted = True
                break

        if not accepted:
            # return original objects, signal rejection
            return X, U, float(J_old), False

        return X_best, U_best, J_best, True

    # Step 5: final backward pass
    def _final_backward_for_gains(self, X: np.ndarray, U: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """Run a backward pass on the final (X, U) to store K_seq and k_seq."""
        A_seq, B_seq = self._linearize_traj(X, U)

        N = U.shape[0]
        n = X.shape[1]
        Vx  = to_vec(self.Qf @ X[-1])
        Vxx = self.Qf.copy()
        K_seq: List[np.ndarray] = [None] * N
        k_seq: List[float] = [None] * N

        for k in reversed(range(N)):
            A, B = A_seq[k], B_seq[k]
            xk, uk = X[k], float(U[k, 0])

            lx  = to_vec(self.Q @ xk)
            lu  = self.Rs * uk
            lxx = self.Q; luu = self.Rs; lux = np.zeros((1, n))

            Qx  = lx + A.T @ Vx
            Qu  = lu + to_scalar(B.T @ Vx)
            Qxx = lxx + A.T @ Vxx @ A
            Qux = lux + B.T @ Vxx @ A
            Quu = luu + to_scalar(B.T @ Vxx @ B)

            Quu_reg = Quu + 1e-8
            kff = - Quu_reg**-1 * Qu
            Kfb = - (1.0 / Quu_reg) * Qux

            k_seq[k] = float(kff)
            K_seq[k] = row1(Kfb)

            k_vec   = Kfb.ravel()
            qux_vec = Qux.ravel()
            Vx  = Qx + k_vec * (Quu * kff + Qu) + qux_vec * kff
            Vxx = Qxx + Quu * (Kfb.T @ Kfb) + (Kfb.T @ Qux) + (Qux.T @ Kfb)
            Vxx = 0.5 * (Vxx + Vxx.T)

        return K_seq, k_seq

    def _clip_u(self, u: float) -> float:
        """Actuator-aware clipping used in line search to respect real force limits."""
        return float(np.clip(u, -self.act_limit, self.act_limit))

    def reset_pwm(self) -> None:
        self._sigma = 0.0
