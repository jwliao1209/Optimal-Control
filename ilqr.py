#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math

# ========= CartPole 物理參數 =========
g = 9.8
m_c = 1.0
m_p = 0.1
l   = 0.5               # 半長（樞軸到質心）
M   = m_c + m_p

# ========= 小工具：統一 scalar / 向量形狀 =========
def to_scalar(x):
    """把 0-d 或 1×1 ndarray 轉成純 Python float。"""
    return np.asarray(x, dtype=float).reshape(()).item()

def to_vec(x):
    """把輸入轉成 (n,) 一維向量。"""
    return np.asarray(x, dtype=float).reshape(-1)

def row1(x):
    """把輸入轉成 (1,n) 單列向量。"""
    return np.asarray(x, dtype=float).reshape(1, -1)

# ========= 連續時間動態 f(x,u) =========
def f_cont(x, u):
    # x = [x_pos, x_dot, theta, theta_dot]
    x_pos, x_dot, th, th_dot = x
    s, c = math.sin(th), math.cos(th)
    temp  = (u + m_p * l * th_dot**2 * s) / M
    denom = l * (4.0/3.0 - (m_p/M) * c*c)
    th_dd = (g * s - c * temp) / denom
    x_dd  = temp - (m_p * l * th_dd * c) / M
    # 第 3 個分量必須是角速度 th_dot（不是 th）
    return np.array([x_dot, x_dd, th_dot, th_dd], dtype=float)

# ========= RK4 離散步進 x_{k+1} = f_d(x_k,u_k) =========
def rk4_step(x, u, dt):
    k1 = f_cont(x, u)
    k2 = f_cont(x + 0.5*dt*k1, u)
    k3 = f_cont(x + 0.5*dt*k2, u)
    k4 = f_cont(x + dt*k3, u)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def f_discrete(x, u, dt):
    return rk4_step(x, u, dt)

# ========= 成本 =========
def stage_cost(x, u, Q, R_s):
    # R_s 必須是純 scalar
    u = float(u)
    return x @ Q @ x + R_s * (u**2)

def terminal_cost(x, Qf):
    return x @ Qf @ x

# ========= 有限差分 Jacobian of f_d =========
def fd_jacobian_fd(f_d, x, u, dt, eps=1e-5):
    n = x.size
    A = np.zeros((n, n))
    fx = f_d(x, u, dt)
    for i in range(n):
        dx = np.zeros_like(x)
        dx[i] = eps
        A[:, i] = (f_d(x + dx, u, dt) - f_d(x - dx, u, dt)) / (2*eps)
    du = eps
    B = ((f_d(x, u + du, dt) - f_d(x, u - du, dt)) / (2*du)).reshape(-1, 1)
    return A, B

# ========= iLQR 主函式 =========
def ilqr(f_d, x0, U_init, Q, R, Qf, dt,
         max_iter=100, tol=1e-6, reg_min=1e-6, reg_max=1e10):
    """
    f_d: x_{k+1} = f_d(x_k, u_k, dt)
    x0:  (n,)
    U_init: (N,1)
    Q,Qf: (n,n)
    R: 標量或 1x1 ndarray
    """
    n = x0.size
    N = U_init.shape[0]
    R_s = to_scalar(R)              # 把 R 變成純 scalar

    U = U_init.copy().reshape(N, 1)
    X = np.zeros((N+1, n)); X[0] = x0

    # 初次 forward
    J = 0.0
    for k in range(N):
        X[k+1] = f_d(X[k], U[k,0], dt)
        J += stage_cost(X[k], U[k,0], Q, R_s)
    J += terminal_cost(X[-1], Qf)

    lam = 1.0                       # Levenberg–Marquardt 正則
    alphas = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]

    for it in range(max_iter):
        # 線性化
        A_seq, B_seq = [], []
        for k in range(N):
            A, B = fd_jacobian_fd(f_d, X[k], U[k,0], dt)
            A_seq.append(A); B_seq.append(B)

        # ----- backward pass（shape-safe） -----
        Vx  = to_vec(Qf @ X[-1])     # (n,)
        Vxx = Qf.copy()              # (n,n)
        k_seq = [None]*N
        K_seq = [None]*N
        diverged = False

        for k in reversed(range(N)):
            A, B = A_seq[k], B_seq[k]     # (n,n), (n,1)
            xk, uk = X[k], U[k,0]         # (n,), scalar

            lx  = to_vec(Q @ xk)          # (n,)
            lu  = R_s * float(uk)         # scalar
            lxx = Q
            luu = R_s
            lux = np.zeros((1, n))        # (1,n)

            Qx  = lx + A.T @ Vx                              # (n,)
            Qu  = lu + to_scalar(B.T @ Vx)                   # scalar
            Qxx = lxx + A.T @ Vxx @ A                        # (n,n)
            Qux = lux + B.T @ Vxx @ A                        # (1,n)
            Quu = luu + to_scalar(B.T @ Vxx @ B)             # scalar

            Quu_reg = Quu + lam
            if Quu_reg <= 0:
                diverged = True
                break

            kff_s = - Quu_reg**-1 * Qu                       # scalar
            Kfb   = - (1.0/Quu_reg) * Qux                    # (1,n)

            k_seq[k] = kff_s                                 # 存純 scalar
            K_seq[k] = row1(Kfb)                             # (1,n)

            # ---- 形狀安全的 Vx / Vxx 更新 ----
            k_vec   = Kfb.ravel()                            # (n,)
            qux_vec = Qux.ravel()                            # (n,)

            Vx  = Qx + k_vec*(Quu*kff_s + Qu) + qux_vec*kff_s        # (n,)
            Vxx = Qxx + Quu*(Kfb.T @ Kfb) + (Kfb.T @ Qux) + (Qux.T @ Kfb)
            Vxx = 0.5*(Vxx + Vxx.T)

        if diverged:
            lam = min(lam * 10.0, reg_max)
            if lam >= reg_max:
                break
            continue

        # ----- line-search forward -----
        improved = False
        J_old = J
        for alpha in alphas:
            X_new = np.zeros_like(X); X_new[0] = x0
            U_new = np.zeros_like(U)
            J_new = 0.0
            for k in range(N):
                dx = to_vec(X_new[k] - X[k])                 # (n,)
                du = k_seq[k] + to_scalar(K_seq[k] @ dx)     # scalar
                # line-search 內部
                u  = U[k,0] + alpha * du
                u  = float(np.clip(u, -20.0, 20.0))
                X_new[k+1] = f_d(X_new[k], u, dt)
                U_new[k,0] = u
                J_new += stage_cost(X_new[k], u, Q, R_s)
            J_new += terminal_cost(X_new[-1], Qf)

            if J_new < J:
                X, U, J = X_new, U_new, J_new
                lam = max(lam/10.0, reg_min)
                improved = True
                break

        if not improved:
            lam = min(lam*10.0, reg_max)

        if abs(J_old - J) < tol:
            break

    # ----- 再跑一次 backward 取最終 K,k（供時變閉迴路用） -----
    A_seq, B_seq = [], []
    for k in range(N):
        A,B = fd_jacobian_fd(f_d, X[k], U[k,0], dt)
        A_seq.append(A); B_seq.append(B)

    Vx  = to_vec(Qf @ X[-1])
    Vxx = Qf.copy()
    K_seq = [None]*N; k_seq = [None]*N

    for k in reversed(range(N)):
        A,B = A_seq[k], B_seq[k]
        xk, uk = X[k], U[k,0]

        lx  = to_vec(Q @ xk)
        lu  = R_s * float(uk)
        lxx = Q; luu = R_s; lux = np.zeros((1, n))

        Qx  = lx + A.T @ Vx
        Qu  = lu + to_scalar(B.T @ Vx)
        Qxx = lxx + A.T @ Vxx @ A
        Qux = lux + B.T @ Vxx @ A
        Quu = luu + to_scalar(B.T @ Vxx @ B)

        Quu_reg = Quu + 1e-8
        kff_s = - Quu_reg**-1 * Qu
        Kfb   = - (1.0/Quu_reg) * Qux

        k_seq[k] = kff_s
        K_seq[k] = row1(Kfb)

        k_vec   = Kfb.ravel()
        qux_vec = Qux.ravel()
        Vx  = Qx + k_vec*(Quu*kff_s + Qu) + qux_vec*kff_s
        Vxx = Qxx + Quu*(Kfb.T @ Kfb) + (Kfb.T @ Qux) + (Qux.T @ Kfb)
        Vxx = 0.5*(Vxx + Vxx.T)

    return X, U, K_seq, k_seq, J

# ========= Gym 連結（把連續力映射到離散動作） =========
def run_on_gym(X_ref, U_ff, K_seq, k_seq, T_steps, MODE="pwm"):
    """把 iLQR 的連續力映射到 Gym CartPole-v1 上運行並顯示動畫。
    MODE: "bangbang" 或 "pwm"（建議 pwm，比較平滑）
    """
    # --- 嘗試同時相容 gymnasium 與舊版 gym ---
    try:
        import gymnasium as gym
    except ImportError:
        import gym

    # 建立環境（開啟 render）
    try:
        env = gym.make("CartPole-v1", render_mode="human")
    except TypeError:
        env = gym.make("CartPole-v1")

    # 時間步長、力大小
    dt_env = getattr(env.unwrapped, "tau", 0.02)
    force_mag = getattr(env.unwrapped, "force_mag", 10.0)

    # ΣΔ 調變器狀態（pwm 模式用）
    sigma = 0.0

    # reset
    try:
        obs, info = env.reset(seed=0)
    except:
        obs = env.reset()
        info = {}

    total_r = 0.0
    for k in range(T_steps):
        x_ref = X_ref[min(k, len(X_ref)-1)]
        # 狀態差（Gym obs 同順序：[x, xdot, theta, thetadot]）
        dx = to_vec(obs) - to_vec(x_ref)

        # iLQR 力（前饋 + 回授）
        u_des = float(U_ff[min(k, len(U_ff)-1), 0] + k_seq[min(k, len(k_seq)-1)] \
                      + to_scalar(K_seq[min(k, len(K_seq)-1)] @ dx))
        # 可選限幅
        u_des = float(np.clip(u_des, -2*force_mag, 2*force_mag))

        # 連續力 -> Gym 離散動作
        if MODE == "bangbang":
            action = 1 if u_des >= 0.0 else 0
        else:
            # ΣΔ (sigma-delta) 近似連續平均力：
            # 期望歸一化 d in [-1,1]，輸出 y ∈ {-1,+1}，平均(y) ≈ d
            d = float(np.clip(u_des/force_mag, -1.0, 1.0))
            y = 1.0 if sigma >= 0.0 else -1.0
            sigma += d - y
            action = 1 if y > 0 else 0

        step_out = env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
        else:
            obs, reward, done, info = step_out
            terminated, truncated = done, False

        total_r += reward
        if terminated or truncated:
            break

    print(f"[Gym] return={total_r}, steps={k+1}, mode={MODE}")
    env.close()

# ========= 參數、初始化與執行 =========
if __name__ == "__main__":
    # 設定軌跡規劃的步長與時域
    dt = 0.001
    T  = 5.0
    N  = int(T/dt)

    # 成本權重（可自行調整）
    Q  = np.diag([1.0, 0.1, 20.0, 10.0])
    R  = 10                      # 直接給 scalar（或 np.array([[0.1]]) 也行）
    Qf = np.diag([5.0, 0.0, 10.0, 0.0])

    # 初始狀態（10° 偏置）
    th0 = np.deg2rad(1)
    x0  = np.array([0.0, 0.0, th0, 0.0])

    # 初始控制序列（全 0）
    U0  = np.zeros((N, 1))

    # === 先用自家模型做 iLQR 規劃 ===
    X_opt, U_opt, K_seq, k_seq, J = ilqr(
        f_discrete, x0, U0, Q, R, Qf, dt,
        max_iter=60, tol=1e-6,
    )
    print("Final cost J =", J)

    # # ======= 數值閉迴路驗證（可視化；選用） =======
    # try:
    #     import matplotlib.pyplot as plt
    #     X_cl = np.zeros_like(X_opt); X_cl[0] = x0
    #     U_cl = np.zeros_like(U_opt)
    #     for k in range(N):
    #         dx = to_vec(X_cl[k] - X_opt[k])
    #         u  = U_opt[k,0] + k_seq[k] + to_scalar(K_seq[k] @ dx)
    #         X_cl[k+1] = f_discrete(X_cl[k], u, dt)
    #         U_cl[k,0] = u

    #     t = np.arange(N+1) * dt
    #     fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    #     axs[0].plot(t, X_cl[:,2]*180/np.pi); axs[0].axhline(0, color='k', lw=0.5)
    #     axs[0].set_ylabel("theta [deg]"); axs[0].grid(True)
    #     axs[1].plot(t, X_cl[:,0]); axs[1].axhline(0, color='k', lw=0.5)
    #     axs[1].set_ylabel("x [m]"); axs[1].grid(True)
    #     axs[2].plot(t[:-1], U_cl[:-1,0])
    #     axs[2].set_ylabel("u [N]"); axs[2].set_xlabel("time [s]"); axs[2].grid(True)
    #     plt.suptitle("CartPole iLQR: closed-loop rollout (internal model)")
    #     plt.show()
    # except Exception as e:
    #     print("Plot skipped:", e)

    # ======= 接上 Gym，顯示動畫（兩種模式擇一） =======
    # 你可以改 "bangbang" 或 "pwm"
    run_on_gym(X_opt, U_opt, K_seq, k_seq, T_steps=N, MODE="pwm")
