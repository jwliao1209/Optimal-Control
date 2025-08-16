#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# =========================
# 全域設定（對齊 Gym）
# =========================
device = torch.device("cpu")
DT = 0.02                     # Gym CartPole tau
FORCE_MAG = 10.0              # Gym ±10N
THETA_MAX = 12.0*np.pi/180.0  # 0.20944 rad
X_MAX = 2.4                   # m

# 物理常數（與 Gym 一致）
g = 9.8
masscart = 1.0
masspole = 0.1
total_mass = masscart + masspole
length = 0.5                    # 半長（與 Gym 同）
polemass_len = masspole * length

# =========================
# ΣΔ 量化器（帶 STE）
# =========================
class SigmaDeltaQuantizer:
    def __init__(self, force_mag=FORCE_MAG, leak=0.95, device="cpu", batch=1):
        self.force_mag = float(force_mag)
        self.leak = float(leak)
        self.sigma = torch.zeros((batch,1), dtype=torch.float32, device=device)

    def reset(self, batch):
        self.sigma = torch.zeros((batch,1), dtype=torch.float32, device=self.sigma.device)

    def step(self, u_cont):
        """
        u_cont: (B,1) 連續力
        return: (B,1) 量化後實際施加的力 ∈ { -force_mag, +force_mag }
        """
        d = torch.clamp(u_cont / self.force_mag, -1.0, 1.0)  # 期望歸一化
        # STE：forward 用 sign(sigma)，backward 對 d 傳梯度
        with torch.no_grad():
            y_det = torch.where(self.sigma >= 0.0, torch.ones_like(self.sigma), -torch.ones_like(self.sigma))
        y = d + (y_det - d).detach()  # STE
        self.sigma = self.leak * self.sigma + (d - y)
        return y * self.force_mag

# =========================
# 工具
# =========================
def wrap_angle_torch(theta):
    return (theta + np.pi) % (2*np.pi) - np.pi

def torch_wrap_angle_inplace(x):
    x[:,2] = wrap_angle_torch(x[:,2])
    return x

# =========================
# Gym 版連續時間動態 + Euler 前向（與 Gym 一致）
# x = [x, xdot, theta, thetadot]
# =========================
def f_cont_torch(x, u):
    # x: (B,4), u: (B,1)
    theta  = wrap_angle_torch(x[:,2])
    x_dot  = x[:,1]
    th_dot = x[:,3]
    costh = torch.cos(theta); sinth = torch.sin(theta)
    force = u[:,0]

    temp = (force + polemass_len * th_dot**2 * sinth) / total_mass
    thetaacc = (g * sinth - costh * temp) / (length * (4.0/3.0 - masspole * costh**2 / total_mass))
    xacc = temp - polemass_len * thetaacc * costh / total_mass
    return torch.stack([x_dot, xacc, th_dot, thetaacc], dim=1)

def gym_euler_step_torch(x, u, dt=DT):
    # Euler 更新，與 Gym 的 step 相同
    dx = f_cont_torch(x, u)
    x_next = x + dt * dx
    return torch_wrap_angle_inplace(x_next)

# =========================
# Policy：u = pi(x)
# =========================
class Policy(nn.Module):
    def __init__(self, nx=4, hidden=128, u_max=FORCE_MAG):
        super().__init__()
        self.u_max = float(u_max)
        self.net = nn.Sequential(
            nn.Linear(nx, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1), nn.Tanh()
        )
        # 小初始化，動作更溫和
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.u_max * self.net(x)

# =========================
# PINN 風格 rollout loss
# （任務成本 + 物理殘差 + 早停遮罩 + ΣΔ量化器）
# =========================
def rollout_loss(policy, x0_batch, T, dt, Q, R, Qf,
                 u_smooth=0.0, u_max=FORCE_MAG,
                 lambda_phys=0.0, use_phys=True):
    """
    - 以 Gym 一致的 Euler 前向
    - ΣΔ 量化器在訓練中生效
    - 早停遮罩 + 固定死亡懲罰
    """
    B = x0_batch.shape[0]
    x = x0_batch
    alive = torch.ones((B,1), dtype=torch.float32, device=x0_batch.device)

    # 成本累加（每步平均）
    task_sum = torch.zeros((), device=x0_batch.device)
    phys_sum = torch.zeros((), device=x0_batch.device)

    # 量化器（與 Gym 對齊）
    q = SigmaDeltaQuantizer(force_mag=FORCE_MAG, leak=0.95, device=x0_batch.device, batch=B)

    u_prev = None
    for k in range(T):
        u_cont = torch.clamp(policy(x), -u_max, u_max)  # 策略輸出（連續）
        u = q.step(u_cont)                              # 量化成 ±FORCE_MAG
        x_next = gym_euler_step_torch(x, u, dt)

        # 任務成本（二次型；只對還活著的樣本）
        quad_x = torch.sum(x @ Q * x, dim=1, keepdim=True)  # (B,1)
        quad_u = torch.sum(u @ R * u, dim=1, keepdim=True)
        task_sum += torch.mean(alive * (quad_x + quad_u))

        # 物理殘差（Euler 殘差；可關閉/調權重）
        if use_phys and lambda_phys > 0.0:
            r = (x_next - x)/dt - f_cont_torch(x, u)
            # 逐維尺度（避免單位差異拉炸）
            S = torch.tensor([0.5, 1.0, 0.1745, 1.0], dtype=torch.float32, device=x0_batch.device)
            r_n = r / S
            phys_sum += torch.mean(alive * torch.sum(r_n*r_n, dim=1, keepdim=True))

        # 控制平滑（可選）
        if u_smooth > 0 and u_prev is not None:
            du = u - u_prev
            task_sum += u_smooth * torch.mean(alive * (du*du))
        u_prev = u

        # 早停遮罩（與 Gym 臨界一致）
        bad = ((torch.abs(x_next[:,2]) > THETA_MAX) | (torch.abs(x_next[:,0]) > X_MAX)).float().unsqueeze(1)
        alive = alive * (1.0 - bad)

        x = x_next

    # 終端成本（只對活著的）
    term = torch.sum(x @ Qf * x, dim=1, keepdim=True)
    task_sum += torch.mean(alive * term)

    # 對死亡樣本給固定懲罰（不把爆炸後續全灌進成本）
    DEAD_PENALTY = 10.0
    task_sum += DEAD_PENALTY * torch.mean(1.0 - alive)

    # 每步平均
    task = task_sum / (T+1)
    phys = phys_sum / max(T,1)
    loss = task + lambda_phys * phys
    return loss, task.detach().squeeze(), phys.detach().squeeze()

# =========================
# 訓練（短→長課程，與 Gym 對齊）
# =========================
def train_policy(epochs=500, batch_size=256, dt=DT,
                 T_min=0.5, T_max=4.0, growth_every=50,
                 u_max=FORCE_MAG, lambda_phys0=0.0,
                 use_phys=True, u_smooth=1e-3, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    policy = Policy(u_max=u_max).to(device)

    # 任務成本權重（可依需求調，但保持合理尺度）
    Q  = torch.tensor(np.diag([0.2, 0.05, 6.0, 1.0]), dtype=torch.float32, device=device)
    R  = torch.tensor([[0.05]], dtype=torch.float32, device=device)  # 小一點，避免動作被壓太小
    Qf = torch.tensor(np.diag([0.5, 0.0, 15.0, 3.0]), dtype=torch.float32, device=device)

    opt = optim.Adam(policy.parameters(), lr=1e-4, weight_decay=1e-6)

    for ep in range(1, epochs+1):
        # 自適應課程（每 growth_every 個 epoch 拉長視界，最多 T_max）
        T_sec = min(T_min + 0.5 * ((ep-1)//growth_every), T_max)
        T = int(T_sec/dt)
        lambda_phys = lambda_phys0  # 若要逐步提高，可以在這裡按 ep 增加

        # 初始分佈對齊 Gym reset：Uniform(-0.05,0.05)
        x0 = np.random.uniform(-0.05, 0.05, size=(batch_size,4)).astype(np.float32)
        x0 = torch.tensor(x0, dtype=torch.float32, device=device)

        loss, task_cost, phys_cost = rollout_loss(
            policy, x0, T, dt, Q, R, Qf,
            u_smooth=u_smooth, u_max=u_max,
            lambda_phys=lambda_phys, use_phys=use_phys
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        opt.step()

        if ep % 10 == 0 or ep == 1:
            print(f"[ep {ep:03d}] loss={loss.item():.4e}  task={task_cost.item():.4e}  "
                  f"phys={phys_cost.item():.4e}  T={T_sec:.1f}s  lr={opt.param_groups[0]['lr']:.1e}")

    return policy, (Q,R,Qf), dt

# =========================
# 視覺化（數值模擬）
# =========================
def eval_and_plot(policy, dt=DT, T_sec=5.0):
    import matplotlib.pyplot as plt
    T = int(T_sec/dt)

    # 測試初始狀態（微小偏置）
    x = torch.tensor([[0.0, 0.0, np.deg2rad(5.0), 0.0]], dtype=torch.float32, device=device)
    q = SigmaDeltaQuantizer(force_mag=FORCE_MAG, leak=0.95, device=device, batch=1)

    X = [x.detach().cpu().numpy()[0]]
    U = []
    for k in range(T):
        u_cont = torch.clamp(policy(x), -FORCE_MAG, FORCE_MAG)
        u = q.step(u_cont)
        x = gym_euler_step_torch(x, u, dt)
        X.append(x.detach().cpu().numpy()[0])
        U.append(u.detach().cpu().numpy()[0,0])

    X = np.asarray(X); U = np.asarray(U); t = np.arange(T+1)*dt
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3,1, figsize=(8,8), sharex=True)
    axs[0].plot(t, X[:,2]*180/np.pi); axs[0].axhline(0, color='k', lw=0.5)
    axs[0].set_ylabel("theta [deg]"); axs[0].grid(True)
    axs[1].plot(t, X[:,0]); axs[1].axhline(0, color='k', lw=0.5)
    axs[1].set_ylabel("x [m]"); axs[1].grid(True)
    axs[2].plot(t[:-1], U)
    axs[2].set_ylabel("u [N]"); axs[2].set_xlabel("time [s]"); axs[2].grid(True)
    plt.suptitle("PINN policy with Gym-matched Euler & ΣΔ quantizer")
    plt.show()

# =========================
# Gym 播放（pwm/bangbang）
# =========================
def run_on_gym(policy, T_steps=int(6.0/DT), mode="pwm"):
    import gymnasium as gym
    env = gym.make("CartPole-v1", render_mode="human")

    dt_env = getattr(env.unwrapped, "tau", DT)
    force_mag = getattr(env.unwrapped, "force_mag", FORCE_MAG)

    obs, _ = env.reset(seed=0)
    sigma = 0.0
    leak  = 0.95
    total_r = 0.0

    for k in range(T_steps):
        x_t = torch.tensor(obs[None,:], dtype=torch.float32, device=device)
        u_cont = policy(x_t).detach().cpu().numpy()[0,0]
        u_cont = float(np.clip(u_cont, -2*force_mag, 2*force_mag))

        if mode == "bangbang":
            action = 1 if u_cont >= 0.0 else 0
        else:
            d = float(np.clip(u_cont/force_mag, -1.0, 1.0))
            y = 1.0 if sigma >= 0.0 else -1.0
            sigma = leak * sigma + (d - y)
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

    print(f"[Gym] return={total_r}, steps={k+1}, dt_env={dt_env}, mode={mode}")
    env.close()

# =========================
# 主程式
# =========================
if __name__ == "__main__":
    # 訓練：與 Gym 對齊（Euler, dt=0.02, ±10N, Gym 門檻/初始化）
    policy, (Q,R,Qf), dt = train_policy(
        epochs=1000, batch_size=256, dt=DT,
        T_min=0.5, T_max=10.0, growth_every=50,
        u_max=FORCE_MAG,
        use_phys=True,          # ← 開啟物理殘差
        lambda_phys0=1e-4,      # ← 從很小開始
        u_smooth=1e-3, seed=42
    )

    # 可視化（數值）
    try:
        eval_and_plot(policy, dt, T_sec=10.0)
    except Exception as e:
        print("Plot skipped:", e)

    # Gym 播放（連續→ΣΔ→離散）
    try:
        run_on_gym(policy, T_steps=int(10.0/DT), mode="pwm")
    except Exception as e:
        print("Gym skipped:", e)
