from argparse import ArgumentParser, Namespace
from typing import Optional

import numpy as np
from omegaconf import OmegaConf

from src.env import CartPoleEnv
from src.params import ControlParams
from src.lqr import BaseLQR, ContinuousLQR, DiscreteLQR
from src.ilqr import ILQRController, to_vec, to_scalar
# from src.pinn import train_policy, PolicyPWMController


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="clqr",
    )
    return parser.parse_args()


class Runner:
    def __init__(
        self,
        controller: BaseLQR,
        max_episode_steps: int = 10000,
        dt: float = 0.02,
        force_mag: float = 10.0,
        render_mode="human",
    ):
        self.controller = controller
        self.env = CartPoleEnv(
            max_episode_steps=max_episode_steps,
            dt=dt,
            force_mag=force_mag,
            render_mode=render_mode,
        )

    def run(self, n_episodes=1, max_steps=10000, seed_offset=0, deadband=0.0):
        for ep in range(n_episodes):
            self.env.start_recording(fps=30, step_capture="post")
            obs = self.env.reset(seed=seed_offset + ep)
            if hasattr(self.controller, "reset_episode"):
                self.controller.reset_episode()
            total_r = 0.0
            for _ in range(max_steps):
                action = self.controller.action(obs, deadband=deadband)
                out = self.env.step(action)

                if len(out) == 5:
                    obs, r, terminated, truncated, _ = out
                    done = terminated or truncated
                else:
                    obs, r, done = out
                total_r += r
                if done:
                    break
            print(f"[Runner] Episode {ep}: return={total_r}")

        self.env.stop_recording()
        self.env.save_gif("cartpole_run.gif")
        self.env.close()


class ILQRMPCRunner:
    """
    Receding-horizon iLQR (MPC) runner.
    At each env step: re-plan from current state for a short horizon, apply only the first control.
    """
    def __init__(
        self,
        ilqr: ILQRController,
        max_episode_steps: int = 10000,
        dt: float = 0.02,
        force_mag: float = 10.0,
        render_mode="human",
        T_hor: float = 2.0,
    ):
        self.ilqr = ilqr
        self.force_mag = force_mag
        self.env = CartPoleEnv(max_episode_steps, dt=dt, force_mag=force_mag, render_mode=render_mode)
        # horizon length aligned with env dt
        self.N_hor = int(T_hor / dt)
        assert self.N_hor >= 5, "Horizon too short."

        # warm-start buffer for U (N_hor, 1)
        self.U_ws = np.zeros((self.N_hor, 1), dtype=float)

    def run(self, max_steps: int = 1000, seed: int = 0, mode: str = "bangbang") -> float:
        obs = self.env.reset(seed=seed)
        self.ilqr.reset_pwm()
        total_r = 0.0

        for t in range(max_steps):
            # Re-plan from current state with warm-start
            X_opt, U_opt, K_seq, k_seq, J = self.ilqr.plan(obs, self.U_ws)

            # First control (continuous)
            u0 = float(U_opt[0, 0] + k_seq[0] + to_scalar(K_seq[0] @ (to_vec(obs) - to_vec(X_opt[0]))))
            # Hard clip to env's force magnitude (since real actuator is ±force_mag)
            u0 = float(np.clip(u0, -self.force_mag, self.force_mag))

            # Map to discrete action
            if mode == "bangbang":
                action = 1 if u0 >= 0.0 else 0
            else:  # sigma-delta pwm on single step (still ok as a dither)
                d = np.clip(u0 / self.force_mag, -1.0, 1.0)
                y = 1.0 if self.ilqr._sigma >= 0.0 else -1.0
                self.ilqr._sigma += d - y
                action = 1 if y > 0 else 0

            obs, r, done = self.env.step(action)
            total_r += r
            if done:
                break

            # Warm-start for next step: shift left and append zero
            self.U_ws[:-1, 0] = U_opt[1:, 0]
            self.U_ws[-1, 0] = 0.0

        print(f"[iLQR-MPC/{mode}] return={total_r}, steps={t+1}")
        self.env.close()
        return total_r


if __name__ == "__main__":
    args = parse_arguments()
    config = OmegaConf.load(f"configs/{args.config}.yaml")
    params = ControlParams(**config.params)

    match args.config:
        case "clqr":
            clqr = ContinuousLQR(
                A=params.A,
                B=params.B,
                Q=params.Q,
                R=params.R,
            )
            Kc = clqr.solve()
            runner = Runner(clqr, **config.runner_args)
            runner.run(**config.run_params)

        case "dlqr":
            dlqr = DiscreteLQR(
                A=params.A,
                B=params.B,
                Q=params.Q,
                R=params.R,
                dt=config.runner_args.dt,
            )
            Kd = dlqr.solve()
            runner = Runner(dlqr, **config.runner_args)
            runner.run(**config.run_params)

        case "ilqr":
            ilqr = ILQRController(
                g=config.params.g,
                m_c=config.params.m_c,
                m_p=config.params.m_p,
                l=config.params.l,
                Q=params.Q,
                R=params.R,
                Qf=params.Qf,
                dt=config.runner_args.dt,
                **config.controller,
            )
            mpc = ILQRMPCRunner(ilqr, **config.runner_args)
            mpc.run(**config.run_params)

        # case "pinn":
        #     THETA_MAX = 12.0 * np.pi / 180.0
        #     policy, (Qp, Rp, Qfp) = train_policy(
        #         epochs=config.trainer.epochs,
        #         batch_size=config.trainer.batch_size,
        #         T_min=config.trainer.T_min,
        #         T_max=config.trainer.T_max,
        #         growth_every=config.trainer.growth_every,
        #         use_phys=config.trainer.use_phys,
        #         lambda_phys0=config.trainer.lambda_phys0,
        #         u_smooth=config.trainer.u_smooth,
        #         seed=config.trainer.seed,
        #         g=config.params.g,
        #         masscart=config.params.m_c,
        #         masspole=config.params.m_p,
        #         length=config.params.l,
        #         dt=config.runner_args.dt,
        #         u_max=config.params.force_mag,
        #         force_mag=config.params.force_mag,
        #         theta_max=THETA_MAX,
        #         x_max=config.trainer.x_max,
        #         leak=config.trainer.leak,
        #     )
        #     pinn_ctrl = PolicyPWMController(
        #         policy,
        #         u_max=config.params.force_mag,
        #         force_mag=config.params.force_mag,
        #         leak=config.trainer.leak
        #     )
        #     runner = Runner(pinn_ctrl, **config.runner_args)
        #     runner.run(**config.run_params)

        case _:
            raise ValueError(f"Unknown method: {args.method}")
