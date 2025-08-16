import gymnasium as gym
import numpy as np
from typing import Tuple, Optional


class CartPoleEnv:
    """Thin wrapper around Gymnasium CartPole-v1 to normalize return formats."""

    def __init__(
        self,
        max_episode_steps: int = 10000,
        dt: float = 0.02,
        force_mag: float = 10.0,
        render_mode: Optional[str] = None,
    ):
        self.env = gym.make("CartPole-v1", render_mode=render_mode)
        self.env._max_episode_steps = max_episode_steps
        self.env.tau = dt
        self.env.force_mag = force_mag

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset and return the initial observation as float array."""
        obs, _ = self.env.reset(seed=seed)
        return np.array(obs, dtype=float)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Step the environment with a discrete action.

        Returns
        -------
        obs : np.ndarray
            Next observation.
        reward : float
            Immediate reward.
        done : bool
            Whether the episode has terminated or been truncated.
        """
        step_out = self.env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, _ = step_out
            done = bool(terminated or truncated)
        else:
            obs, reward, done, _ = step_out
        return np.array(obs, dtype=float), float(reward), done

    def close(self):
        """Close the underlying environment."""
        self.env.close()
