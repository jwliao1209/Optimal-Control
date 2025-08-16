import gymnasium as gym
import numpy as np
from typing import Tuple, Optional, Literal


class CartPoleEnv:
    """Thin wrapper around Gymnasium CartPole-v1 to normalize return formats,
    with optional GIF recording that hooks into reset()/step().
    """

    def __init__(
        self,
        max_episode_steps: int = 10000,
        dt: float = 0.02,
        force_mag: float = 10.0,
        render_mode: Optional[str] = None,
    ):
        self._max_episode_steps = max_episode_steps
        self._dt = dt
        self._force_mag = force_mag
        self._render_mode = render_mode

        self.env = gym.make("CartPole-v1", render_mode=render_mode)
        self.env._max_episode_steps = max_episode_steps
        self.env.tau = dt
        self.env.force_mag = force_mag

        # --- Recording state ---
        self._rec_enabled: bool = False
        self._rec_frames = []
        self._rec_fps: int = 30
        self._rec_on_reset: bool = True
        self._rec_when: Literal["pre", "post", "both"] = "post"
        self._rec_env = None  # a shadow env with render_mode="rgb_array"

    def start_recording(
        self,
        fps: int = 30,
        *,
        record_on_reset: bool = True,
        step_capture: Literal["pre", "post", "both"] = "post",
    ) -> None:
        """Enable recording. Frames are captured on reset()/step().
        If the current env cannot render frames, a shadow rgb_array env is created.
        """
        self._rec_enabled = True
        self._rec_frames = []
        self._rec_fps = int(fps)
        self._rec_on_reset = bool(record_on_reset)
        self._rec_when = step_capture

        # Prepare a render-capable env if needed
        if getattr(self.env, "render_mode", None) == "rgb_array":
            self._rec_env = self.env
        else:
            rec_env = gym.make("CartPole-v1", render_mode="rgb_array")
            rec_env._max_episode_steps = self._max_episode_steps
            rec_env.tau = self._dt
            rec_env.force_mag = self._force_mag
            self._rec_env = rec_env

    def stop_recording(self) -> None:
        """Disable recording. Keeps frames in memory so you can still save_gif()."""
        self._rec_enabled = False

    def save_gif(self, path: str) -> None:
        """Save recorded frames as a GIF."""
        if not self._rec_frames:
            raise RuntimeError("No recorded frames to save. Did you call start_recording()?")

        try:
            import imageio.v2 as imageio
        except Exception:
            import imageio  # fallback

        imageio.mimsave(path, self._rec_frames, fps=self._rec_fps)

    # ------------------------ Core env API ------------------------ #
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset and return the initial observation as float array."""
        obs, _ = self.env.reset(seed=seed)

        # Keep the recorder env in sync and capture an initial frame if requested
        if self._rec_enabled:
            self._sync_rec_env_state(obs, seed=seed)
            if self._rec_on_reset:
                self._capture_frame()

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
        if self._rec_enabled and self._rec_when in ("pre", "both"):
            # Capture the frame representing the *current* state before stepping
            self._sync_rec_env_state()
            self._capture_frame()

        step_out = self.env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, _ = step_out
            done = bool(terminated or truncated)
        else:
            obs, reward, done, _ = step_out

        if self._rec_enabled and self._rec_when in ("post", "both"):
            # Capture the frame representing the *next* state after stepping
            self._sync_rec_env_state(obs)
            self._capture_frame()

        return np.array(obs, dtype=float), float(reward), done

    def close(self):
        """Close the underlying environment and any shadow recorder env."""
        if self._rec_env is not None and self._rec_env is not self.env:
            self._rec_env.close()
        self.env.close()

    # ------------------------ Internal helpers ------------------------ #
    def _sync_rec_env_state(self, obs: Optional[np.ndarray] = None, seed: Optional[int] = None) -> None:
        """Synchronize the recorder env's internal state with the main env."""
        if not self._rec_enabled or self._rec_env is None:
            return

        # Ensure recorder env is reset at least once so it has valid internals
        if getattr(self._rec_env.unwrapped, "state", None) is None:
            # Safe reset (seed only on first time or if provided)
            _ = self._rec_env.reset(seed=seed)

        # CartPole exposes `unwrapped.state` as its dynamics state (x, x_dot, theta, theta_dot)
        # Mirror the main env state directly if available.
        if getattr(self.env.unwrapped, "state", None) is not None:
            self._rec_env.unwrapped.state = np.array(self.env.unwrapped.state, dtype=float)
        else:
            # Fallback: if we only have the latest obs, try to use it (CartPole obs == state)
            if obs is not None:
                self._rec_env.unwrapped.state = np.array(obs, dtype=float)

        # Keep key parameters aligned
        self._rec_env._max_episode_steps = self._max_episode_steps
        self._rec_env.tau = self._dt
        self._rec_env.force_mag = self._force_mag

    def _capture_frame(self) -> None:
        """Append a rendered RGB frame from the recorder env."""
        if self._rec_env is None:
            return
        frame = self._rec_env.render()
        if frame is not None:
            self._rec_frames.append(frame)
