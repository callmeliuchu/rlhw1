import argparse
import pickle
import time

import gym
import numpy as np


def load_expert_data_for_env(path, action_dim):
    """
    Load expert data and extract (observations, actions) trajectories.
    Uses action_dim to robustly pick the correct array for actions.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    def split_traj(traj):
        # Preferred: explicit keys
        if isinstance(traj, dict):
            obs = traj.get("observations", traj.get("obs"))
            acs = traj.get("actions", traj.get("acs"))
            if obs is not None and acs is not None:
                return obs, acs

            # Fallback: infer from array-like fields using action_dim
            array_fields = [
                v for v in traj.values() if isinstance(v, (list, tuple, np.ndarray))
            ]
            if not array_fields:
                raise ValueError(
                    f"Trajectory dict has no array-like fields. Keys: {list(traj.keys())}"
                )

            # Choose actions as the first field whose last dim == action_dim
            acs = None
            obs = None
            for v in array_fields:
                arr = np.asarray(v)
                if arr.ndim >= 2 and arr.shape[-1] == action_dim and acs is None:
                    acs = arr
                elif obs is None:
                    obs = arr

            if obs is None or acs is None:
                raise ValueError(
                    "Could not infer observations/actions from trajectory dict. "
                    f"Available keys: {list(traj.keys())}"
                )
            return obs, acs

        # Tuple/list like (obs, acs, ...)
        if isinstance(traj, (list, tuple)) and len(traj) >= 2:
            return traj[0], traj[1]

        raise ValueError(f"Unrecognized trajectory element type: {type(traj)}")

    # Case 1: dict representing a whole dataset
    if isinstance(data, dict):
        # Might already be stacked arrays
        obs = data.get("observations", data.get("obs"))
        acs = data.get("actions", data.get("acs"))
        if obs is not None and acs is not None:
            if isinstance(obs, list):
                return obs, acs
            return [obs], [acs]

        # Or dict with per-trajectory entries
        trajs = data.get("trajectories") or data.get("paths")
        if trajs is None:
            raise ValueError(
                "Expert data dict does not contain 'observations'/'actions', "
                "'obs'/'acs', or 'trajectories'/'paths'."
            )

        obs_trajs, act_trajs = [], []
        for traj in trajs:
            o, a = split_traj(traj)
            obs_trajs.append(o)
            act_trajs.append(a)
        return obs_trajs, act_trajs

    # Case 2: list of trajectories
    if isinstance(data, list):
        obs_trajs, act_trajs = [], []
        for traj in data:
            o, a = split_traj(traj)
            obs_trajs.append(o)
            act_trajs.append(a)
        return obs_trajs, act_trajs

    raise ValueError("Unrecognized expert data format (expected dict or list).")


def main(args):
    # Match the environment used in the homework (e.g., Ant-v4, Hopper-v4, etc.)
    env = gym.make(args.env_name)
    action_dim = env.action_space.shape[0]

    obs_trajs, act_trajs = load_expert_data_for_env(args.expert_data, action_dim)

    for traj_idx, (obs_traj, act_traj) in enumerate(zip(obs_trajs, act_trajs)):
        print(
            f"Playing trajectory {traj_idx + 1}/{len(obs_trajs)} "
            f"with length {len(act_traj)}"
        )
        env.reset()

        for t, a in enumerate(act_traj):
            action = np.array(a).reshape(-1)
            # Clip to valid action range just in case
            low, high = env.action_space.low, env.action_space.high
            if low is not None and high is not None:
                action = np.clip(action, low, high)

            _, _, done, _ = env.step(action)
            env.render()
            time.sleep(args.dt)

            if done:
                print(f"Episode finished at step {t + 1}")
                break

        if args.one_traj:
            break

    # Keep the window open until user confirms
    try:
        input("Finished playing trajectories. Press Enter to close the window...")
    except EOFError:
        # In case input is not available (e.g., some notebook environments),
        # just proceed to close.
        pass

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        required=True,
        help="Environment name, e.g. Ant-v4, Hopper-v4, HalfCheetah-v4, Walker2d-v4",
    )
    parser.add_argument(
        "--expert_data",
        type=str,
        required=True,
        help="Path to expert_data_*.pkl file",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.02,
        help="Time delay between steps (seconds)",
    )
    parser.add_argument(
        "--one_traj",
        action="store_true",
        help="Only play the first trajectory",
    )
    main(parser.parse_args())

