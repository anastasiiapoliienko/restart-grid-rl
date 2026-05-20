"""Train a MaskablePPO policy on the priority-aware restoration env.

Usage:
    python scripts/train_priority.py --steps 750000 --seed 1 --out results/policy_priority_s1.zip
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor

from restart.env_priority import PriorityRestorationEnv


def mask_fn(env):
    return env.action_masks()


def make_env(seed: int):
    env = PriorityRestorationEnv(seed=seed)
    env = ActionMasker(env, mask_fn)
    env = Monitor(env)
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=750_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=str, default="results/policy_priority.zip")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    env = make_env(args.seed)
    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env,
        verbose=1,
        seed=args.seed,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.05,
    )
    model.learn(total_timesteps=args.steps)
    model.save(args.out)
    print(f"saved policy to {args.out}")


if __name__ == "__main__":
    main()
