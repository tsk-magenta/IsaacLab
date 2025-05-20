#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a Rainbow Paint environment."""

import argparse
import isaaclab as il
import gym


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Run a Rainbow robot painting environment.")
    parser.add_argument(
        "--num_envs", type=int, default=1, help="Number of environments. Defaults to 1."
    )
    parser.add_argument(
        "--task", type=str, default="Isaac-Paint-Rainbow-v0", help="Task name. Defaults to Isaac-Paint-Rainbow-v0."
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run in headless mode. Defaults to False."
    )
    return parser.parse_args()


def main():
    """Main."""
    args = parse_args()

    # Launch the simulator
    sim_cfg = il.SimulatorCfg(headless=args.headless)
    sim_app = il.SimulatorApp(sim_cfg)

    # Create the environment
    env = gym.make(args.task, num_envs=args.num_envs)

    # Reset the environment and perform random actions
    obs, info = env.reset()
    done = False
    step_count = 0

    # Run the environment until closed
    while sim_app.is_running():
        # Get an action: simple random actions
        action = env.action_space.sample()
        
        # Apply the action to the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check if the episode is done
        done = terminated or truncated
        
        # Reset if done
        if done.any():
            env_ids = done.nonzero(as_tuple=False).squeeze(-1)
            obs, info = env.reset(env_ids)
            
        # Increment step count
        step_count += 1
        
        # Print progress
        if step_count % 100 == 0:
            print(f"Step: {step_count}")
    
    # Close the environment
    env.close()
    
    # Close the simulator
    sim_app.close()


if __name__ == "__main__":
    main() 