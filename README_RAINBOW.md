# Rainbow Robot for Paint Task

This repository contains the implementation of a Paint task using the Rainbow RB10 1300E robot arm instead of the Franka Panda robot.

## Overview

The Rainbow RB10 1300E is a 6-DOF industrial robot arm without a gripper. In this implementation:
- The robot is configured to perform painting motions along a tilted wall
- The robot uses joint position control 
- End-effector tracking is done via the tool link instead of a gripper

## Setup

1. Make sure the Rainbow robot USD file is located at:
   ```
   C:\Users\tsk\Downloads\rainbow\rb10_1300e\rb10_1300e.usd
   ```

2. The robot asset is defined in:
   ```
   source/isaaclab_assets/isaaclab_assets/robots/rainbow.py
   ```

3. The Paint task configuration is in:
   ```
   source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/paint/config/franka/paint_joint_pos_env_cfg.py
   ```

## Running the Task

You can run the Rainbow paint task using the provided script:

```bash
python scripts/run_rainbow_paint.py
```

Additional options:
- `--num_envs <N>`: Set the number of environments to run (default: 1)
- `--task <TASK_NAME>`: Specify a different task name (default: "Isaac-Paint-Rainbow-v0")
- `--headless`: Run in headless mode without visualization

## Key Differences from Franka Robot

1. **No Gripper:**
   - The Rainbow robot has no gripper, so painting operations are performed by proximity
   - Gripper-related observations and actions are disabled

2. **Joint Configuration:**
   - The Rainbow robot has 6 joints (rb10_joint1 through rb10_joint6)
   - Default pose is set to all zeros, but can be adjusted as needed

3. **End Effector:**
   - The end effector is defined at the rb10_tool_link
   - Tracking is done through this tool link for painting operations

## Custom Configurations

The implementation supports several configuration options:
- Modify the pose and orientation of the robot
- Adjust PD control parameters
- Change the tilted wall configuration
- Update joint limits and control parameters

## Troubleshooting

If you encounter issues:
1. Check that the USD file path is correct
2. Ensure all Isaac Lab dependencies are installed
3. Verify that the joint names match those in the USD file
4. Check for any linter errors indicating incorrect imports

For more information, refer to the Isaac Lab documentation. 