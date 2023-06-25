# Rat-robot: NERMO

Different test cases in Mujoco

# Getting started
### Environment: 
System with Python 3 (Personal recommendation Ubutnu 20.04 or Ubuntu 22.04)

### Dependencies
Package: <strong>matplotlib</strong>  and <strong>numpy</strong>
 
Install command: `pip3 install <package name>`

### Mujoco 
- Version: 2.1.0 in https://github.com/deepmind/mujoco/releases/tag/2.1.0
- Python package: Mujoco_py in https://github.com/openai/mujoco-py
- Install process: please follow the readme in https://github.com/openai/mujoco-py

# Run experiments

There are four test cases in the folders
- Balance: To test the balance of the robot using different controllers.
- Maze: To test the robot traveling maze using different gaits.
- GoStraight: Ask the robot to go straight using different gaits with different stride frequencies.
- Turn: Ask the robot to turn around using different gaits with different stride frequencies.

For test cases, there are several parameters to modify the robot gait in simulation that can be set in the simulation starts.
