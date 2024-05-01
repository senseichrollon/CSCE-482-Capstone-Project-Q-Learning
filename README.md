# CSCE 482 Semester Project - Autonomous Driving Simulation

## Overview

## Getting Started

### CARLA Simulator
For this project we are using the CARLA simulator to train our vehicle agent in a virtual environment. CARLA is a prominent open-source simulator for autonomous driving research.

Learn more about CARLA here:
https://carla.org/

[User Manual](UserManual.pdf)


## How to Train

Make sure CARLA Simulator is open, executable is called CarlaUE4.exe

### D3QN
Run the command:
py -3.8 frontend.py
- This will open a UI which allows you to select parameters to train your specific model.

- Uncomment the cv2.imshow in the process image function to visualize the actual camera sensor data.

### Preferences
Currently, we suggest you to use reward function 4 over the other reward functions because it seems to have better performance. Additionally, you could create your own reward functions to use for training.