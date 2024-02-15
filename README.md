# CSCE 482 Semester Project - Autonomous Driving Simulation

## Overview

## Getting Started

### VISTA Simulator
For this project we are using the VISTA simulator to train our vehicle agent in a virtual environment. VISTA is a data-based simulator that provides a flexible API to render environments and train models. 

Learn more about VISTA here:
https://vista.csail.mit.edu/ 

Setup VISTA following the instructions link below (**NOTE:** To create a Conda environment download the environment.yaml file from the VISTA GitHub repository [HERE](https://github.com/vista-simulator/vista/tree/main)).

We recommend not using the conda method on the VISTA site as the environment.yaml file provided is outdated. We created our own environment.yml file instead that doesn't have version issues.

https://vista.csail.mit.edu/getting_started/installation.html

## How to Train

### D3QN
Run the command:
python lane_keeping_d3qn.py --trace-path PATH-TO-TRACE-DIRECTORY --reward-function 1 or 2
- This will train the model for 300 episodes and save the best model as well as the final target model to your current directory
- Uncomment the cv2.imshow and display lines in the main training loop to visualize the simulation

### DQN
The command options are:
- trace-path : path to trace directory on your local machine
- version : what version you want prepended on your saved models
- operation : can be load or new (load to load a saved model or new to make a new one)
- save-path : if operation == load, then this is the path to that model