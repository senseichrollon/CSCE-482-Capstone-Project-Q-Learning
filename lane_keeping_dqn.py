import argparse
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import vista
from copy import deepcopy
from vista.utils import transform
from vista.entities.agents.Dynamics import tireangle2curvature
from vista.utils import logging, misc
import random
import cv2
import matplotlib.pyplot as plt
import math

from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='In future, it will be an error for \'np.bool_\' scalars to be interpreted as an index')

NUM_ACTIONS = 6191

"""
Creating the DQN class
"""
class DQN(nn.Module):
    def __init__(self, action_dim):
        super(DQN, self).__init__()
        # Convolutional and pooling layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten the output of the final convolutional layer
        self.flatten_size = self._get_conv_output((3, 200, 320))
        # print(f"flatten_size = {self.flatten_size}")
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, action_dim)
        

    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.zeros(1, *shape)

            output = F.relu(self.pool1(self.conv1(input)))
            output = F.relu(self.pool2(self.conv2(output)))
            output = F.relu(self.pool3(self.conv3(output)))

            # print(f"output.shape = {output.shape}")
            return int(np.prod(output.size()))

    def forward(self, state):
        # print("__FUNCTION__forward")
        # print(f"\tstate.shape = {state.shape}")
        # Convert state to float and scale if necessary
        state = state.float() / 255.0  # Scale images to [0, 1]

        x = F.relu(self.pool1(self.conv1(state)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))

        # Flatten and pass through fully connected layer
        # print(f"\tx_reshaped = {x.shape}")
        x = x.reshape(x.size(0), -1)
        # print(f"\tx_reshaped = {x.shape}")
        x = F.relu(self.fc1(x))
        q_values = F.relu(self.fc2(x))
        # print(f"\tq_values.shape = {q_values.shape}")
        return q_values

"""
Creating the behavior and target neural networks
Initializing the loss function and optimizer
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network = DQN(NUM_ACTIONS).to(device)
target_network = deepcopy(network)

optimizer = torch.optim.Adam(network.parameters(), lr=1e-5)
loss_fn = nn.SmoothL1Loss() # huber loss


"""
Defining the environment class
"""
class environment:
    def __init__(
            self,
            trace_paths,
            trace_config,
            car_config,
            sensor_config
    ):
        self.world = vista.World(trace_paths, trace_config)
        self.agent = self.world.spawn_agent(car_config)
        self.agent.spawn_camera(sensor_config)

        self.distance = 0
        self.prev_xy = np.zeros((2, ))

        self.action_space = self.empty_action_space()
        # print(f"self.action_space = {self.action_space}")

    def empty_action_space(self):
        # creating an action space where curvature ranges from -0.2 to 0.2 and speed ranges from 0 to 15
        curvature_increment = 0.01
        speed_increment = 0.1

        curvature_range = np.arange(-0.2, 0.2+curvature_increment, curvature_increment)
        speed_range = np.arange(0, 15+speed_increment, speed_increment)

        curvature_grid, speed_grid = np.meshgrid(curvature_range, speed_range)
        return np.stack([curvature_grid.ravel(), speed_grid.ravel()], axis=1)
    
    def reset(self):
        self.world.reset()
        self.agent = self.world.agents[0]
        observations = self.agent.observations
        self.distance = 0
        self.prev_xy = np.zeros((2, ))
        self.action_idx = -1
        return observations
    
    def reward_3(self):
        """
        Continous reward function:
        r = c1*cos(theta) - c2*abs(P_y / W_d) - c3 * I_fail
        - c1,c2,c3 are coefficients with values 1,1,2 respectively
        - theta is the angle between road direction and vehicle
        - P_y is is the lateral position error between the road center and the gravity center of the vehicle
        - W_d is the lane width
        - I_fail = 1 if agent is out of lane and 0 otherwise
        First term is used to encourage staying along the road, second term is to encourage being centered,
        third term is to heavily penalize going off road
        """

        # angle between road tangent and vehicle
        agent_yaw = self.agent.ego_dynamics.yaw
        road_dir = self.agent.human_dynamics.yaw
        theta = abs(agent_yaw - road_dir)

        # Normalize theta to [-pi, pi]
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        p_y = self.agent.relative_state.y
        W_d = self.agent.trace.road_width / 2.
        
        # i_fail is 1 if out of lane, else 0
        i_fail = 1 if np.abs(self.agent.relative_state.x) > W_d else 0

        # Reward calculation
        reward = math.cos(theta) - abs(p_y / W_d) - (2 * i_fail)

        return reward, self.agent.done


    
    def step(self, action, dt = 1/30):
        self.agent.step_dynamics(action, dt=dt)
        self.agent.step_sensors()
        next_state = self.agent.observations
        
        # get other info
        info = misc.fetch_agent_info(self.agent)
        
        # Update car ego info
        current_xy = self.agent.ego_dynamics.numpy()[:2]
        self.distance += np.linalg.norm(current_xy - self.prev_xy)
        self.prev_xy = current_xy
        info['distance'] = self.distance

         # Calculate lateral distance from road center
        lateral_distance = np.abs(self.agent.relative_state.x)
        road_half_width = self.agent.trace.road_width / 2.
        
        # Set a threshold for being too far off the road
        max_allowed_distance = road_half_width * 3  # Example threshold

        # Check if the agent is too far off the road
        too_far_off_road = lateral_distance > max_allowed_distance
        done = self.agent.done or too_far_off_road

        # reward, _ = self.reward_1() if self.rf == 1 else self.reward_2()
        reward, _ = self.reward_3()

        return next_state, reward, done, info
    

    def epsilon_greedy_action(self, state, epsilon):
        # Restructuring the states to match the input of the conv layers
        # print("__FUNCTION__, epsilon_greedy_action")
        # print(f"\tstate = {state}")
        # print(f"\tstate.shape = {state.shape}")
        state = state.permute(0, 3, 1, 2)
        # print(f"\tstate.shape = {state.shape}")
        prob = np.random.uniform()

        if prob < epsilon:
            self.action_idx = np.random.randint(len(self.action_space))
            return self.action_space[self.action_idx]
        else:
            qs = network.forward(state).cpu().data.numpy()
            self.action_idx = np.argmax(qs)
            return self.action_space[self.action_idx]

"""
Replay buffer class
"""
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def store(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

def optimize_model(memory, batch_size, gamma):
    # print("__FUNCTION__optimize_model()")
    if memory.size() < batch_size:
        return
    
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    # convert to tensors and move to device
    state_batch = torch.cat([s for s in batch.state]).to(device)
    # action_batch = torch.cat([a for a in batch.action]).to(device)
    action_batch = torch.cat([torch.tensor([a]).to(device) for a in batch.action])
    reward_batch = torch.cat([torch.tensor([r]).to(device) for r in batch.reward])
    next_state_batch = torch.cat([s for s in batch.next_state if s is not None]).to(device)
    done_batch = torch.cat([torch.tensor([d]).to(device) for d in batch.done])
    # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool).to(device)
    
    # print(f"\tstate_batch.shape = {state_batch.shape}")
    state_batch = state_batch.permute(0, 3, 1, 2)
    next_state_batch = next_state_batch.permute(0, 3, 1, 2)
    # print(f"\tstate_batch.shape = {state_batch.shape}")
    # print(f"\taction_batch.shape = {action_batch.shape}")
    # print(f"\treward_batch.shape = {reward_batch.shape}")
    # print(f"\tnext_state_batch.shape = {next_state_batch.shape}")
    # print(f"\tdone_batch.shape = {done_batch.shape}")
    # Compute Q
    current_q = network(state_batch)
    # print("  __FUNCTION__optimize_model()")
    # print(f"\tcurrent_q.shape = {current_q.shape}")
    # print(f"\taction_batch.unsqueeze(1).shape = {action_batch.unsqueeze(1).shape}")
    # print(f"\taction_batch.unsqueeze(1).long().shape = {action_batch.unsqueeze(1).long().shape}")
    current_q = torch.gather(current_q, dim=1, index=action_batch.unsqueeze(1).long()).squeeze(-1)
    # print(f"\tcurrent_q.shape = {current_q.shape}")
    # print(f"\tcurrent_q = {current_q}")

    # print(f"\tlen(next_state_batch) = {len(next_state_batch)}")

    with torch.no_grad():
        # compute target Q
        target_q = torch.full([len(next_state_batch)], 0, dtype=torch.float32)
        # target_q = torch.zeros(batch_size, NUM_ACTIONS, dtype=torch.float32)

        for idx in range(len(next_state_batch)):
            reward = reward_batch[idx]
            next_state = next_state_batch[idx].unsqueeze(0)
            done = done_batch[idx]
            if (done):
                target_q[idx] = reward
                # print(f"\t\t\treward = {reward}")
            else:
                # print(f"\tstate_batch.shape = {state_batch.shape}")
                # print(f"\t\t\tnext_state.shape = {next_state.shape}")
                q_values = target_network(next_state)
                # print(f"\t\t\tt_q_values.shape = {q_values.shape}")
                max_q = q_values[0][torch.argmax(q_values)]
                target = reward + gamma * max_q
                target_q[idx] = target

    # print(f"\ttarget_q.shape = {target_q.shape}")

    # Compute Huber loss
    loss_q = loss_fn(current_q, target_q)

    # Optimize the model
    optimizer.zero_grad()
    loss_q.backward()
    optimizer.step()

if __name__ == '__main__':
    """
    Defining environment configurations 
    """
    parser = argparse.ArgumentParser(
        description='Run the simulator with random actions')
    parser.add_argument('--trace-path',
                        type=str,
                        nargs='+',
                        help='Path to the traces to use for simulation',
                        required=True)
    parser.add_argument('--version',
                        type=str,
                        nargs='+',
                        help='version number for model naming',
                        required=False)
    parser.add_argument('--operation',
                        type=str,
                        nargs='+',
                        help='Load or New',
                        required=True)
    parser.add_argument('--save-path',
                        type=str,
                        nargs='+',
                        help='Path to the saved model state',
                        required=False)
    args = parser.parse_args()
    trace_config={'road_width': 4}
    car_config={
            'length': 5.,
            'width': 2.,
            'wheel_base': 2.78,
            'steering_ratio': 14.7,
            'lookahead_road': True
        }
    sensor_config={
        'size': (200, 320),
    }

    env = environment(args.trace_path, trace_config, car_config, sensor_config)
    display = vista.Display(env.world)

    if args.operation[0].lower() == 'new':

        """
        Initializing hyper-parameters and beginning the training loop
        """
        replay_buffer = ReplayBuffer(10000)
        batch_size = 32
        gamma = 0.99 
        epsilon_start = 1.0
        epsilon_end = 0.01
        epsilon_decay = 0.96
        num_episodes = 200
        target_update = 10  # Update target network every 10 episodes
        max_num_steps = 700

        best_dict_reward = -1e10

        # per episode
        rewards = np.array([])
        num_steps = np.array([])

        epsilon = epsilon_start
        for episode in range(num_episodes):
            state = env.reset()['camera_front']
            # print(f"main, state.shape after reset = {state.shape}")
            # print(state)
            display.reset()
            total_reward = 0
            done = False
            step = 0

            while step < max_num_steps and not done:
                # Convert state to the appropriate format and move to device
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)

                # Select action using epsilon greedy policy
                action = env.epsilon_greedy_action(state_tensor, epsilon)
                next_state, reward, done, _ = env.step(action)
                next_state = next_state['camera_front']

                # Convert next_state to tensor and move to device
                next_state_tensor = torch.from_numpy(next_state).unsqueeze(0).to(device) if next_state is not None else None

                # Store the transition in the replay buffer
                # action_tensor = torch.zeros(1, NUM_ACTIONS, dtype=torch.int64)
                # print(f"action_tensor = {action_tensor}")
                # print(f"action_tensor.shape = {action_tensor.shape}")
                # print(f"action = {action}")
                # print(f"action.shape = {action.shape}")
                # print(f"env.action_idx = {env.action_idx}")
                # action_tensor[0][env.action_idx] = 1

                replay_buffer.store((state_tensor, env.action_idx, reward, next_state_tensor, done))

                state = next_state
                total_reward += reward

                # vis_img = display.render()

                # Optimize the model if the replay buffer has enough samples
                optimize_model(replay_buffer, batch_size, gamma)

                if step % target_update == 0 or done:
                    target_network.load_state_dict(network.state_dict())

                step += 1
                # cv2.imshow(f'Car Agent in Episode {episode}', vis_img[:, :, ::-1])
                # cv2.waitKey(5)
            rewards = np.append(rewards, total_reward / step)
            num_steps = np.append(num_steps, step)

            if total_reward > best_dict_reward:
                print("Saving new best")
                torch.save(target_network.state_dict(), 'saves/v'+args.version[0]+'_best_dqn_network_nn_model.pth')
                best_dict_reward = total_reward

            print(f'Episode {episode}: Total Reward: {total_reward}, Epsilon: {epsilon}, NumSteps: {step}')

            # Update epsilon
            epsilon = max(epsilon_end, epsilon_decay * epsilon)
        
        # Save the model's state dictionary
        torch.save(target_network.state_dict(), 'saves/v'+args.version[0]+'_final_dqn_network_nn_model.pth')

        eps = np.arange(0, num_episodes)
        print(f"rewards = {rewards}")
        print(f"num_steps = {num_steps}")
        display.render()
        plt.show()
        # Create a line graph
        plt.plot(eps, rewards)

        # Add labels and a title
        plt.xlabel('Training Episodes')
        plt.ylabel('Average Reward per Episode')
        plt.title('Average Reward')

        # Display the plot
        plt.show()

        plt.plot(eps, num_steps)
        plt.xlabel('Training Episodes')
        plt.ylabel('Number of Steps per Episode')
        plt.title('Steps per Episode')
        # Display the plot
        plt.show()


    elif args.operation[0].lower() == 'load':
        target_network.load_state_dict(torch.load(args.save_path[0]))
        target_network.eval()
        max_num_steps = 1000
        state = env.reset()['camera_front']
        display.reset()
        total_reward = 0
        done = False
        step = 0
        while step < max_num_steps and not done:
            # Convert state to the appropriate format and move to device
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
            q_values = target_network(state_tensor.permute(0, 3, 1, 2)).cpu().data.numpy()
            action = env.action_space[np.argmax(q_values)]

            next_state, reward, done, _ = env.step(action)
            next_state = next_state['camera_front']

            # Convert next_state to tensor and move to device
            next_state_tensor = torch.from_numpy(next_state).unsqueeze(0).to(device) if next_state is not None else None
            next_state_tensor.to(device)

            state = next_state

            vis_img = display.render()

            total_reward += reward

            step += 1
            cv2.imshow(f'Evaluating Car Agent', vis_img[:, :, ::-1])
            cv2.waitKey(5)
        print(f"Total reward = {total_reward}; Steps = {step}")
    else:   
        print("incorrect operation command, only 'new' or 'load' allowed")
