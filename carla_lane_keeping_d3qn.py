import argparse
from collections import deque, namedtuple
import random
import numpy as np
import carla
import torch
import torch.nn as nn
import torch.nn.functional as F



"""
Creating the D3QN class that splits into two streams: advantage and value
"""
class DuelingDDQN(nn.Module):
    def __init__(self, action_dim):
        super(DuelingDDQN, self).__init__()
        # Convolutional and pooling layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten the output of the final convolutional layer
        self.flatten_size = self._get_conv_output((3, 200, 320))

        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 512)

        # State Value stream
        self.value_stream = nn.Linear(512, 1)

        # Advantage stream
        self.advantage_stream = nn.Linear(512, action_dim)

    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.zeros(1, *shape)
            output = self.conv1(input)
            output = self.pool1(output)
            output = self.conv2(output)
            output = self.pool2(output)
            output = self.conv3(output)
            output = self.pool3(output)
            return int(np.prod(output.size()))

    def forward(self, state):
        # Convert state to float and scale if necessary
        state = state.float() / 255.0  # Scale images to [0, 1]

        x = F.relu(self.pool1(self.conv1(state)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))

        # Flatten and pass through fully connected layer
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))

        # Value and advantage streams
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Combine to get Q-values
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

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
    
class Environment:
    def __init__(self, carla_client, car_config, sensor_config, reward_function):
        self.client = carla_client
        self.world = self.client.get_world()
        self.car_config = car_config
        self.sensor_config = sensor_config
        self.rf = reward_function

        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_bp = self.blueprint_library.filter(car_config['model'])[0]
        self.camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        self.camera_bp.set_attribute('image_size_x', str(sensor_config['image_size_x']))
        self.camera_bp.set_attribute('image_size_y', str(sensor_config['image_size_y']))
        self.camera_bp.set_attribute('fov', str(sensor_config['fov']))

        self.distance = 0
        self.prev_xy = np.zeros((2, ))

        # Action space is now defined in terms of throttle and steer instead of curvature and speed.
        throttle_range = np.linspace(0, 1, 10)
        steer_range = np.linspace(-1, 1, 10)
        self.action_space = np.array(np.meshgrid(throttle_range, steer_range)).T.reshape(-1, 2)

    def reset(self):
        # Spawn or respawn the vehicle at a random location
        if hasattr(self, 'vehicle'):
            self.vehicle.destroy()
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        self.vehicle = self.world.spawn_actor(self.vehicle_bp, spawn_point)

        # Attach the camera sensor
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(self.camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera.listen(lambda data: self.process_image(data))

        self.distance = 0
        self.prev_xy = np.array([self.vehicle.get_location().x, self.vehicle.get_location().y])

        # Start collecting data
        self.image = None
        while self.image is None:
            self.world.tick()

        return self.image

    def process_image(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.sensor_config['image_size_y'], self.sensor_config['image_size_x'], 4))
        self.image = i2[:, :, :3]

    def step(self, action):
        throttle, steer = action
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))

        self.world.tick()

        # Compute the distance traveled since the last step
        current_location = self.vehicle.get_location()
        current_xy = np.array([current_location.x, current_location.y])
        dd = np.linalg.norm(current_xy - self.prev_xy)
        self.distance += dd
        self.prev_xy = current_xy

        # Calculate reward based on the chosen reward function
        if self.rf == 1:
            reward, done = self.reward_1()
        elif self.rf == 2:
            reward, done = self.reward_2()
        else:
            reward, done = self.reward_3()

        info = {}  

        return self.image, reward, done, info

    def reward_1(self):
        return reward, done

    def reward_2(self):
        return reward, done

    def reward_3(self):
        return reward, done

    def epsilon_greedy_action(self, state, epsilon):
        return action_idx



if __name__ == 'main':
    parser = argparse.ArgumentParser(
        description='Run the simulator with random actions')
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
    