import argparse
from collections import deque, namedtuple
import random
import numpy as np
import carla
import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import cv2
import sys
from PIL import Image

"""
Replay buffer class
"""
NUM_ACTIONS = 50


# Carla Client attribute
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)  # Set a timeout in seconds for client connection
print("Client established") 



class DuelingDDQN(nn.Module):
    def __init__(self, action_dim, image_dim=(480,640)):
        super(DuelingDDQN, self).__init__()
        # Convolutional and pooling layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Flatten the output of the final convolutional layer
        self.flatten_size = self._get_conv_output((3, image_dim[0], image_dim[1]))

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
  #      print(f'Shapes of network, Value{value.shape}, advantage{advantage.shape}, q_values{q_values.shape}')
        return q_values


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network = DuelingDDQN(NUM_ACTIONS).to(device)
target_network = deepcopy(network)

optimizer = torch.optim.Adam(network.parameters(), lr=1e-5)
loss_fn = nn.SmoothL1Loss() # huber loss

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def store(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)
    
class HUD:
    def __init__(self, width, height):
        self.dim = (width, height)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.font_color = (255, 255, 255)
        self.line_height = 20
        self.x_offset = 10
        self.y_offset = 20
        self.speed = 0
        self.throttle = 0
        self.steer = 0
        self.heading = ''
        self.location = ''
        self.collision = []
        self.nearby_vehicles = []
        
    
    
    def update(self, speed, throttle, steer, heading, location, collision, nearby_vehicles):
        self.speed = speed
        self.throttle = throttle
        self.steer = steer
        self.heading = heading
        self.location = location
        self.collision = collision
        self.nearby_vehicles = nearby_vehicles
        
    def tick(self, camera_image):
        
        # Create a blank HUD image
        hud_image = np.zeros((self.dim[1], self.dim[0], 3), dtype=np.uint8)
        
        # Add black block on the left side
        hud_image[:, :100, :] = (0, 0, 0)

        # Add HUD elements
        cv2.putText(hud_image, f'Speed: {self.speed:.2f} m/s', (10, 40), self.font, self.font_scale, self.font_color, 1)
        cv2.putText(hud_image, f'Throttle: {self.throttle:.2f}', (10, 60), self.font, self.font_scale, self.font_color, 1)
        cv2.putText(hud_image, f'Steer: {self.steer:.2f}', (10, 80), self.font, self.font_scale, self.font_color, 1)
        cv2.putText(hud_image, f'Heading: {self.heading}', (10, 100), self.font, self.font_scale, self.font_color, 1)
        cv2.putText(hud_image, f'Location: {self.location}', (10, 120), self.font, self.font_scale, self.font_color, 1)
        cv2.putText(hud_image, 'Collision:', (10, 140), self.font, self.font_scale, self.font_color, 1)
        for i, value in enumerate(self.collision):
            cv2.putText(hud_image, f'{i}: {value:.2f}', (10, 160 + i * 20), self.font, self.font_scale, self.font_color, 1)
        cv2.putText(hud_image, f'Nearby vehicles:', (10, 160 + len(self.collision) * 20), self.font, self.font_scale, self.font_color, 1)
        for i, vehicle in enumerate(self.nearby_vehicles):
            cv2.putText(hud_image, f'{i}: {vehicle}', (10, 180 + (len(self.collision) + i) * 20), self.font, self.font_scale, self.font_color, 1)

        # Overlay HUD image onto camera image
        camera_image_with_hud = cv2.addWeighted(camera_image, 1, hud_image, 0.5, 0)
        
        return camera_image_with_hud

        
class Environment:
    def __init__(self, carla_client, car_config, sensor_config, reward_function, map=0, spawn_index=67):
        
        #Connecting to Carla Client
        self.client = carla_client
        self.world = self.client.get_world()

        # if loading specifc map
        if (map != 0):
            self.world= self.client.load_world(map)
        """ This portion can be moved to env.reset
         #delete what we created, eg. vehicles and sensors
        actor_list = self.world.get_actors()
        vehicle_and_sensor_ids = [actor.id for actor in actor_list if (('vehicle' in  actor.type_id) or ('sensor' in actor.type_id))]
        for id in vehicle_and_sensor_ids:   #delete all vehicles and cameras
            created_actor = self.world.get_actor(id)
            created_actor.destroy()
            print("Deleted", created_actor)
        """
        ## Setting environment attributes
        self.car_config = car_config
        self.sensor_config = sensor_config
        self.rf = int(reward_function[0])
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_bps = [bp for bp in self.blueprint_library.filter('vehicle') if bp.has_attribute('number_of_wheels')]
        self.vehicle_bp = self.vehicle_bps[0]
        self.camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        self.camera_bp.set_attribute('image_size_x', str(sensor_config['image_size_x']))
        self.camera_bp.set_attribute('image_size_y', str(sensor_config['image_size_y']))
        self.camera_bp.set_attribute('fov', str(sensor_config['fov']))

        """ This portion can be moved to env.reset
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        #available spawn points:
        spawn_points = self.world.get_map().get_spawn_points()
        self.spawn_point = random.choice(spawn_points)
        # adding vehicle to self
        self.vehicle = self.world.spawn_actor(self.vehicle_bp, self.spawn_point)
        # spawned vehicle in simulator

        self.camera = self.world.spawn_actor(self.camera_bp, camera_transform, attach_to=self.vehicle)

        self.distance = 0
        self.prev_xy = np.zeros((2, ))
        """
        # Action space is now defined in terms of throttle and steer instead of curvature and speed.
        throttle_range = np.linspace(0, 0.5, 5)
        steer_range = np.linspace(-.25, .25, 10)
        self.action_space = np.array(np.meshgrid(throttle_range, steer_range)).T.reshape(-1, 2)
        self.spawn_point = None
        if spawn_index is not None:
            self.spawn_point = self.world.get_map().get_spawn_points()[spawn_index]
        
        #self.camera.listen(lambda data: self.process_image(data))
        
        # Initialize HUD
        self.hud = HUD(sensor_config['image_size_x'], sensor_config['image_size_y'])
    def on_collision(self, event):
        self.collision_detected = True

    # def define_path(self):
    #     # This is a simplified example. You might want to define a more complex path.
    #     start_waypoint = self.world.get_map().get_waypoint(self.spawn_point.location)
    #     end_waypoint = start_waypoint.next(100.0)[0]  # Assuming 100 meters ahead for this example
    #     self.path = [start_waypoint, end_waypoint]
    #     # In a real scenario, you'd populate 'self.path' with a series of waypoints forming the desired route.
    # def is_vehicle_on_path(self):
    #     vehicle_location = self.vehicle.get_location()
    #     # Find the closest waypoint to the vehicle on the predefined path
    #     closest_distance = min([vehicle_location.distance(waypoint.transform.location) for waypoint in self.path])
    
    #     # Define a threshold for being off-path. This will need to be adjusted based on your scenario.
    #     off_path_threshold = 5.0  # meters
        
    #     return closest_distance <= off_path_threshold

    


    def reset(self):   # reset is to reset world?
        # Spawn or respawn the vehicle at a random location
        #delete what we created, eg. vehicles and sensors
        actor_list = self.world.get_actors()

        # Identify IDs of vehicles and sensors to be deleted
        vehicle_and_sensor_ids = [
            actor.id for actor in actor_list
            if 'vehicle' in actor.type_id or 'sensor' in actor.type_id
        ]

        # Iterate over the list of IDs to delete each actor
        for actor_id in vehicle_and_sensor_ids:
            # Attempt to get the actor by ID
            actor_to_delete = self.world.get_actor(actor_id)
            if actor_to_delete is not None:
                # If the actor exists, delete it
                actor_to_delete.destroy()
                print("Deleted:", actor_to_delete.type_id, "with ID:", actor_to_delete.id)
            else:
                # If the actor doesn't exist, print a message (optional)
                print("Actor with ID", actor_id, "not found.")

        if self.spawn_point is None:
            spawn_points = self.world.get_map().get_spawn_points()
            self.spawn_point = random.choice(spawn_points)
            print(f'spawn index: {spawn_points.index(self.spawn_point)}')
        self.vehicle = self.world.spawn_actor(self.vehicle_bp, self.spawn_point)

        # Attach the camera sensor
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4)) #camera offset relative to cars origin, 
        # changeable, depending where the camera actually is
        self.camera = self.world.spawn_actor(self.camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera.listen(lambda data: self.process_image(data
                                                           ))
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_detected = False
        self.collision_sensor.listen(lambda event: self.on_collision(event))

        self.distance = 0
        self.prev_xy = np.array([self.vehicle.get_location().x, self.vehicle.get_location().y])
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        # waypoint = map.get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)

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
      #  print(self.action_space)
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))

        self.world.tick()

        # Compute the distance traveled since the last step
        current_location = self.vehicle.get_location()
        current_xy = np.array([current_location.x, current_location.y])
        dd = np.linalg.norm(current_xy - self.prev_xy)
        self.distance += dd


        # Calculate reward based on the chosen reward function
        if self.rf == 1:
            reward, done = self.reward_1()
        elif self.rf == 2:
            reward, done = self.reward_2()
        else:
            reward, done = self.reward_3()

        info = {}  
        self.prev_xy = current_xy
        return self.image, reward, done, info

    def reward_1(self):
        """
        Discrete reward function for CARLA:
        - Penalize the agent heavily for getting out of lane.
        - Penalize for exceeding max rotation.
        - Penalize for not being centered on the road.
        - Penalize heavily if the vehicle is going in the opposite direction of the road.
        """
        
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_rotation = vehicle_transform.rotation.yaw

        # Convert yaw to radians and normalize between -pi and pi
        vehicle_rotation_radians = math.radians(vehicle_rotation)
        vehicle_rotation_radians = (vehicle_rotation_radians + np.pi) % (2 * np.pi) - np.pi

        # Maximal rotation (yaw angle) allowed
        maximal_rotation = np.pi / 10
        exceed_max_rotation = np.abs(vehicle_rotation_radians) > maximal_rotation

        # Getting the vehicle's lane information
        map = self.world.get_map()
        waypoint = map.get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)

        # Calculate the heading difference between the vehicle and the road
        road_direction = waypoint.transform.rotation.yaw
        road_direction_radians = math.radians(road_direction)
        heading_difference = abs(vehicle_rotation_radians - road_direction_radians) % (2 * np.pi)
        if heading_difference > np.pi:
            heading_difference = 2 * np.pi - heading_difference
        
        # Heavily penalize if the vehicle is going in the opposite direction (more than 90 degrees away from road direction)
        going_opposite_direction = heading_difference > np.pi / 2

        road_half_width = waypoint.lane_width / 2.
        center_of_lane = waypoint.transform.location
        distance_from_center = vehicle_location.distance(center_of_lane)

        out_of_lane = self.is_vehicle_within_lane() is False
        not_near_center = distance_from_center > road_half_width / 2
        print(not_near_center, math.degrees(heading_difference))
        # Determine if the episode should end
        done = not_near_center or going_opposite_direction or self.collision_detected

        # Compute reward based on conditions
        current_xy = np.array([vehicle_location.x, vehicle_location.y])
        reward = 0
        if self.collision_detected:
            done = True
            reward = -1000
        elif done:
            reward = -100 if not_near_center else -500  # More severe penalty for going in the opposite direction
        elif exceed_max_rotation:
            reward = -50
        else:
            # Calculate distance moved towards the driving direction since last tick
            dd = np.linalg.norm(current_xy - self.prev_xy)
            reward = dd * 50  # Assuming the simulation has a tick rate where this scaling makes sense
        
        reward += (np.pi/2 - heading_difference) * -100

        self.prev_xy = current_xy
        self.collision_detected = False
        return reward, done


    def reward_2(self):

        """
        Reward function that does not account for max rotation exceeded
        """

        exceed_max_rotation = np.abs(vehicle_rotation_radians) > maximal_rotation

        # Getting the vehicle's lane information
        map = self.world.get_map()
        waypoint = map.get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)
        road_half_width = waypoint.lane_width / 2.

        # Calculate the distance from the center of the lane+
        center_of_lane = waypoint.transform.location
        distance_from_center = vehicle_location.distance(center_of_lane)

        # Determine if the vehicle is out of lane or not near the center
        out_of_lane = self.is_vehicle_within_lane() is False
        not_near_center = distance_from_center > road_half_width / 4
        print(distance_from_center)

        # Determine if the episode should end
        done = out_of_lane

        # Compute reward based on conditions
        current_xy = np.array([vehicle_location.x, vehicle_location.y])
        dd = np.linalg.norm(current_xy - self.prev_xy)
        reward = 0

        if out_of_lane:
            reward = -100
        else:
            reward = dd * 5

        if not_near_center:
            reward -= 0.5
        else:
            reward += 2

        return reward, done

    def reward_3(self):
        reward=0
        done=False
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_rotation = vehicle_transform.rotation.yaw
        map = self.world.get_map()
        waypoint = map.get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)



        # Convert yaw to radians and normalize between -pi and pi
        vehicle_rotation_radians = math.radians(vehicle_rotation)
        vehicle_rotation_radians = (vehicle_rotation_radians + np.pi) % (2 * np.pi) - np.pi


        road_direction = waypoint.transform.rotation.yaw
        road_direction_radians = math.radians(road_direction)
        theta = abs(vehicle_rotation_radians - road_direction_radians) % (2 * np.pi)
        if theta > np.pi:
            theta = 2 * np.pi - theta
        going_opposite_direction = theta > np.pi / 2

        road_half_width = waypoint.lane_width / 2.

        # Calculate the distance from the center of the lane+
        center_of_lane = waypoint.transform.location
        distance_from_center = vehicle_location.distance(center_of_lane)

        not_near_center = distance_from_center > road_half_width / 1.5
        done = not_near_center or going_opposite_direction or self.collision_detected

        current_xy = np.array([vehicle_location.x, vehicle_location.y])
        dd = np.linalg.norm(current_xy - self.prev_xy)

        # left_waypoint = waypoint.get_left_lane()
        # right_waypoint = waypoint.get_right_lane()
        # Py = 0
        # if left_waypoint is not None and right_waypoint is not None:
        #     # Calculate the midpoint between the two waypoints
        #     midpoint = carla.Location(
        #         x=(left_waypoint.transform.location.x + right_waypoint.transform.location.x) / 2,
        #         y=(left_waypoint.transform.location.y + right_waypoint.transform.location.y) / 2,
        #         z=(left_waypoint.transform.location.z + right_waypoint.transform.location.z) / 2
        #     )

        #     # Calculate the distance from the vehicle to the midpoint
        #     Py = vehicle_location.distance(midpoint)
       # print(f'theta: {math.degrees(theta)}')
        Py = distance_from_center
        Wd = waypoint.lane_width/2.5
      #  print(f'Py, Wd: {Py}, {Wd}')
        i_fail = 1 if done else 0
      #  print(f'ifail: {i_fail}')
     #   print(Py)
        reward = dd + 2*math.cos(theta) - abs(Py / Wd) - (4 * i_fail)
        print(reward)
        return reward, done
    def reward_4(self):
        reward=0
        done=False
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_rotation = vehicle_transform.rotation.yaw
        map = self.world.get_map()
        waypoint = map.get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)



        # Convert yaw to radians and normalize between -pi and pi
        vehicle_rotation_radians = math.radians(vehicle_rotation)
        vehicle_rotation_radians = (vehicle_rotation_radians + np.pi) % (2 * np.pi) - np.pi


        road_direction = waypoint.transform.rotation.yaw
        road_direction_radians = math.radians(road_direction)
        theta = abs(vehicle_rotation_radians - road_direction_radians) % (2 * np.pi)
        if theta > np.pi:
            theta = 2 * np.pi - theta
        going_opposite_direction = theta > np.pi / 2

        road_half_width = waypoint.lane_width / 2.

        # Calculate the distance from the center of the lane+
        center_of_lane = waypoint.transform.location
        distance_from_center = vehicle_location.distance(center_of_lane)

        not_near_center = distance_from_center > road_half_width / 3
        done = not_near_center or going_opposite_direction or self.collision_detected

        current_xy = np.array([vehicle_location.x, vehicle_location.y])
        dd = np.linalg.norm(current_xy - self.prev_xy)

        # left_waypoint = waypoint.get_left_lane()
        # right_waypoint = waypoint.get_right_lane()
        # Py = 0
        # if left_waypoint is not None and right_waypoint is not None:
        #     # Calculate the midpoint between the two waypoints
        #     midpoint = carla.Location(
        #         x=(left_waypoint.transform.location.x + right_waypoint.transform.location.x) / 2,
        #         y=(left_waypoint.transform.location.y + right_waypoint.transform.location.y) / 2,
        #         z=(left_waypoint.transform.location.z + right_waypoint.transform.location.z) / 2
        #     )

        #     # Calculate the distance from the vehicle to the midpoint
        #     Py = vehicle_location.distance(midpoint)
       # print(f'theta: {math.degrees(theta)}')
        Py = distance_from_center
        Wd = waypoint.lane_width/2.5
      #  print(f'Py, Wd: {Py}, {Wd}')
        i_fail = 1 if done else 0
      #  print(f'ifail: {i_fail}')
     #   print(Py)
        reward = math.sqrt(dd) *(math.cos(theta) - abs(Py / Wd) - (2 * i_fail))
        print(reward)
        return reward, done   
    def get_vehicle_direction(self):
        transform = self.vehicle.get_transform()
        rotation = transform.rotation
        radians = math.radians(rotation.yaw)
        return carla.Vector3D(math.cos(radians), math.sin(radians), 0.0)
    
    def get_road_direction(self):
        # This is a simplified example. You'll need to adapt it based on how your road data is structured
        map = self.world.get_map()
        waypoint = map.get_waypoint(self.vehicle.get_location())
        next_waypoint = waypoint.next(1.0)[0]  # Assuming there's a next waypoint
        direction = next_waypoint.transform.location - waypoint.transform.location
        return direction.make_unit_vector()
    
    def calculate_angle_between_vectors(self, v1, v2):
        dot_product = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z
        magnitude_v1 = math.sqrt(v1.x**2 + v1.y**2 + v1.z**2)  # Magnitude of v1
        magnitude_v2 = math.sqrt(v2.x**2 + v2.y**2 + v2.z**2)  # Magnitude of v2
        
        # Ensure the magnitude is not zero to avoid division by zero error
        if magnitude_v1 == 0 or magnitude_v2 == 0:
            raise ValueError("One or both vectors have zero magnitude, can't calculate angle.")
        
        # Normalize the dot product by the magnitudes of v1 and v2
        cosine_of_angle = dot_product / (magnitude_v1 * magnitude_v2)
        
        # Clamp the cosine_of_angle to the domain of acos to avoid domain errors
        cosine_of_angle = max(min(cosine_of_angle, 1), -1)
        
        angle = math.acos(cosine_of_angle)  # Angle in radians
        
        return math.degrees(angle)  # Convert the angle to degrees
    
    def get_lateral_position_error_and_lane_width(self):
        # Get the vehicle's location
        vehicle_location = self.vehicle.get_location()
        map = self.world.get_map()
        
        # Get the closest waypoint to the vehicle's location
        closest_waypoint = map.get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)
        
        # Calculate the lateral position error
        # This is a simple approximation. For more accuracy, consider the direction of the road
        lateral_position_error = vehicle_location.distance(closest_waypoint.transform.location) - closest_waypoint.lane_width / 2
        
        # Get the lane width
        lane_width = closest_waypoint.lane_width
        
        return lateral_position_error, lane_width
    
    def is_vehicle_within_lane(self):
        # Get the vehicle's location
        map = self.world.get_map()
        vehicle_location = self.vehicle.get_location()
        
        # Get the closest waypoint to the vehicle's location, considering only driving lanes
        closest_waypoint = map.get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)
        
        # Get the transform of the closest waypoint
        waypoint_transform = closest_waypoint.transform
        
        # Calculate the vector from the waypoint to the vehicle
        vehicle_vector = vehicle_location - waypoint_transform.location
        vehicle_vector = carla.Vector3D(vehicle_vector.x, vehicle_vector.y, 0)  # Ignore Z component
        
        # Calculate the forward vector of the waypoint (direction the lane is facing)
        waypoint_forward_vector = waypoint_transform.get_forward_vector()
        waypoint_forward_vector = carla.Vector3D(waypoint_forward_vector.x, waypoint_forward_vector.y, 0)  # Ignore Z component
        
        # Calculate the right vector of the waypoint (perpendicular to the forward vector)
        waypoint_right_vector = carla.Vector3D(-waypoint_forward_vector.y, waypoint_forward_vector.x, 0)
        
        # Project the vehicle vector onto the waypoint right vector to get the lateral distance from the lane center
        lateral_distance = vehicle_vector.dot(waypoint_right_vector)
        
        # Check if the absolute value of lateral_distance is less than or equal to half the lane width
        is_within_lane = abs(lateral_distance) <= (closest_waypoint.lane_width / 2)
        
        return is_within_lane

    def epsilon_greedy_action(self, state, epsilon):
    
        # print(f"\tstate.shape = {state.shape}")
        state = state.permute(0, 3, 1, 2)
        prob = np.random.uniform()
        if prob < epsilon:
            self.action_idx = np.random.randint(len(self.action_space))
            return self.action_space[self.action_idx]
        else:
            qs = network.forward(state).cpu().data.numpy()
            self.action_idx = np.argmax(qs)
            return self.action_space[self.action_idx]


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

    print ("Running Main")
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
    #reward function
    parser.add_argument('--reward-function',
                        type=str,
                        nargs='+',
                        help='1 or 2 or 3',
                        required=True)
    parser.add_argument('--map',
                        type=str,
                        nargs='+',
                        help='Specify CARLA map: (Town01, ...  Town07)',
                        required=False)
    args = parser.parse_args()      


    if not args.operation:
        print ("Operation argument is required")
        raise ValueError("Operation argument is required. Use '--operation Load' or '--operation New'.")


    # configuring environment
    car_config=0 #likely not needed

    sensor_config = { #default sensor configuration
        
    'image_size_x': 640,  # Width of the image in pixels
    'image_size_y': 480,  # Height of the image in pixels
    'fov': 90,            # Field of view in degrees
} 
    

    map = 0 # default map
    if args.map: #specifed map is chosen
        map = args.map[0]
    env = Environment( client, car_config, sensor_config, args.reward_function, map)
    
    
    # initialize HUD
    hud = HUD(sensor_config['image_size_x'], sensor_config['image_size_y'])
    if args.operation[0].lower() == 'new':

        """
        Initializing hyper-parameters and beginning the training loop
        """
        replay_buffer = ReplayBuffer(10000)
        batch_size = 32
        gamma = 0.99 
        epsilon_start = 1.0
        epsilon_end = 0.01
        epsilon_decay = 0.995
        num_episodes = 1000
        target_update = 10  # Update target network every 10 episodes
        max_num_steps = 800

        best_dict_reward = -1e10

        # per episode
        rewards = np.array([])
        num_steps = np.array([])

        epsilon = epsilon_start
        
        for episode in range(num_episodes):
            state = env.reset()
            # print(f"main, state.shape after reset = {state.shape}")
            # print(state)
       #     display.reset()
            total_reward = 0
            done = False
            step = 0

            while step < max_num_steps and not done:
                # Convert state to the appropriate format and move to device
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)

                # Select action using epsilon greedy policy
                action = env.epsilon_greedy_action(state_tensor, epsilon)
                next_state, reward, done, _ = env.step(action)
                # next_state = next_state

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
                
                # hud.update(env.vehicle.get_velocity().x, throttle, steer)
                
                # Get camera image
                camera_image = env.image  # Assumss
                
                # Display HUD and camera view
                camera_image_with_hud = hud.tick(camera_image)
                cv2.imshow("Camera View with HUD", camera_image_with_hud)
                cv2.waitKey(1)
                
            
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
     #   display.render()
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