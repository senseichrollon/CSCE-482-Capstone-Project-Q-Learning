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

# Carla Client attribute
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)  # Set a timeout in seconds for client connection
print("Client established") 

class Environment:
    def __init__(self, carla_client, sensor_config):
        
        #Connecting to Carla Client
        self.client = carla_client
        self.world = self.client.get_world()
        
        # spawn world
        self.world = self.client.load_world("Town01")
        
        #Retrieve Vehicle
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_bps = [bp for bp in self.blueprint_library.filter('vehicle') if bp.has_attribute('number_of_wheels')]
        self.vehicle_bp = self.vehicle_bps[0]
        
        # spawn car
        self.spawn_point = None
        if self.spawn_point is None:
            spawn_points = self.world.get_map().get_spawn_points()
            self.spawn_point = random.choice(spawn_points)
        self.vehicle = self.world.spawn_actor(self.vehicle_bp, self.spawn_point)
        
        # Attach the camera sensor
        self.sensor_config = sensor_config
        self.camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        self.camera_bp.set_attribute('image_size_x', str(sensor_config['image_size_x']))
        self.camera_bp.set_attribute('image_size_y', str(sensor_config['image_size_y']))
        self.camera_bp.set_attribute('fov', str(sensor_config['fov']))
        
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4)) #camera offset relative to cars origin, 
        # changeable, depending where the camera actually is
        self.camera = self.world.spawn_actor(self.camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera.listen(lambda data: self.process_image(data
                                                           ))
        collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_detected = False
        self.collision_sensor.listen(lambda event: self.on_collision(event))  
        
    def data_collection(self):
        # Start collecting data
        self.image = None
        while self.image is None:
            self.world.tick()
        return self.image
        
    def process_image(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.sensor_config['image_size_y'], self.sensor_config['image_size_x'], 4))
        self.image = i2[:, :, :3]
        cv2.imshow(self.image)
        cv2.waitKey(1)
        self.world.tick()
    
        
        # Control the vehicle
        # self.vehicle.set_autopilot(True)


if __name__ == '__main__':

    sensor_config = { #default sensor configuration
        
    'image_size_x': 640,  # Width of the image in pixels
    'image_size_y': 480,  # Height of the image in pixels
    'fov': 90,            # Field of view in degrees
} 
    
    env = Environment(client, sensor_config)
    num_episodes = 500
    while True:
        i = 0