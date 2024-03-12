import argparse
from collections import deque, namedtuple
import random
import numpy as np
import carla


"""
Replay buffer class
"""


# Carla Client attribute
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)  # Set a timeout in seconds for client connection
print("Client established") 


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
    def __init__(self, carla_client, car_config, sensor_config, reward_function, map=0):
        
        #reset environment
        self.client = carla_client
        self.world = self.client.get_world()
        # if loading specifc map
        if (map != 0):
            self.world= self.client.load_world(map)
         #take out cars
        actor_list = self.world.get_actors()
        for actor in actor_list:
            if actor.type_id != 'sensor.camera.rgb':
                actor.destroy()

        self.car_config = car_config
        self.sensor_config = sensor_config
        self.rf = reward_function
        
        

        self.blueprint_library = self.world.get_blueprint_library()

        print(self.blueprint_library)
        self.vehicle_bps = [bp for bp in self.blueprint_library.filter('vehicle') if bp.has_attribute('number_of_wheels')]


        print(self.vehicle_bps[0])
        print(self.vehicle_bps[1])
        print(self.vehicle_bps[2])
        self.vehicle_bp = self.vehicle_bps[0]
        self.camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        self.camera_bp.set_attribute('image_size_x', str(sensor_config['image_size_x']))
        self.camera_bp.set_attribute('image_size_y', str(sensor_config['image_size_y']))
        self.camera_bp.set_attribute('fov', str(sensor_config['fov']))


        #available spawn points:
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        # adding vehicle to self
        self.vehicle = self.world.spawn_actor(self.vehicle_bp, spawn_point)
        self.distance = 0
        self.prev_xy = np.zeros((2, ))

        # Action space is now defined in terms of throttle and steer instead of curvature and speed.
        throttle_range = np.linspace(0, 1, 10)
        steer_range = np.linspace(-1, 1, 10)
        self.action_space = np.array(np.meshgrid(throttle_range, steer_range)).T.reshape(-1, 2)

    def reset(self):   # reset is to reset world?
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


        reward =0
        done=True
        return reward, done

    def reward_2(self):


        reward=0
        done=True
        return reward, done

    def reward_3(self):

        reward=0
        done=True
        return reward, done

    def epsilon_greedy_action(self, state, epsilon):
    
        action_idx=0


        return action_idx




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
                        help='1 or 2',
                        required=True)
    parser.add_argument('--map',
                        type=str,
                        nargs='+',
                        help='Specify CARLA map: (Town01, ...  Town07)',
                        required=False)
    args = parser.parse_args()


    if not args.operation:
        raise ValueError("Operation argument is required. Use '--operation Load' or '--operation New'.")


    # configuring environment
    car_config=0 #likely not needed

    sensor_config = { #default sensor configuration
        
    'image_size_x': 800,  # Width of the image in pixels
    'image_size_y': 600,  # Height of the image in pixels
    'fov': 90,            # Field of view in degrees
} 
    map = 0 # default map
    if args.map: #specifed map is chosen
        map = args.map[0]
    env = Environment( client, car_config, sensor_config, args.reward_function, map)
    while True:
        #run
        x=0
        print("xsxs")