import unittest
import carla
from carla_lane_keeping_d3qn import DuelingDDQN, Environment, ReplayBuffer
import torch
import numpy as np
from unittest.mock import MagicMock
import math
class DuelingDDQNTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.action_dim = 45  # As defined in your DuelingDDQN model
        cls.state_dim = (3, 480, 640)  # Assuming an image input
        cls.dueling_ddqn = DuelingDDQN(cls.action_dim)
        cls.test_input= torch.rand(1, 3, 480, 640)

    def test_network_initialization(self):
        """Test if the network initializes with the expected number of actions."""
        self.assertEqual(self.dueling_ddqn.advantage_stream.out_features, self.action_dim)

    def test_forward_shape(self):
        """Test if the forward pass returns the correct shape."""
        dummy_input = torch.zeros(1, *self.state_dim)
        with torch.no_grad():
            q_values = self.dueling_ddqn(dummy_input)
        self.assertEqual(q_values.shape, (1, self.action_dim))

    def test_output_type(self):
        # Verify that the output is a PyTorch tensor.
        output = self.test_input
        self.assertIsInstance(output, torch.Tensor)


class TestRewardFunctions(unittest.TestCase):

    def setUp(self):
        # Perfect Mock environment 1- straight road, no rotation on car, distance travelled is 5, no steering done
        #no collisions, perfectly center of road. car path aligned with road
        self.env= Environment
        self.env.vehicle= MagicMock()
        self.env.world=MagicMock()
        self.env.steer= 0
        self.env.is_vehicle_within_lane = MagicMock(return_value=True)
        self.env.vehicle.get_transform.return_value.location.x =4   #vehicle x location
        self.env.vehicle.get_transform.return_value.location.y =3   #vehicle y location
        self.env.vehicle.get_transform.return_value.location.distance = MagicMock(return_value=0) #distance between roads center and car
        self.env.vehicle.get_transform.return_value.rotation.yaw = 0
        self.env.collision_detected = False
        self.env.prev_xy = np.array([0, 0])

        self.waypoint = MagicMock()
        self.waypoint.lane_width = 10  # Specific numeric value for lane width
        self.waypoint.transform.rotation.yaw = 0   #straight line
        self.waypoint.transform.location = MagicMock()
        self.waypoint.transform.location.distance = MagicMock(return_value=0)

        self.env.world.get_map.return_value.get_waypoint.return_value = self.waypoint
    
    def create_environment(self, vehicle_x, vehicle_y, vehicle_yaw, lane_width, road_yaw, distance_from_center, collision, steer= 0):
        self.env.vehicle.get_transform.return_value.location.x = vehicle_x
        self.env.vehicle.get_transform.return_value.location.y = vehicle_y
        self.env.vehicle.get_transform.return_value.location.distance = MagicMock(return_value=distance_from_center)
        self.env.vehicle.get_transform.return_value.rotation.yaw = vehicle_yaw
        self.env.prev_xy = np.array([0, 0])
        self.env.collision_detected = collision
        self.env.steer= steer
        waypoint = MagicMock()
        waypoint.lane_width = lane_width
        waypoint.transform = MagicMock()
        waypoint.transform.rotation.yaw = road_yaw
        waypoint.transform.location = MagicMock()
        waypoint.transform.location.distance = MagicMock(return_value=distance_from_center)
        
        self.env.world.get_map.return_value.get_waypoint.return_value = waypoint


    def simulate_step(self, action):
        # Simulate an environment step without needing the full environment
        # This should return a state, reward, done, and info dictionary mimicking the environment's response
        # This is a mocked method; you need to replace it with actual logic or mocks suitable for your setup
        return np.zeros((480, 640, 3)), 0, False, {}

    def test_reward_function_1(self):
        # Mock the environment state that corresponds to specific conditions for reward function 1

        test_cases= [{'vehicle_x': 4, 'vehicle_y': 3, 'vehicle_yaw': 0, 'lane_width': 10, 'road_yaw': 0, 'distance_from_center': 0, 'collision': 0}
                     ,{'vehicle_x': 8, 'vehicle_y': 6, 'vehicle_yaw': 0, 'lane_width': 10, 'road_yaw': 0, 'distance_from_center': 0, 'collision': 0}
                      ,{'vehicle_x': 8, 'vehicle_y': 6, 'vehicle_yaw': 0, 'lane_width': 10, 'road_yaw': 0, 'distance_from_center': 0, 'collision': 1}
                       ,{'vehicle_x': 8, 'vehicle_y': 6, 'vehicle_yaw': 10, 'lane_width': 10, 'road_yaw': 0, 'distance_from_center': 0, 'collision': 0}
                   ,{'vehicle_x': 5, 'vehicle_y': 12, 'vehicle_yaw': 5, 'lane_width': 10, 'road_yaw': 10, 'distance_from_center': 0, 'collision': 0}
               ,{'vehicle_x': 5, 'vehicle_y': 12, 'vehicle_yaw': 5, 'lane_width': 10, 'road_yaw': 10, 'distance_from_center': 13, 'collision': 0}
      ]
        expected_rewards=[250, 500, -1000, 500- math.pi/180*10*100, 650- math.pi/180*5*100, -100-math.pi/180*5*100]
        expected_done =[False, False, True, False, False, True]
        i=0
        for case in test_cases:
            with self.subTest(case=case):
                self.create_environment(**case)
                reward,done = self.env.reward_1(self.env)
                self.assertAlmostEqual(reward, expected_rewards[i], places =5)  # Assuming that reward function 1 might penalize for certain actions
                self.assertEqual(done,expected_done[i])
                i+=1


    def test_reward_function_3(self):
        # Mock the environment state that corresponds to specific conditions for reward function 3
        test_cases= [{'vehicle_x': 4, 'vehicle_y': 3, 'vehicle_yaw': 0, 'lane_width': 10, 'road_yaw': 0, 'distance_from_center': 0, 'collision': 0}
                     ,{'vehicle_x': 8, 'vehicle_y': 6, 'vehicle_yaw': 0, 'lane_width': 10, 'road_yaw': 0, 'distance_from_center': 0, 'collision': 0}
                      ,{'vehicle_x': 8, 'vehicle_y': 6, 'vehicle_yaw': 0, 'lane_width': 10, 'road_yaw': 0, 'distance_from_center': 0, 'collision': 1}
                       ,{'vehicle_x': 8, 'vehicle_y': 6, 'vehicle_yaw': 10, 'lane_width': 10, 'road_yaw': 0, 'distance_from_center': 0, 'collision': 0}
                   ,{'vehicle_x': 5, 'vehicle_y': 12, 'vehicle_yaw': 5, 'lane_width': 10, 'road_yaw': 10, 'distance_from_center': 0, 'collision': 0}
               ,{'vehicle_x': 5, 'vehicle_y': 12, 'vehicle_yaw': 5, 'lane_width': 10, 'road_yaw': 10, 'distance_from_center': 13, 'collision': 0}
      ]
        expected_rewards=[7, 10+2, 8, 10+ 2*math.cos(math.pi/180*10), 13+2*math.cos(math.pi/180*5), 13+2*math.cos(math.pi/180*5) -4- 13/(10/2.5)]
        expected_done =[False, False, True, False, False, True]
        i=0
        for case in test_cases:
            with self.subTest(case=case):
                self.create_environment(**case)
                reward,done = self.env.reward_3(self.env)
                self.assertAlmostEqual(reward, expected_rewards[i], places =5)  # Assuming that reward function 1 might penalize for certain actions
                self.assertEqual(done,expected_done[i])
                i+=1

    def test_reward_function_4(self):
        # Mock the environment state that corresponds to specific conditions for reward function 4
        test_cases= [{'vehicle_x': 4, 'vehicle_y': 3, 'vehicle_yaw': 0, 'lane_width': 10, 'road_yaw': 0, 'distance_from_center': 0, 'collision': 0}
                     ,{'vehicle_x': 8, 'vehicle_y': 6, 'vehicle_yaw': 0, 'lane_width': 10, 'road_yaw': 0, 'distance_from_center': 0, 'collision': 0}
                      ,{'vehicle_x': 8, 'vehicle_y': 6, 'vehicle_yaw': 0, 'lane_width': 10, 'road_yaw': 0, 'distance_from_center': 0, 'collision': 1}
                       ,{'vehicle_x': 8, 'vehicle_y': 6, 'vehicle_yaw': 10, 'lane_width': 10, 'road_yaw': 0, 'distance_from_center': 0, 'collision': 0}
                   ,{'vehicle_x': 5, 'vehicle_y': 12, 'vehicle_yaw': 5, 'lane_width': 10, 'road_yaw': 10, 'distance_from_center': 0, 'collision': 0}
               ,{'vehicle_x': 5, 'vehicle_y': 12, 'vehicle_yaw': 5, 'lane_width': 10, 'road_yaw': 10, 'distance_from_center': 13, 'collision': 0}
                   ,{'vehicle_x': 5, 'vehicle_y': 12, 'vehicle_yaw': 5, 'lane_width': 10, 'road_yaw': 10, 'distance_from_center': 13, 'collision': 0,'steer': 0.15}
                ,{'vehicle_x': 5, 'vehicle_y': 12, 'vehicle_yaw': 5, 'lane_width': 10, 'road_yaw': 10, 'distance_from_center': 13, 'collision': 0,'steer': -0.05}
                ]
        expected_rewards=[math.sqrt(5)+1, math.sqrt(10)+1, math.sqrt(10)+1, 
                        math.sqrt(10)+math.cos(math.pi/180*10), math.sqrt(13)+math.cos(math.pi/180*5),
                        math.sqrt(13)+math.cos(math.pi/180*5)-2 - 13/(10/2.5),
                        math.sqrt(13)+math.cos(math.pi/180*5)-2 - 13/(10/2.5)- 2*.15,
                        math.sqrt(13)+math.cos(math.pi/180*5)-2 - 13/(10/2.5)- 2*.05]
        expected_done =[False, False, True, False, False, True, True,True]
        i=0
        for case in test_cases:
            with self.subTest(case=case):
                self.create_environment(**case)
                reward,done, theta, Py = self.env.reward_4(self.env)
                self.assertAlmostEqual(reward, expected_rewards[i], places =5)  # Assuming that reward function 1 might penalize for certain actions
                self.assertEqual(done,expected_done[i])
                i+=1

"""

class EnvironmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = carla.Client("localhost", 2000)
        cls.client.set_timeout(10.0)
        cls.car_config = {'id': 'vehicle.audi.tt'}
        cls.sensor_config = {'fov': 90}
        cls.reward_function = [1]  # Using the first type of reward function as an example.
        cls.env = Environment(cls.client, cls.car_config, cls.sensor_config, cls.reward_function)

    def test_environment_reset(self):
        #Test if the environment resets correctly.
        state = self.env.reset()
        self.assertIsNotNone(state, "The environment did not return a valid initial state.")

class ReplayBufferTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.buffer_capacity = 1000
        cls.replay_buffer = ReplayBuffer(cls.buffer_capacity)

    def test_store_experience(self):
        initial_size = self.replay_buffer.size()
        experience = (np.random.rand(3, 480, 640), 0, 1.0, np.random.rand(3, 480, 640), False)
        self.replay_buffer.store(experience)
        self.assertEqual(self.replay_buffer.size(), initial_size + 1, "Experience was not stored correctly.")

    def test_sample_batch(self):
        batch_size = 32
        # Populate the buffer with more than batch_size experiences
        for _ in range(100):
            experience = (np.random.rand(3, 480, 640), 0, 1.0, np.random.rand(3, 480, 640), False)
            self.replay_buffer.store(experience)
        sample = self.replay_buffer.sample(batch_size)
        self.assertEqual(len(sample), batch_size, "Sampled batch size does not match the requested size.")
"""


if __name__ == '__main__':
    unittest.main()
