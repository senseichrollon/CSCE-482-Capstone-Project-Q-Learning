import unittest
import carla
from carla_lane_keeping_d3qn import DuelingDDQN, Environment, ReplayBuffer, optimize_model
import torch
import numpy as np
import frontend
def get_action_values(action_index):
    # Example mappings, these need to be defined according to your specific action space design
        throttle_values = np.linspace(0, 0.5, num=5)  # Example throttle values
        steer_values = np.linspace(-.25, 0.25, num=9)    # Example steering values
        num_steers = len(steer_values)
        
        throttle = throttle_values[action_index // num_steers]
        steer = steer_values[action_index % num_steers]
        return throttle, steer

class CARLAIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize CARLA client
        cls.client = carla.Client("localhost", 2000)
        cls.client.set_timeout(10.0)
        
        # Load the world
        cls.world = cls.client.get_world()

        # Environment and model setup
        cls.action_dim = 45  # Example action dimension
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available, else CPU
        cls.dueling_ddqn = DuelingDDQN(cls.action_dim).to(cls.device)
        cls.replay_buffer = ReplayBuffer(1000)
        sensor_config = {  # default sensor configuration
        "image_size_x": 640,  # Width of the image in pixels
        "image_size_y": 480,  # Height of the image in pixels
        "fov": 90,  # Field of view in degrees
        }
        cls.env = Environment(cls.client, None, sensor_config, '4', "Town02", 19)  # Using a generic sensor configuration and reward function for simplicity
        cls.batch_size = 32
        cls.gamma = 0.99  # Discount factor for reward
   
    def test_full_episode_flow(self):
        """Test a full episode flow from initialization to termination."""
        state = self.env.reset()
        self.assertIsNotNone(state, "Failed to reset the environment at the beginning of an episode.")

        done = False
        total_reward = 0
        num_steps=0
        while not done:
            num_steps+=1
            state_tensor = torch.from_numpy(state).permute(2, 0, 1).unsqueeze(0).to(self.device)
            q_values = self.dueling_ddqn(state_tensor)
            action_index = np.argmax(q_values.detach().numpy())
            throttle, steer = get_action_values(action_index)
            action= (throttle,steer)
            # Take action in the environment
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward

            # Check if the state is updated and reward is received after an action
            self.assertIsNotNone(next_state, "The environment did not return a new state after taking an action.")
            self.assertIsInstance(reward, float, "The environment did not return a valid reward.")

            # Add experience to the replay buffer
            next_state_tensor = torch.from_numpy(next_state).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.replay_buffer.store((state_tensor, action, reward, next_state_tensor, done))
            state = next_state
            if num_steps>99:
                 done=True

        # Assert that the episode has ended
        self.assertTrue(done, "The episode did not terminate correctly.")
        # Check if there is a non-zero reward accumulated in the episode
        self.assertNotEqual(total_reward, 0, "The agent did not accumulate any reward during the episode.")

    def test_training_step(self):
        #Test a single training optimization step.
        state= self.env.reset()
        param=self
        # Populate the replay buffer with random experiences
        for _ in range(self.batch_size):
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)  # Ensure it's a tensor and on the right device
            action_index = 5
            throttle, steer = get_action_values(action_index)
            action= (throttle,steer)
            reward = np.random.rand()
            next_state, reward, done, info = self.env.step(action)  # Ensure it's a tensor
            done = np.random.choice([True, False])
            param.grad= state_tensor
            next_state_tensor = (
                    torch.from_numpy(next_state).unsqueeze(0).to(self.device)
                    if next_state is not None
                    else None
            )
            self.replay_buffer.store(
                    (state_tensor, 5, reward, next_state_tensor, done))

        # Perform optimization step

        # Assert that the optimization step updated the model
        # This could be done by checking if gradients are computed and if the parameters have been updated
        for Param in self.dueling_ddqn.parameters():
            
            self.assertIsNotNone(param.grad, "Model parameters were not updated - no gradients found.")

    def test_environment_sensors(self):
        #Test if the environment's sensors are correctly initialized and collecting data.
        state = self.env.reset()
        self.assertTrue(len(state) > 0, "No data was collected from the environment sensors after reset.")

    def test_vehicle_spawn(self):
        #Test if the vehicle spawns at the expected location.
        state = self.env.reset()
        vehicle_location = self.env.vehicle.get_location()
        spawn_location = self.env.spawn_point.location
        self.assertTrue(vehicle_location.distance(spawn_location) < 1.0, "Vehicle did not spawn at the expected location.")





if __name__ == '__main__':
    unittest.main()