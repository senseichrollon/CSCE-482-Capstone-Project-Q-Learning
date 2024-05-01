import unittest
import carla
from carla_lane_keeping_d3qn import DuelingDDQN, Environment, ReplayBuffer, optimize_model
import torch
import numpy as np

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
        cls.dueling_ddqn = DuelingDDQN(cls.action_dim).to(torch.device('cpu'))
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

        while not done:
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
            q_values = self.dueling_ddqn(state_tensor)
            action = np.argmax(q_values.detach().numpy())

            # Take action in the environment
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward

            # Check if the state is updated and reward is received after an action
            self.assertIsNotNone(next_state, "The environment did not return a new state after taking an action.")
            self.assertIsInstance(reward, float, "The environment did not return a valid reward.")

            # Add experience to the replay buffer
            self.replay_buffer.store((state_tensor, action, reward, next_state, done))
            state = next_state

        # Assert that the episode has ended
        self.assertTrue(done, "The episode did not terminate correctly.")
        # Check if there is a non-zero reward accumulated in the episode
        self.assertNotEqual(total_reward, 0, "The agent did not accumulate any reward during the episode.")

    def test_training_step(self):
        """Test a single training optimization step."""
        self.env.reset()

        # Populate the replay buffer with random experiences
        for _ in range(self.batch_size):
            state = np.random.rand(3, 480, 640)
            action = np.random.randint(0, self.action_dim)
            reward = np.random.rand()
            next_state = np.random.rand(3, 480, 640)
            done = np.random.choice([True, False])
            self.replay_buffer.store((state, action, reward, next_state, done))

        # Perform optimization step
        optimize_model(self.replay_buffer, self.batch_size, self.gamma)

        # Assert that the optimization step updated the model
        # This could be done by checking if gradients are computed and if the parameters have been updated
        for param in self.dueling_ddqn.parameters():
            self.assertIsNotNone(param.grad, "Model parameters were not updated - no gradients found.")

    def test_environment_sensors(self):
        """Test if the environment's sensors are correctly initialized and collecting data."""
        state = self.env.reset()
        self.assertTrue(len(state) > 0, "No data was collected from the environment sensors after reset.")

    def test_vehicle_spawn(self):
        """Test if the vehicle spawns at the expected location."""
        state = self.env.reset()
        vehicle_location = self.env.vehicle.get_location()
        spawn_location = self.env.spawn_point.location
        self.assertTrue(vehicle_location.distance(spawn_location) < 1.0, "Vehicle did not spawn at the expected location.")

    # Additional detailed integration tests could include:
    # - Verifying that the reward function provides expected rewards for various scenarios.
    # - Confirming that the agent learns to improve over time by comparing performance metrics across episodes.

if __name__ == '__main__':
    unittest.main()