import carla

# Connect to the CARLA simulator
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)  # Set a timeout in seconds for client connection

# Get the world object
world = client.get_world()

# Access world attributes or methods
print("Number of vehicles:", len(world.get_actors().filter('vehicle')))
print("Number of pedestrians:", len(world.get_actors().filter('walker')))

# Spawn a vehicle
spawn_point = carla.Transform(carla.Location(x=10, y=200, z=0), carla.Rotation(yaw=0))
vehicle_blueprint = world.get_blueprint_library().find('vehicle.tesla.model3')
vehicle = world.spawn_actor(vehicle_blueprint, spawn_point)