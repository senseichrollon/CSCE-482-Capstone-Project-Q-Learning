import carla
import keyboard #py -3.8 -m pip install keyboard

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

# Function to update vehicle control based on keyboard input
def update_vehicle_control():
    control = carla.VehicleControl()
    if keyboard.is_pressed('w'):
        control.throttle = 1.0
    if keyboard.is_pressed('a'):
        control.steer = 1.0
    if keyboard.is_pressed('s'):
        control.brake = 1.0
    if keyboard.is_pressed('d'):
        control.steer = 1.0
    vehicle.apply_control(control)

while True:
    update_vehicle_control()
        