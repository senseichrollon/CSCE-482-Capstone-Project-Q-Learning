import argparse
from collections import deque, namedtuple
import random
import numpy as np
import carla
import time
import keyboard 
import cv2

# Connect to the CARLA simulator
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)  # Set a timeout in seconds for client connection

# Get the world object
world = client.get_world()
actors = world.get_actors()
print(world.get_actors())
vehicle_ids = [actor.id for actor in actors if 'vehicle' in  actor.type_id]
#vehicle = world.get_actor('vehicle')
print("Number of vehicles beforehand :", len(world.get_actors().filter('vehicle')))
print("Number of pedestrians beforehand :", len(world.get_actors().filter('walker')))


#for vehicleid in vehicle_ids:
 #   vehicle = world.get_actor(vehicleid)
  #  vehicle.destroy()

#for cameraa in world.get_actors().filter('sensor.camera.rgb'):
 #   cameraa.destroy()


vehicle_and_sensor_ids = [actor.id for actor in actors if (('vehicle' in  actor.type_id) or ('sensor' in actor.type_id))]
        #sensor_ids= [actor.id for actor in actor_list if 'sensor' in  actor.type_id]
for id in vehicle_and_sensor_ids:
    created_actor = world.get_actor(id)
    created_actor.destroy()
# Access world attributes or methods
print("Number of vehicles after destruction:", len(world.get_actors().filter('vehicle')))
print("Number of pedestrians after:", len(world.get_actors().filter('walker')))

# Spawn a vehicle

    

spawn_points = world.get_map().get_spawn_points()
spawn_point = random.choice(spawn_points)

#spawn_point = carla.Transform(carla.Location(x=10, y=200, z=0), carla.Rotation(yaw=0))
vehicle_blueprint = world.get_blueprint_library().find('vehicle.tesla.model3')
vehicle = world.spawn_actor(vehicle_blueprint, spawn_point)
sensor_config = { #default sensor configuration
        
    'image_size_x': 800,  # Width of the image in pixels
    'image_size_y': 600,  # Height of the image in pixels
    'fov': 90,            # Field of view in degrees
} 
blueprint_library = world.get_blueprint_library()
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', str(sensor_config['image_size_x']))
camera_bp.set_attribute('image_size_y', str(sensor_config['image_size_y']))
camera_bp.set_attribute('fov', str(sensor_config['fov']))

camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))  # Adjust position relative to vehicle
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
texture_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
texture_bp.set_attribute('image_size_x', '800')
texture_bp.set_attribute('image_size_y', '600')
texture = world.spawn_actor(texture_bp, carla.Transform(), attach_to=camera)

def process_image(image):
    # Convert the image to an array
    image_array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    image_array = np.reshape(image_array, (image.height, image.width, 4))
    image_array = image_array[:, :, :3]  # Remove the alpha channel
    
    # Display the image using OpenCV
    cv2.imshow("Camera View", image_array)
    cv2.waitKey(1)  # Update the display
    #texture.write(image_array.tobytes())

camera.listen(process_image)
print("Number of vehicles after creation:", len(world.get_actors().filter('vehicle')))

print("Number of actors after creation:", len(world.get_actors()))
print(world.get_actors())

# Function to update vehicle control based on keyboard input
def update_vehicle_control():
    control = carla.VehicleControl()
    if keyboard.is_pressed('w'):
        control.throttle = 1.0
    if keyboard.is_pressed('a'): #goes right
        control.steer = 1.0
    if keyboard.is_pressed('d'):
        control.steer = 1.0
    if keyboard.is_pressed('s'):
        control.throttle= -1.0
    vehicle.apply_control(control)



while True:
    
    update_vehicle_control()
    location = vehicle.get_location()
    velocity = vehicle.get_velocity()
    
    print(location)
    world.tick()
