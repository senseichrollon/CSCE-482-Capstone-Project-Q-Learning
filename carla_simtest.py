import carla
import pygame
import random
import numpy as np


def process_img(image):
    """Process the image from the camera sensor and convert it to a format suitable for Pygame."""
    i = np.array(image.raw_data)  # Convert the raw data to an array
    i2 = i.reshape(
        (image.height, image.width, 4)
    )  # Reshape it to have the proper dimension
    i3 = i2[:, :, :3]  # Drop the alpha channel
    return i3.swapaxes(0, 1)  # Pygame uses width x height, so we need to swap the axes


def image_callback(image, surface):
    """Update Pygame surface with camera image."""
    image.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array.swapaxes(0, 1)
    surface = pygame.surfarray.make_surface(array)
    screen.blit(surface, (0, 0))


def main():
    pygame.init()
    display_width = 640
    display_height = 480
    global screen
    screen = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption("CARLA Keyboard Control")
    clock = pygame.time.Clock()

    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)

        world = client.get_world()

        blueprint_library = world.get_blueprint_library()
        car_model = blueprint_library.filter("model3")[0]

        spawn_point = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(car_model, spawn_point)

        # Attach a camera sensor to the vehicle
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", f"{display_width}")
        camera_bp.set_attribute("image_size_y", f"{display_height}")
        camera_bp.set_attribute("fov", "110")
        camera_transform = carla.Transform(carla.Location(x=5, z=1))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        # Create a Pygame surface for camera images
        camera_surface = pygame.surface.Surface((display_width, display_height))

        # Listen to camera images
        camera.listen(lambda image: image_callback(image, camera_surface))

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()
            control = carla.VehicleControl()

            if keys[pygame.K_UP] or keys[pygame.K_w]:
                control.throttle = 1.0
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                control.brake = 1.0
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                control.steer = -0.5
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                control.steer = 0.5
            if keys[pygame.K_SPACE]:
                control.hand_brake = True

            vehicle.apply_control(control)

            pygame.display.flip()
            clock.tick_busy_loop(60)

    finally:
        vehicle.destroy()
        camera.destroy()
        pygame.quit()
        print("Simulation ended.")


if __name__ == "__main__":
    main()
