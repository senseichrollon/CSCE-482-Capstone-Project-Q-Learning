import argparse

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
    print("Operation is", args.operation[0])