import math
import random
import carla
import time
import numpy as np
import cv2

client = carla.Client('localhost', 2000)
client.set_timeout(5.0)
world = client.load_world('Town03_Opt', carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)
#world = client.get_world()
spectator = world.get_spectator()
bp_lib = world.get_blueprint_library()

bike_bp = bp_lib.find("vehicle.diamondback.century")
target_pylon_bp = bp_lib.find('static.prop.gnome')

#transform = random.choice(world.get_map().get_spawn_points())
bike_spawn_transform = carla.Transform(carla.Location(x=7, y=-117, z=1), carla.Rotation(pitch=0, yaw=-300, roll=0))

bike = world.try_spawn_actor(bike_bp, bike_spawn_transform)

bike_location = bike.get_transform().location


time.sleep(1)
target_location = bike.get_transform().location + (bike.get_transform().get_forward_vector() * 8) 
print("bike location: ", bike.get_transform().location)
print("forward vector: ", bike.get_transform().get_forward_vector())
print("target_location: ", target_location)
target_location = bike.get_transform().location + (bike.get_transform().get_forward_vector() * 8) 
target_location.x, target_location.y = np.clip(target_location.x, 0, 1), np.clip(target_location.y, 0, 1)

print("new target location: ", target_location)
target_transform = carla.Transform(target_location)
target_pylon = world.spawn_actor(target_pylon_bp, target_transform)

time.sleep(1)
# print(world.get_map().get_spawn_points())
spec_set_transform = carla.Transform(bike.get_transform().transform(carla.Location(x=-4, y=2, z=2)), bike.get_transform().rotation)
spectator.set_transform(spec_set_transform) 
print("spectator transform: ", spec_set_transform)



target_location = [target_transform.location.x, target_transform.location.y]
bike_location = [bike_spawn_transform.location.x, bike_spawn_transform.location.y] 
dx = target_location[0] - bike_location[0]
dy = target_location[1] - bike_location[1]
target_direction = np.arctan2(dy, dx) / np.pi
print("target_direction: ", target_direction) 

bike_rotation = bike.get_transform().rotation.yaw / 180

print("bike rotation: ", bike_rotation)
# for _ in range(30):
#    # time.sleep(0.5)
#     bike.apply_control(carla.VehicleControl(throttle=100))
# XMAX=13.5
# XMIN=-10
# YMAX=-35
# YMIN=-124
# spectator.set_transform(carla.Transform(carla.Location(x=7.727516, y=-127.762421, z=13), carla.Rotation(pitch=-19.756622, yaw=-250.927879, roll=0.000024)))
# for _ in range(5):
#      time.sleep(10)
#      spectator_transform = spectator.get_transform()
#      print("specpos:" + str(spectator_transform))


