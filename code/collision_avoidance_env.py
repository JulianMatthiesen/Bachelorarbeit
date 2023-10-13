#!/usr/bin/env python3

import glob
import math
import os
import sys
import time
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import gym
import numpy as np
from gym import spaces
import carla
import random


class BikeEnv(gym.Env):
    """Custom Environment that follows gym interface."""
    
    XMAX=-10
    XMIN=-70
    YMAX=-5
    YMIN=-130

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super(BikeEnv, self).__init__()
        # Define action and observation space

        high = np.array([
            1.0,   # throttle bike
            1.0    # steer bike
        ])

        low = np.array([
            -1.0,   # throttle bike
            -1.0   # steer bike
        ])

        self.action_space = spaces.Box(low=low, high=high, shape=(2,), dtype=np.float32)


        self.observation_space = spaces.Dict({
            "angle_to_target": spaces.Box(low=0, high=360, shape=(1,)),
            "distance": spaces.Box(low=0, high=140, shape=(1,)),
            "image": spaces.Box(low=1, high=255, shape=(21,), dtype=np.uint8)
            })
        

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world('Town03_Opt', carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)
        time.sleep(1)
        self.bp_lib = self.world.get_blueprint_library()
        self.bike_bp = self.bp_lib.find("vehicle.diamondback.century")

        self.sensor_data = {}
        self.bike, self.depth_sensor, self.collision_sensor = self.spawn_bike()

        # set synchronous mode und fixed time-step 
        settings = self.world.get_settings()
        settings.synchronous_mode = True  
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        self.target_location = None
        self.front_camera=None
        self.done = False
        self.reward = 0
        self.ret = 0
        self.depth_ret = 0
        self.steps_with_depth_rew = 0
        self.tick_count = 0
        self.max_time_steps = 1000
        self.world.tick()
        self.info = {"actions": []}

    def step(self, action):
        throttle = float((action[0] + 1)/ 2)
        steer=float(action[1])
        self.bike.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))

        # update bike_location
        self.bike_location = self.bike.get_transform().location
        
        self.world.debug.draw_string(self.target_location, "X", draw_shadow=False,
                                     color=carla.Color(r=255, g=0, b=0), life_time=0.1,
                                     persistent_lines=True)

        self.info["actions"].append(action.tolist())
        observation = self.get_observation()
        self.reward = self.calculate_reward()
        self.ret += self.reward
        return observation, self.reward, self.done, self.info

    def reset(self):
        if self.tick_count != 0:
            print("\nThis Episode: ")
            print("Steps: " + str(self.tick_count))
            print("Steps with Depth Reward: ", self.steps_with_depth_rew)   # number of steps where obstacles were close
            print("Depth Return: ", round(self.depth_ret))                  # proportion of Depth Reward to total return
            print("Return: ", round(self.ret))

        if not len(self.world.get_actors()) == 0:
            self.bike.destroy()
            self.depth_sensor.destroy()
            self.collision_sensor.destroy()

        self.bike, self.depth_sensor, self.collision_sensor = self.spawn_bike()
        
        while self.front_camera is None: 
            time.sleep(0.01)                                                # wait until front camera delivers first image

        self.done = False
        self.reward = 0
        self.ret = 0
        self.depth_ret = 0
        self.steps_with_depth_rew = 0
        self.tick_count = 0
        self.info = {"actions": []}
        self.world.tick()
        self.tick_count += 1
        return self.get_observation() 

    def close(self):
        self.bike.destroy()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
         
        self.world.tick()

    """
    def render(self):
        ...
    """

# ================== Helper Methods ==================

    def spawn_bike(self):        
        # spawn bike and set target location
        spawn_point_bike, spawn_point_target = self.get_random_spawn_point()
        spawn_transform = carla.Transform(spawn_point_bike, carla.Rotation())
        bike = self.world.try_spawn_actor(self.bike_bp, spawn_transform)
        self.target_location = spawn_point_target

        # if spawn failed
        while bike == None:
            spawn_point_bike, spawn_point_target = self.get_random_spawn_point()
            spawn_transform = carla.Transform(spawn_point_bike, carla.Rotation())
            bike = self.world.try_spawn_actor(self.bike_bp, spawn_transform)

        self.bike_location = spawn_point_bike
        self.target_location = spawn_point_target

        # spawn depth sensor
        depth_sensor_bp = self.bp_lib.find('sensor.camera.depth') 
        depth_sensor_bp.set_attribute("fov", "110") 
        depth_sensor_bp.set_attribute("image_size_x", "84")
        depth_sensor_bp.set_attribute("image_size_y", "84")
        depth_camera_init_trans = carla.Transform(carla.Location(x=0.5, z=0.95))
        depth_sensor = self.world.spawn_actor(depth_sensor_bp, depth_camera_init_trans, attach_to=bike)
        
        # initialize sensor_data
        image_w = depth_sensor_bp.get_attribute("image_size_x").as_int()
        image_h = depth_sensor_bp.get_attribute("image_size_y").as_int()
        self.sensor_data = {"depth_image": np.zeros((image_h, image_w, 4)),
                            "collision": False}
        
        depth_sensor.listen(lambda image: self.depth_callback(image, self.sensor_data))

        # spawn collision sensor
        collision_sensor = self.world.spawn_actor(self.world.get_blueprint_library().find('sensor.other.collision'), carla.Transform(), attach_to=bike)
        collision_sensor.listen(lambda event: self.collision_callback(event))

        return bike, depth_sensor, collision_sensor

    def depth_callback(self, image, data_dict):
        image.convert(carla.ColorConverter.LogarithmicDepth)
        data_dict["depth_image"] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        self.front_camera = data_dict["depth_image"]

    def collision_callback(self, event):
        self.sensor_data['collision'] = True

    def get_random_spawn_point(self):
       # initialize spawn- und target location oppositely in area 2 and 3 
       # modify for different training structure
        xSpawn = [] 
        ySpawn = [] 
        while True:
            spawn_places = random.sample(range(2, 4), 2)  
            for spawn_place in spawn_places:
                if spawn_place == 1:
                    xSpawn.append(random.uniform(-27, -16))
                    ySpawn.append(random.uniform(-125, -54))
                elif spawn_place == 2:
                    xSpawn.append(random.uniform(-64, -29))  
                    ySpawn.append(random.uniform(-125, -120))
                elif spawn_place == 3:
                    xSpawn.append(random.uniform(-63, -29))
                    ySpawn.append(random.uniform(-72, -65))
                elif spawn_place == 4:
                    xSpawn.append(random.uniform(-65, -53))
                    ySpawn.append(random.uniform(-98, -87))
                elif spawn_place == 5:
                    xSpawn.append(random.uniform(-50, -41))
                    ySpawn.append(random.uniform(-98, -94))

            if spawn_places[0] != spawn_places[1]:
                break
        location1 = carla.Location(x= float(xSpawn[0]), y= float(ySpawn[0]), z=0.05)
        location2 = carla.Location(x= float(xSpawn[1]), y= float(ySpawn[1]), z=0.0)
        return location1, location2
    
    def get_observation(self):
        # return observation of angle to target, distance and relevant pixels
        angle_to_target = [self.get_angle_to_target()]
        observed_dist = [self.get_distance_to_target()]
        observed_pixels = self.front_camera[42, :, 0]
        reduced_pixels = np.min(observed_pixels.reshape(-1, 4), axis=1)
        observation = {
            "angle_to_target": angle_to_target,
            "distance": observed_dist,
            "image": reduced_pixels
        }
        return observation

    def get_angle_to_target(self):
        # compute and return angle between forward vector and vector to target
        bike_transform = self.bike.get_transform()
        vector_to_target = self.target_location - bike_transform.location
        forward_vector  = bike_transform.get_forward_vector()
        angle_to_target = math.degrees(math.atan2(forward_vector.y, forward_vector.x) - math.atan2(vector_to_target.y, vector_to_target.x))
        angle_to_target = angle_to_target % 360
        return angle_to_target

    def get_distance_to_target(self):
        return self.bike_location.distance(self.target_location)
        
    def calculate_reward(self):
        # calculate Distance Reward
        current_distance = self.get_distance_to_target()
        distance_reward = -1.0 * (current_distance / 140)                                    # normalized to [-1;0]
        
        # calculate Depth Reward
        depth_reward = 0
        selected_data = self.front_camera[42, :, 0]
        selected_data = selected_data[selected_data < 20]
        if len(selected_data) != 0:
            self.steps_with_depth_rew += 1
            n = len(selected_data)                                                           # amount of pixels below 30
            depth_reward = -np.sum(20 - selected_data).astype(np.int64) / (n * 19)           # normalized to [-1;0] 
            depth_reward = depth_reward / 20
            self.depth_ret += depth_reward
        
        reward = (distance_reward + depth_reward)
   
        # end episode, when reaching the target, colliding, reaching time limit or leaving the square 
        if current_distance < 1.0:
            print("\ntarget reached!")
            self.done = True
            reward = 300

        elif self.sensor_data["collision"] == True:
            print("\ncollision!")
            self.done = True
            reward = -100
        
        elif self.tick_count >= self.max_time_steps:
            print("\nmax_time_steps reached!")
            self.done = True
            reward = -100

        elif not self.is_within_bounds():
             print("\nout of bounds!")
             self.done = True
             reward = 0

        self.world.tick()
        self.tick_count += 1

        return reward
    
    def is_within_bounds(self):
        return self.XMIN <= self.bike_location.x <= self.XMAX and self.YMIN <= self.bike_location.y <= self.YMAX

    
