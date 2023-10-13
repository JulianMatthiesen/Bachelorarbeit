from stable_baselines3 import  PPO
import os
from collision_avoidance_env import BikeEnv
import carla

models_dir = f"models/newModelName"
logdir = f"logs/newModelName"

# create directories for the new models and the logs
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

env = BikeEnv()
obs = env.reset()



"""use this to create a new model from scratch """
# for observation without camera use MlpPolicy instead of MultiInputPolicy
model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=logdir)

"""use this to load and train an old model from the directory 'models' """
#model_path = f"models/oldModelName/80000.zip"
#model = PPO.load(model_path, env=env)

"""use this to load and train an old model with custom learning rate """
#model_path = f"models/oldModelName/80000.zip"
#custom_objects = { 'learning_rate': 0.00005 }
#loaded_model = PPO.load(model_path, env=env, custom_objects=custom_objects)



# set spectator view  
spectator = env.world.get_spectator()
transform = carla.Transform(carla.Location(x=-51, y=-126, z=62), carla.Rotation(pitch=-60, yaw=84, roll=0))
spectator.set_transform(transform)

# train for 400.000 steps and save every 10.000 steps
TIMESTEPS=10000
i = 0
while i < 40:
    i+=1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=logdir)
    model.save(f"{models_dir}/{TIMESTEPS*i}")

env.close()
        