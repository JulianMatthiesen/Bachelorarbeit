from GoToGoalBikeEnv import BikeEnv
from gym.wrappers import FlattenObservation

env = BikeEnv()
env = FlattenObservation(env)

episodes = 10

for episode in range(episodes):
    done = False
    obs = env.reset()
    while not done:
        random_action = env.action_space.sample()
       # print("random Action: ", random_action)
        obs, reward, done, info = env.step(random_action)
       # print("observation: ", obs)
        error_angle = obs[0] - obs[2]
       # print("error_angle: ", error_angle)
