from stable_baselines3.common.env_checker import check_env
from GoToGoalBikeEnv import BikeEnv
from gym.wrappers import FlattenObservation


env = BikeEnv()
env = FlattenObservation(env)
check_env(env)