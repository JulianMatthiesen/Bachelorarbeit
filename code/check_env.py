from stable_baselines3.common.env_checker import check_env
from collision_avoidance_env_modified import BikeEnv
from gym.wrappers import FlattenObservation

"""use environment checker of stable baselines 3 to check if custom environment follows gymnasium interface"""

env = BikeEnv()
env = FlattenObservation(env)

check_env(env)