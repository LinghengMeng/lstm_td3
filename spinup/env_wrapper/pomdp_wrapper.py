import numpy as np
import gym


class POMDPWrapper(gym.ObservationWrapper):
    def __init__(self, env_name, pomdp_type='remove_velocity', flicker_prob=0.2):
        """

        :param env_name:
        :param pomdp_type:
            1. remove_velocity: remove velocity related observation
            2. flickering: obscure the entire observation with a certain probability at each time step with the
                  probability flicker_prob.
            3.
        """
        super().__init__(gym.make(env_name))
        self.pomdp_type = pomdp_type
        self.flicker_prob = flicker_prob

        if pomdp_type == 'remove_velocity':
            # Remove velocity info
            # OpenAIGym
            #  1. Classic Control
            if env_name == "Pendulum-v0":
                self.remain_obs_idx = np.arange(0, 2)
            elif env_name == "Acrobot-v1":
                self.remain_obs_idx = list(np.arange(0, 4))
            elif env_name == "MountainCarContinuous-v0":
                self.remain_obs_idx = list([0])
            #  1. MuJoCo
            elif env_name == "HalfCheetah-v3" or env_name == "HalfCheetah-v2":
                self.remain_obs_idx = np.arange(0, 8)
            elif env_name == "Ant-v3" or env_name == "Ant-v2":
                self.remain_obs_idx = list(np.arange(0, 13)) + list(np.arange(27, 111))
            elif env_name == 'Walker2d-v3' or env_name == "Walker2d-v2":
                self.remain_obs_idx = np.arange(0, 8)
            elif env_name == 'Hopper-v3' or env_name == "Hopper-v2":
                self.remain_obs_idx = np.arange(0, 5)
            elif env_name == "InvertedPendulum-v2":
                self.remain_obs_idx = np.arange(0, 2)
            elif env_name == "InvertedDoublePendulum-v2":
                self.remain_obs_idx = list(np.arange(0, 5)) + list(np.arange(8, 11))
            elif env_name == "Swimmer-v3" or env_name == "Swimmer-v2":
                self.remain_obs_idx = np.arange(0, 3)
            elif env_name == "Thrower-v2":
                self.remain_obs_idx = list(np.arange(0, 7)) + list(np.arange(14, 23))
            elif env_name == "Striker-v2":
                self.remain_obs_idx = list(np.arange(0, 7)) + list(np.arange(14, 23))
            elif env_name == "Pusher-v2":
                self.remain_obs_idx = list(np.arange(0, 7)) + list(np.arange(14, 23))
            elif env_name == "Reacher-v2":
                self.remain_obs_idx = list(np.arange(0, 6)) + list(np.arange(8, 11))
            elif env_name == 'Humanoid-v3' or env_name == "Humanoid-v2":
                self.remain_obs_idx = list(np.arange(0, 22)) + list(np.arange(45, 185)) + list(np.arange(269, 376))
            elif env_name == 'HumanoidStandup-v2':
                self.remain_obs_idx = list(np.arange(0, 22)) + list(np.arange(45, 185)) + list(np.arange(269, 376))
            # PyBulletGym
            #  1. MuJoCo
            elif env_name == 'HalfCheetahMuJoCoEnv-v0':
                self.remain_obs_idx = np.arange(0, 8)
            elif env_name == 'AntMuJoCoEnv-v0':
                self.remain_obs_idx = list(np.arange(0, 13)) + list(np.arange(27, 111))
            elif env_name == 'Walker2DMuJoCoEnv-v0':
                self.remain_obs_idx = np.arange(0, 8)
            elif env_name == 'HopperMuJoCoEnv-v0':
                self.remain_obs_idx = np.arange(0, 7)
            elif env_name == 'InvertedPendulumMuJoCoEnv-v0':
                self.remain_obs_idx = np.arange(0, 3)
            elif env_name == 'InvertedDoublePendulumMuJoCoEnv-v0':
                self.remain_obs_idx = list(np.arange(0, 5)) + list(np.arange(8, 11))
            #  2. Roboschool
            elif env_name == 'HalfCheetahPyBulletEnv-v0':
                self.remain_obs_idx = list(set(np.arange(0, 26)) - set(np.arange(3, 6)))
            elif env_name == 'AntPyBulletEnv-v0':
                self.remain_obs_idx = list(set(np.arange(0, 28)) - set(np.arange(3, 6)))
            elif env_name == 'Walker2DPyBulletEnv-v0':
                self.remain_obs_idx = list(set(np.arange(0, 22)) - set(np.arange(3, 6)))
            elif env_name == 'HopperPyBulletEnv-v0':
                self.remain_obs_idx = list(set(np.arange(0, 15)) - set(np.arange(3, 6)))
            elif env_name == 'InvertedPendulumPyBulletEnv-v0':
                self.remain_obs_idx = list(set(np.arange(0, 5)) - set([1, 4]))
            elif env_name == 'InvertedDoublePendulumPyBulletEnv-v0':
                self.remain_obs_idx = list(set(np.arange(0, 9)) - set([1, 5, 8]))
            elif env_name == 'ReacherPyBulletEnv-v0':
                self.remain_obs_idx = list(set(np.arange(0, 9)) - set([6, 8]))
            else:
                raise ValueError('POMDP for {} is not defined!'.format(env_name))

            # Redefine observation_space
            obs_low = np.array([-np.inf for i in range(len(self.remain_obs_idx))], dtype="float32")
            obs_high = np.array([np.inf for i in range(len(self.remain_obs_idx))], dtype="float32")
            self.observation_space = gym.spaces.Box(obs_low, obs_high)
        elif pomdp_type == 'flickering':
            pass
        else:
            raise ValueError('pomdp_type was not specified!')

    def observation(self, obs):
        if self.pomdp_type == 'remove_velocity':
            return obs.flatten()[self.remain_obs_idx]
        elif self.pomdp_type == 'flickering':
            if np.random.rand() <= self.flicker_prob:
                return np.zeros(obs.shape)
            else:
                return obs.flatten()
