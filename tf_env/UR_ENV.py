import tensorflow as tf
import numpy as np

from scipy.spatial.transform import Rotation as rotate
from scipy.spatial import distance

from tf_agents.environments import py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step

class UR_env(py_environment.PyEnvironment):    
    #import date parameters of joints from
    #https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
    _ur10_params=(
        127.3,
        612,
        572.3,
        163.941,
        115.7,
        92.2
    )
    _ur5_params=(
        89.159,
        425,
        392.25,
        109.15,
        94.65,
        82.3
    )
    _ur3_params=np.array(
        [
            [0,         0,          151.9],
            [-243.65,   0,          0],
            [-213.25,   0,           0],
            [0,         0,          112.35],
            [0,         0,          85.35],
            [0,         0,          81.9]
        ]
    )
    _joint_rotation=np.array(
        [[0,0,0],
        [0,-90,0],
        [0,0,0],
        [0,0,0],
        [0,-90,0],
        [0,90,0]]
    )
    
    _home_position=np.array([180,90,0,0,0,0],dtype=np.float32)

    def __init__(self):
        #specification
        self._action_spec = array_spec.BoundedArraySpec(
        shape=(6,), dtype=np.float32, minimum=-1, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
        shape=(2,3), dtype=np.float32, minimum=-np.inf, maximum=np.inf, name='observation')        
        self._episode_ended = False     

        self.max_anglular_velocity=5
        self.duration_step=1
        self.stop_acuracy=1
        self._batch_size=25
        self.stop_counter:int=0
        self.max_steps=200

        self._joint_state=np.copy(self._home_position)       
        self._begin_position =self.__forward_kinematic_ur3(self._joint_state)        
        self._state=np.copy(self._begin_position)
        self._target=np.array(
            [[500,500,0],
            [0,180,00]],
            dtype=np.float32
        )
        
        self._all_distance= distance.euclidean(self._begin_position[0],self._target[0])
        self._previous_distance:distance=distance.euclidean(self._begin_position[0],self._target[0])

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.stop_counter=0
        self._state = np.copy(self._begin_position)
        self._joint_state=np.copy(self._home_position)
        self._episode_ended = False
        return time_step.restart(self._state)
    
    def set_target(self, target):
        self._target=np.copy(target)
    
    # def render_policy_net(model, n_max_steps=200, seed=42):
    #     frames = []
    #     env = gym.make("CartPole-v1")
    #     env.seed(seed)
    #     np.random.seed(seed)
    #     obs = env.reset()
    #     for step in range(n_max_steps):
    #         frames.append(env.render(mode="rgb_array"))
    #         left_proba = model.predict(obs.reshape(1, -1))
    #         action = int(np.random.rand() > left_proba)
    #         obs, reward, done, info = env.step(action)
    #         if done:
    #             break
    #     env.close()
    #     return frames
    
    def __forward_kinematic_ur3(self,parameters):
        position:np.float64=[0,0,0]
        orientation=rotate.from_euler('ZXZ',[0,0,0],degrees=True)
        base_angle=np.copy(self._joint_rotation)
        for index, param in enumerate(parameters):
            base_angle[index][2]+=param
            new_rotation=rotate.from_euler('ZXZ',base_angle[index],degrees=True)
            orientation:rotate=orientation*new_rotation
            rot_pos=orientation.apply(self._ur3_params[index])
            position+=rot_pos
        orientation=orientation.as_euler('ZXZ',degrees=True)
        comlex=np.array([position,orientation],dtype=np.float32)
        return comlex
    
    def __find_reward(self):        
        this_distance=distance.euclidean(self._target[0],self._state[0])
        if this_distance < self.stop_acuracy:
            self._episode_ended=True   
        if self.stop_counter>self.max_steps:
            self._episode_ended=True 
        if this_distance<self._previous_distance:
            # reward=1-(this_distance/self._previous_distance)
            reward=1
        else:
            reward=0
        self._previous_distance=this_distance
        return reward


    def _step(self, action):       
        if self._episode_ended:
            return self.reset()
        
        self.stop_counter+=1

        for index, act in enumerate(action):
            angle=(self.max_anglular_velocity*act)*self.duration_step
            self._joint_state[index]+=angle
            if (self._joint_state[index]>180):
                self._joint_state[index]=self._joint_state[index]-360
            elif (self._joint_state[index]<-180):
                self._joint_state[index]=self._joint_state[index]+360

        self._state = self.__forward_kinematic_ur3(self._joint_state)
        reward=self.__find_reward()
        if self._episode_ended==True:
            return time_step.termination(self._state, reward)
        return time_step.transition(self._state, reward)
        

if __name__=='__main__':
    environment = UR_env()
    # tf_env=tf_py_environment.TFPyEnvironment(environment)
    
    # action =np.array([[1,1,0,0,0,0]],dtype=np.float32)
    
    # observation=tf_env.reset()
    # observation=tf_env.step(action)
    # print(observation.observation[0][0])


    # ts=tf_env.reset()
    # num_steps = 3
    # transitions = []
    # reward = 0
    # for i in range(num_steps):
    #     action =np.array([0,1,0,0,0,0],dtype=np.float32)
    #     next_ts = tf_env.step(action)
    #     # transitions.append([ts, action, next_ts])
    #     # reward += next_ts.reward
    #     # ts = next_ts

    # np_transitions = tf.nest.map_structure(lambda x: x.numpy(), transitions)
    # print('\n'.join(map(str, np_transitions)))
    # print('Total reward:', reward.numpy())



    # observation=environment.reset()
    # print(observation)

    # observation=environment.step(np.array([0,1,0,0,0,0],dtype=np.float32))
    # print(observation)

    utils.validate_py_environment(environment,episodes=5,)



