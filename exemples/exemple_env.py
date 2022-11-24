import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from math import pi


from scipy.spatial.transform import Rotation as rotate
from scipy.spatial import distance

from tf_agents.environments import py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class EXEMPLE_env(py_environment.PyEnvironment):    

    _defoult_joints_angles = np.array([0,0],dtype=np.float32)
    _defoult_target = np.array([-6,-4],dtype=np.float32)
    
    def __init__(
        self, 
        max_steps=10, 
        discount=0.95, 
        target = _defoult_target,
        max_anglular_velocity=5,
        duration_step=1,
        stop_acuracy=1
        ):
        #specification
        self._action_spec = array_spec.BoundedArraySpec(
        shape=(2,), dtype=np.float32, minimum=-1, maximum=1, name='action')
        
        self._observation_spec = array_spec.BoundedArraySpec(
        shape=(2,), dtype=np.float32, minimum=-np.inf, maximum=np.inf, name='observation')                
        
        self._discount_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.float32, minimum=0., maximum=1., name='discount')

        self._reward_spec = array_spec.ArraySpec(shape=(), dtype=np.float32, name='reward')
        #################### 
        
        self._episode_ended = False        
        self.max_steps=max_steps
        self.discount = discount
        self.max_anglular_velocity=(pi/180*max_anglular_velocity)
        self.duration_step=duration_step
        self.stop_acuracy=stop_acuracy
        self.leg_1 = 5
        self.leg_2 = 3

        self.stop_counter:int=0

        self._joints_angles=np.copy(self._defoult_joints_angles)   

        self._begin_position = self._find_point(self._joints_angles)        
        self._state = np.copy(self._begin_position)
        self._target = np.copy(target)        
        
        self._all_distance= distance.euclidean(self._begin_position, self._target)
        self._previous_distance=self._all_distance

    def action_spec(self):
        """Return the actions that should be provided to `step()`"""
        return self._action_spec

    def observation_spec(self):
        """Return the observations provided by the environment."""
        return self._observation_spec
    
    def discount_spec(self):
        """Return the discount that are returned by `step()`."""
        return self._discount_spec

    def reward_spec(self):
        """Return the rewards that are returned by `step()`."""
        return self._reward_spec

    def _reset(self):
        """Return the rewards that are returned by `step()`."""
        self.stop_counter=0
        self._state = np.copy(self._begin_position)
        self._joints_angles=np.copy(self._defoult_joints_angles)
        self._episode_ended = False
        self._previous_distance = self._all_distance
        return ts.restart(self._state)
   
    def _find_point(self,parameters):       
        self._joints_positions = [[0,0]]
        x1=self.leg_1*math.cos(parameters[0])
        y1=self.leg_1*math.sin(parameters[0])
        x2=x1+self.leg_2*math.cos(parameters[0]+parameters[1])
        y2=y1+self.leg_2*math.sin(parameters[0]+parameters[1])
        self._joints_positions.append([x1,y1])
        self._joints_positions.append([x2,y2])
        return np.array([x2,y2],dtype=np.float32)
    
    def __find_reward(self):        
        this_distance=distance.euclidean(self._target,self._state)
        if this_distance < self.stop_acuracy or self.stop_counter>self.max_steps:
            self._episode_ended=True
            if this_distance < self.stop_acuracy:
                return 1   
        discount=math.pow(self.discount,1+self.stop_counter)
        reward=1-(this_distance/self._previous_distance)
        reward=reward*discount
        if reward>0:
             self._previous_distance=this_distance    
        return reward


    def _step(self, action):       
        if self._episode_ended:
            return self.reset()        
        self.stop_counter+=1
        for index, act in enumerate(action):
            angle=(self.max_anglular_velocity*act)*self.duration_step
            self._joints_angles[index]+=angle
            if (self._joints_angles[index]>pi):
                self._joints_angles[index]=self._joints_angles[index]-(2*pi)
            elif (self._joints_angles[index]<-pi):
                self._joints_angles[index]=self._joints_angles[index]+(2*pi)

        self._state = self._find_point(self._joints_angles)
        reward=self.__find_reward()
        if self._episode_ended==True:
            return ts.termination(observation = self._state, reward = reward)
        return ts.transition(observation = self._state, reward = reward, discount = self.discount)
    
    def _convert_to_array (self, fig):
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data
    
    def render(self):
        fig = plt.figure()
        frame = plt.axes()
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
        plt.figure(figsize=(5,5))
        frame.plot(
            self._target[0],
            self._target[1],
            '-ro',
            label='target')
        colors = ('r','g','b','y','c','m','k')
        iterator = iter(colors)        
        for index in range(len(self._joints_positions[:-1])):          
            color = next(iterator)
            x=[self._joints_positions[index][0],self._joints_positions[index+1][0]]
            y=[self._joints_positions[index][1],self._joints_positions[index+1][1]]

            frame.plot(x,y,color=color)
        fig.show()
        return self._convert_to_array(fig)

if __name__=='__main__':
    environment = EXEMPLE_env()
    utils.validate_py_environment(environment, episodes=5)
    environment.render()






