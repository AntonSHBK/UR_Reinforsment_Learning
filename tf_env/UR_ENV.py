import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math


from scipy.spatial.transform import Rotation as rotate
from scipy.spatial import distance

from tf_agents.environments import py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class UR_env_simple():
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
        [0,90,0],
        [0,0,0],
        [0,0,0],
        [0,90,0],
        [0,-90,0]]
    )
    
    _home_position = np.array([180,-90,0,0,0,0],dtype=np.float32)
    _defoult_target = np.array(
            [[250,0,100],
            [0,180,0]],
            dtype=np.float32
        )
    
    def __init__(
        self, 
        max_steps:int=10, 
        discount=0.95, 
        joints_angles =_home_position, 
        target = _defoult_target,
        max_anglular_velocity=5,
        duration_step=1,
        stop_acuracy=10
        ):
      
        self._episode_ended = False

        self.max_anglular_velocity=max_anglular_velocity
        self.duration_step=duration_step
        self.stop_acuracy=stop_acuracy
        self.stop_counter:int=0
        self.max_steps=max_steps
        self.discount = discount

        self._joints_angles=np.copy(joints_angles)


        self._joints_positions = []    
        self._begin_position =self.__forward_kinematic_ur3(self._joints_angles)
        self._state = np.copy(self._begin_position)
        self._target = np.copy(target)

        self._target_angles = self.__find_angles(target[1])   
        self._previous_angles = self.__find_angles(self._state[1])   
        self._previous_quotient_of_angles = self.__compute_angle_quotient(self._previous_angles)      
        
        self._all_distance= distance.euclidean(self._begin_position[0],self._target[0])
        self._previous_distance=self._all_distance

    # def action_spec(self):
    #     """Return the actions that should be provided to `step()`"""
    #     return self._action_spec

    # def observation_spec(self):
    #     """Return the observations provided by the environment."""
    #     return self._observation_spec
    
    # def discount_spec(self):
    #     """Return the discount that are returned by `step()`."""
    #     return self._discount_spec

    # def reward_spec(self):
    #     """Return the rewards that are returned by `step()`."""
    #     return self._reward_spec

    def reset(self):
        """Return the rewards that are returned by `step()`."""
        self.stop_counter=0
        self._state = np.copy(self._begin_position)
        self._joints_angles=np.copy(self._home_position)
        self._episode_ended = False
        self._previous_distance = self._all_distance
        return np.reshape(self._state,6)

    def set_target(self, target):
        self._target=np.copy(target)
    
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
            j_pos = position.copy()
            j_orient =orientation.as_euler('ZXZ',degrees=True)
            self._joints_positions.append(np.array([j_pos,j_orient],dtype=np.float32))
        orientation=orientation.as_euler('ZXZ',degrees=True)
        comlex=np.array([position,orientation],dtype=np.float32)
        return comlex
    
    def __find_reward(self):        
        this_distance=distance.euclidean(self._target[0],self._state[0])
        if this_distance < self.stop_acuracy or self.stop_counter>self.max_steps:
            self._episode_ended=True
            if this_distance < self.stop_acuracy:
                return 1   
        # if self.stop_counter>self.max_steps:
        #     self._episode_ended=True 
        angles = self.__find_angles(self._state[1])
        quotient_of_angles = self.__compute_angle_quotient(angles)
        
        discount=math.pow(self.discount,1+self.stop_counter)
        reward_1=1-(this_distance/self._previous_distance)
        reward_2=1-(quotient_of_angles/self._previous_quotient_of_angles)
        reward=(reward_1+reward_2)*discount

        if reward>0:
             self._previous_distance=this_distance
             self._previous_quotient_of_angles=quotient_of_angles    
        return reward

    def __find_angles(self, angles_ZXZ):
        rot_matrix=rotate.from_euler('ZXZ',angles_ZXZ,degrees=True)
        axis_x=rot_matrix.apply([1,0,0])
        axis_y=rot_matrix.apply([0,1,0])
        axis_z=rot_matrix.apply([0,0,1])
        
        dist = distance.cosine([1,0,0],axis_x)
        angle_x=np.rad2deg(np.arccos(1 - dist))

        dist = distance.cosine([0,1,0],axis_y)
        angle_y=np.rad2deg(np.arccos(1 - dist))

        dist = distance.cosine([0,0,1],axis_z)
        angle_z=np.rad2deg(np.arccos(1 - dist))

        return [angle_x,angle_y,angle_z]    
    
    def __compute_angle_quotient(self, angle):
        def_angle=self._target[1]
        return math.sqrt(
            pow(def_angle[0]-angle[0],2)+
            pow(def_angle[1]-angle[1],2)+
            pow(def_angle[2]-angle[2],2)
            )

    def step(self, action):       
        if self._episode_ended:
            return self.reset()
        
        self.stop_counter+=1
        self._joints_positions=[]

        for index, act in enumerate(action):
            angle=(self.max_anglular_velocity*act)*self.duration_step
            self._joints_angles[index]+=angle
            if (self._joints_angles[index]>180):
                self._joints_angles[index]=self._joints_angles[index]-360
            elif (self._joints_angles[index]<-180):
                self._joints_angles[index]=self._joints_angles[index]+360

        self._state = self.__forward_kinematic_ur3(self._joints_angles)
        reward=self.__find_reward()
        return np.reshape(self._state,6), reward, self._episode_ended, self.discount

    
    def _convert_to_array (self, fig):
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data
    
    def render(self):
        fig = plt.figure()
        frame = plt.axes(projection="3d")
        frame.set_xlim3d(-500, 500)
        frame.set_ylim3d(-500, 500)
        frame.set_zlim3d(0, 1000)
        frame.plot(
            self._target[0][0],
            self._target[0][1],
            self._target[0][2],
            '-ro',
            label='target')
        colors = ('r','g','b','y','c','m','k')
        iterator = iter(colors)        
        for index in range(len(self._joints_positions[:-1])):          
            color = next(iterator)
            x=[self._joints_positions[index][0][0],self._joints_positions[index+1][0][0]]
            y=[self._joints_positions[index][0][1],self._joints_positions[index+1][0][1]]
            z=[self._joints_positions[index][0][2],self._joints_positions[index+1][0][2]]
            frame.plot(x,y,z,color=color)
        return self._convert_to_array(fig)

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
        [0,90,0],
        [0,0,0],
        [0,0,0],
        [0,90,0],
        [0,-90,0]]
    )
    
    _home_position = np.array([180,-90,0,0,0,0],dtype=np.float32)
    _defoult_target = np.array(
            [250,-100,100,0,180,0],
            dtype=np.float32
        )
        

    def __init__(
        self, 
        max_steps=10, 
        discount=0.95, 
        joints_angles =_home_position, 
        target = _defoult_target,
        max_anglular_velocity=5,
        duration_step=1,
        stop_acuracy=10
        ):
        """Information about this class
        
        """
        #specification
        self._action_spec = array_spec.BoundedArraySpec(
        shape=(6,), dtype=np.float32, minimum=[-1], maximum=[1], name='action')
        
        self._observation_spec = array_spec.BoundedArraySpec(
        shape=(6,), dtype=np.float32, minimum=[-1000,-1000,-1000,-180,-180,-180], maximum=[1000,1000,1000,180,180,180], name='observation')                
        
        self._discount_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.float32, minimum=0., maximum=1, name='discount')

        self._reward_spec = array_spec.ArraySpec(shape=(), dtype=np.float32, name='reward')
        #################### 
        
        self._episode_ended = False

        self.max_anglular_velocity=max_anglular_velocity
        self.duration_step=duration_step
        self.stop_acuracy=stop_acuracy
        self.stop_counter:int=0
        self.max_steps=max_steps
        self.discount = discount

        self._joints_angles=np.copy(joints_angles)

        self.angle_limitations = (
            (0,360),
            (-180,0),
            (0,180),
            (-180,180),
            (-180,180),
            (-180,180)
        )
        self._joints_positions = []    
        self._begin_position =self.__forward_kinematic_ur3(self._joints_angles)
        self._state = np.copy(self._begin_position)
        self._target = np.copy(target)

        self._target_angles = self.__find_angles(target[3:])   
        self._previous_angles = self.__find_angles(self._state[3:])   
        self._previous_quotient_of_angles = self.__compute_angle_quotient(self._previous_angles)
        self._defoult_angle_quotient= self.__compute_angle_quotient(self._previous_angles)         
        
        self._all_distance= distance.euclidean(self._begin_position[:3],self._target[:3])
        self._previous_distance=distance.euclidean(self._begin_position[:3],self._target[:3])

        self.trajectory = []
        self.trajectory.append(self._state[:3])

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
        self._joints_angles=np.copy(self._home_position)
        self._episode_ended = False
        self._previous_distance = self._all_distance
        self._previous_quotient_of_angles=self._defoult_angle_quotient
        self.trajectory = []
        return ts.restart(self._state)

    # def batched(self) -> bool:
    #     return False
    
    # def batch_size(self) -> Optional[int]:
    #     return Optional(2)

    def set_target(self, target):
        self._target=np.copy(target)
    
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
            j_pos = position.copy()
            j_orient =orientation.as_euler('ZXZ',degrees=True)
            self._joints_positions.append(np.array([j_pos,j_orient],dtype=np.float32))
        orientation=np.array(orientation.as_euler('ZXZ',degrees=True),dtype=np.float32)
        comlex = np.concatenate((position, orientation), dtype=np.float32, axis=None)
        return comlex
    
    def __find_reward(self):        
        this_distance=distance.euclidean(self._target[:3],self._state[:3])
        if this_distance < self.stop_acuracy or self.stop_counter>self.max_steps:
            self._episode_ended=True
            # if this_distance < self.stop_acuracy:
            #     return 1   

        discount=math.pow(self.discount,1+self.stop_counter)             
        
        this_angles = self.__find_angles(self._state[3:])
        this_quotient_of_angles = self.__compute_angle_quotient(this_angles) 

        line_difference = self._previous_distance-this_distance
        reward_1=line_difference/self._all_distance

        angle_difference=self._previous_quotient_of_angles- this_quotient_of_angles
        reward_2=angle_difference/self._defoult_angle_quotient

        # reward_2=0

        self._previous_distance=this_distance
        self._previous_quotient_of_angles=this_quotient_of_angles

        reward=(reward_1+reward_2)*discount
        return reward

    def __find_angles(self, angles_ZXZ):
        rot_matrix=rotate.from_euler('ZXZ',angles_ZXZ,degrees=True)
        axis_x=rot_matrix.apply([1,0,0])
        axis_y=rot_matrix.apply([0,1,0])
        axis_z=rot_matrix.apply([0,0,1])
        
        dist = distance.cosine([1,0,0],axis_x)
        angle_x=np.rad2deg(np.arccos(1 - dist))

        dist = distance.cosine([0,1,0],axis_y)
        angle_y=np.rad2deg(np.arccos(1 - dist))

        dist = distance.cosine([0,0,1],axis_z)
        angle_z=np.rad2deg(np.arccos(1 - dist))

        return [angle_x,angle_y,angle_z]    
    
    def __compute_angle_quotient(self, angle):
        def_angle=self._target[:3]
        return math.sqrt(
            pow(def_angle[0]-angle[0],2)+
            pow(def_angle[1]-angle[1],2)+
            pow(def_angle[2]-angle[2],2)
            )

    def _step(self, action):       
        if self._episode_ended:
            return self.reset()
        
        self.stop_counter+=1
        self._joints_positions=[]

        for index, act in enumerate(action):
            angle=(self.max_anglular_velocity*act)*self.duration_step

            self._joints_angles[index]+=angle
                
            if (self._joints_angles[index]>180):
                self._joints_angles[index]=self._joints_angles[index]-360
            elif (self._joints_angles[index]<-180):
                self._joints_angles[index]=self._joints_angles[index]+360

        self._state = self.__forward_kinematic_ur3(self._joints_angles)
        reward=self.__find_reward()

        self.trajectory.append(self._state[:3])

        if self._episode_ended==True:
            return ts.termination(observation = self._state, reward = reward)
        return ts.transition(observation = self._state, reward = reward, discount = self.discount)
    
    def _convert_to_array (self, fig):
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data
    
    def render(self):
        target=self._target[:3]
        fig = plt.figure()
        frame = fig.add_subplot(111,projection="3d")
        frame.set_xlim3d(-500, 500)
        frame.set_ylim3d(-500, 500)
        frame.set_zlim3d(0, 1000)
        # frame.set_xlabel('X-axis')
        # frame.set_ylabel('Y-axis')
        # frame.set_zlabel('Z-axis')
        target = frame.plot(
            target[0],
            target[1],
            target[2],
            '-ro',
            label='target')
        colors = ('r','g','b','y','c','m','k')
        iterator = iter(colors)   

        if self.stop_counter!=1:
            for i in range(len(self.trajectory)-1):
                x=[self.trajectory[i][0],self.trajectory[i+1][0]]
                y=[self.trajectory[i][1],self.trajectory[i+1][1]]
                z=[self.trajectory[i][2],self.trajectory[i+1][2]]
                trajectory = frame.plot(x,y,z,color='k')  
        link = []
        for index in range(len(self._joints_positions[:-1])):          
            color = next(iterator)
            x=[self._joints_positions[index][0][0],self._joints_positions[index+1][0][0]]
            y=[self._joints_positions[index][0][1],self._joints_positions[index+1][0][1]]
            z=[self._joints_positions[index][0][2],self._joints_positions[index+1][0][2]]
            link.append(frame.plot(x,y,z,color=color))
        x = [self._joints_positions[0][0][0],0]
        y = [self._joints_positions[0][0][1],0]
        z = [self._joints_positions[0][0][2],0]
        base = frame.plot(x,y,z,color='m')
        # plt.show()
        return self._convert_to_array(fig)

class UR_env_1(py_environment.PyEnvironment):    
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
        [0,90,0],
        [0,0,0],
        [0,0,0],
        [0,90,0],
        [0,-90,0]]
    )
    
    _home_position = np.array([180,-90,0,0,0,0],dtype=np.float32)
    _defoult_target = np.array(
            [250,-100,100,0,180,0],
            dtype=np.float32
        )
        

    def __init__(
        self, 
        max_steps=10, 
        discount=0.95, 
        joints_angles =_home_position, 
        target = _defoult_target,
        max_anglular_velocity=5,
        duration_step=1,
        stop_acuracy=10
        ):
        """Information about this class
        
        """
        #specification
        self._action_spec = array_spec.BoundedArraySpec(
        shape=(2,), dtype=np.float32, minimum=-1, maximum=1, name='action')
        
        self._observation_spec = array_spec.BoundedArraySpec(
        shape=(6,), dtype=np.float32, minimum=[-1000,-1000,-1000,-180,-180,-180], maximum=[1000,1000,1000,180,180,180], name='observation')                
        
        self._discount_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.float32, minimum=0., maximum=1., name='discount')

        self._reward_spec = array_spec.ArraySpec(shape=(), dtype=np.float32, name='reward')
        #################### 
        
        self._episode_ended = False

        self.max_anglular_velocity=max_anglular_velocity
        self.duration_step=duration_step
        self.stop_acuracy=stop_acuracy
        self.stop_counter:int=0
        self.max_steps=max_steps
        self.discount = discount

        self._joints_angles=np.copy(joints_angles)

        self.angle_limitations = (
            (0,360),
            (-180,0),
            (0,180),
            (-180,180),
            (-180,180),
            (-180,180)
        )
        self._joints_positions = []    
        self._begin_position =self.__forward_kinematic_ur3(self._joints_angles)
        self._state = np.copy(self._begin_position)
        self._target = np.copy(target)

        self._target_angles = self.__find_angles(target[3:])   
        self._previous_angles = self.__find_angles(self._state[3:])   
        self._previous_quotient_of_angles = self.__compute_angle_quotient(self._previous_angles)
        self._defoult_angle_quotient= self.__compute_angle_quotient(self._previous_angles)         
        
        self._all_distance= distance.euclidean(self._begin_position[:3],self._target[:3])
        self._previous_distance=distance.euclidean(self._begin_position[:3],self._target[:3])


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
        self._joints_angles=np.copy(self._home_position)
        self._episode_ended = False
        self._previous_distance = self._all_distance
        self._previous_quotient_of_angles=self._defoult_angle_quotient
        return ts.restart(self._state)

    # def batched(self) -> bool:
    #     return False
    
    # def batch_size(self) -> Optional[int]:
    #     return Optional(2)

    def set_target(self, target):
        self._target=np.copy(target)
    
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
            j_pos = position.copy()
            j_orient =orientation.as_euler('ZXZ',degrees=True)
            self._joints_positions.append(np.array([j_pos,j_orient],dtype=np.float32))
        orientation=np.array(orientation.as_euler('ZXZ',degrees=True),dtype=np.float32)
        comlex = np.concatenate((position, orientation), dtype=np.float32, axis=None)
        return comlex
    
    def __find_reward(self):        
        this_distance=distance.euclidean(self._target[:3],self._state[:3])
        if this_distance < self.stop_acuracy or self.stop_counter>self.max_steps:
            self._episode_ended=True
            # if this_distance < self.stop_acuracy:
            #     return 1   

        discount=math.pow(self.discount,1+self.stop_counter)             
        
        this_angles = self.__find_angles(self._state[3:])
        this_quotient_of_angles = self.__compute_angle_quotient(this_angles) 

        line_difference = self._previous_distance-this_distance
        reward_1=line_difference/self._all_distance

        angle_difference=self._previous_quotient_of_angles- this_quotient_of_angles
        reward_2=angle_difference/self._defoult_angle_quotient

        reward_2=0

        self._previous_distance=this_distance
        self._previous_quotient_of_angles=this_quotient_of_angles

        reward=(reward_1+reward_2)*discount
        return reward

    def __find_angles(self, angles_ZXZ):
        rot_matrix=rotate.from_euler('ZXZ',angles_ZXZ,degrees=True)
        axis_x=rot_matrix.apply([1,0,0])
        axis_y=rot_matrix.apply([0,1,0])
        axis_z=rot_matrix.apply([0,0,1])
        
        dist = distance.cosine([1,0,0],axis_x)
        angle_x=np.rad2deg(np.arccos(1 - dist))

        dist = distance.cosine([0,1,0],axis_y)
        angle_y=np.rad2deg(np.arccos(1 - dist))

        dist = distance.cosine([0,0,1],axis_z)
        angle_z=np.rad2deg(np.arccos(1 - dist))

        return [angle_x,angle_y,angle_z]    
    
    def __compute_angle_quotient(self, angle):
        def_angle=self._target[:3]
        return math.sqrt(
            pow(def_angle[0]-angle[0],2)+
            pow(def_angle[1]-angle[1],2)+
            pow(def_angle[2]-angle[2],2)
            )

    def _step(self, action):       
        if self._episode_ended:
            return self.reset()
        
        self.stop_counter+=1
        self._joints_positions=[]

        for index, act in enumerate(action):
            angle=(self.max_anglular_velocity*act)*self.duration_step
            # angle*=math.pow(self.discount,1+self.stop_counter) 
            self._joints_angles[index]+=angle
            # if self._joints_angles[index]<self.angle_limitations[index][0]:
            #     self._joints_angles[index]=self.angle_limitations[index][0]
            # elif self._joints_angles[index]>self.angle_limitations[index][1]:
            #     self._joints_angles[index]=self.angle_limitations[index][1]
                
            if (self._joints_angles[index]>180):
                self._joints_angles[index]=self._joints_angles[index]-360
            elif (self._joints_angles[index]<-180):
                self._joints_angles[index]=self._joints_angles[index]+360

        self._state = self.__forward_kinematic_ur3(self._joints_angles)
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
        target=self._target[:3]
        fig = plt.figure()
        frame = plt.axes(projection="3d")
        frame.set_xlim3d(-500, 500)
        frame.set_ylim3d(-500, 500)
        frame.set_zlim3d(0, 1000)
        frame.plot(
            target[0],
            target[1],
            target[2],
            '-ro',
            label='target')
        colors = ('r','g','b','y','c','m','k')
        iterator = iter(colors)        
        for index in range(len(self._joints_positions[:-1])):          
            color = next(iterator)
            x=[self._joints_positions[index][0][0],self._joints_positions[index+1][0][0]]
            y=[self._joints_positions[index][0][1],self._joints_positions[index+1][0][1]]
            z=[self._joints_positions[index][0][2],self._joints_positions[index+1][0][2]]
            frame.plot(x,y,z,color=color)
        x = [self._joints_positions[0][0][0],0]
        y = [self._joints_positions[0][0][1],0]
        z = [self._joints_positions[0][0][2],0]
        frame.plot(x,y,z,color='m')
        return self._convert_to_array(fig)


    

if __name__=='__main__':
    target = np.array(
            [250,200,120, 0,180,0],
            dtype=np.float32
        )

    environment = UR_env(target=target)
    obs = environment.reset()

    img = environment.render()
    # print(environment.time_step_spec())


    # from tf_agents.environments import tf_py_environment
    # tf_env=tf_py_environment.TFPyEnvironment(environment)

    # # print(tf_env.time_step_spec)    
    # action =np.array([[1,0,0,0,0,0]],dtype=np.float32)    
    # obs=tf_env.reset()
    # print(obs)    
    # obs=tf_env.step(action)

    # print(obs.observation)
    # print(tf_env.batched)
    # print(tf_env.batch_size)


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
    print('ok')



