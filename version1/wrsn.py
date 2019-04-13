import numpy as np
import gym
from gym import spaces

#define constant value

queue_len=2
battle_level=3
distance_level=1
sensor_node=2

selected_id=[0]
data_prob=[]
S=[]

def env_init():
    
    B=np.random.randint(0,battle_level+1,sensor_node).reshape(-1,1)
    D=np.random.randint(0,3,sensor_node).reshape(-1,1)
    H=np.random.randint(0,1,sensor_node).reshape(-1,1)
    S=np.hstack((B,D,H))
    selected_id[0]=np.random.randint(0,sensor_node)
    #S[sensor_node][0]  B  [1] D [2] H

    data_prob=np.ones(sensor_node)*0.6
    return S,selected_id,data_prob


    
def funEh(distance):
    Eh=[1,1]
    return Eh[distance]

def funPh(distance):
    Ph=[1,1]
    return Ph[distance]    


def calculate_transprob(S,sensor_id,selected,switch=None):
    data_overflow=0
    reward=0
    if selected==True:
        #BH
        assert switch==0 or switch ==1
        if switch==0:
            S[sensor_id][0]=min(S[sensor_id][0]+funEh(S[sensor_id][2]),battle_level)
        else:
            if funPh(S[sensor_id][2])<S[sensor_id][0]:
                S[sensor_id][0]=S[sensor_id][0]-funPh(S[sensor_id][2])
                S[sensor_id][1]=max(S[sensor_id][1]-1,0)
                if S[sensor_id][1]==0:
                    reward=-1
                else:
                    reward=1
            S[sensor_id][1]=S[sensor_id][1]+1 if data_prob[sensor_id]>np.random.uniform(0,1) else S[sensor_id][1]
            if S[sensor_id][1]>queue_len:
                S[sensor_id][1]=queue_len
                data_overflow=1
            else:
                data_overflow=0
   
    else:
        S[sensor_id][1]=S[sensor_id][1]+1 if data_prob[sensor_id]>np.random.uniform(0,1) else S[sensor_id][1]
        if S[sensor_id][1]>queue_len:
            S[sensor_id][1]=queue_len
            data_overflow=1
            reward=-1
        else:
            data_overflow=0

    return data_overflow,reward




#Sensor Node select Env
class nsEnv(gym.Env):
    def __init__(self,sensor_node):
        self.action_space = spaces.Discrete(sensor_node)

        self.lowstate=np.tile(np.array([0,0,0]),(sensor_node,1))
        self.highstate=np.tile(np.array([battle_level-1,queue_len-1,distance_level-1]),(sensor_node,1))
        
        self.observation_space = spaces.Box(low=self.lowstate,high=self.highstate,dtype=np.uint8)


    def step(self,action):
        done=False
        reward=0.0
        data_overflow=0
        #TO DO
        global S,selected_id,data_prob
        selected_id[0]=action
        for i in range(sensor_node):
            if i != action:
                i_data_overflow,i_reward = calculate_transprob(S,i,False)
                reward+=i_reward
                data_overflow+=i_data_overflow
        return self._get_obs(),reward,done,data_overflow
    def _get_obs(self):
        return S
    def reset(self):
        global S,selected_id,data_prob
        S,selected_id,data_prob=env_init()
        return self._get_obs()

#switch EH or DT Env
class chEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        
        self.lowstate=np.array([0,0,0])
        self.highstate=np.array([battle_level-1,queue_len-1,distance_level-1])
        self.observation_space = spaces.Box(low=self.lowstate,high=self.highstate,dtype=np.uint8)
    def step(self,action):
        done=False
        reward=0.0
        global S,selected_id,data_prob
        data_overflow,reward=calculate_transprob(S,selected_id[0],True,action)

        
        
        return self._get_obs(),reward,done,data_overflow

    def _get_obs(self):
        return S[selected_id[0]]

    def reset(self):
        global S,selected_id,data_prob
        S,selected_id,data_prob=env_init()
        return self._get_obs()




        