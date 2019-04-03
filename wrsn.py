import numpy as np
import gym
from gym import spaces

#define constant value

queue_len=20
battle_level=40
distance_level=5
sensor_node=10

selected_id=0
data_prob=[]
S=[]

def env_init():
    global S,selected_id,data_prob
    B=np.random.randint(1,battle_level,sensor_node).reshape(-1,1)
    D=np.random.randint(0,10,sensor_node).reshape(-1,1)
    H=np.random.randint(0,5,sensor_node).reshape(-1,1)
    S=np.hstack((B,D,H))
    selected_id=0
    #S[sensor_node][0]  B  [1] D [2] H

    data_prob=np.ones(sensor_node)*0.1
    
def funEh(distance):
    Eh=[5,5,4,4,3]
    return Eh[distance]

def funPh(distance):
    Ph=[1,1,2,2,3]
    return Ph[distance]    


def calculate_transprob(sensor_id,selected,switch=None):
    global S
    if selected==True:
        #BH
        if (switch!=0 and switch!=1):
            raise Exception("Invalid switch",switch)
        if switch==0:
            S[sensor_id][0]=min(S[sensor_id][0]+funEh(S[sensor_id][2]),battle_level)
        else:
            if funPh(S[sensor_id][2])<S[sensor_id][0]:
                S[sensor_id][0]=S[sensor_id][0]-funPh(S[sensor_id][2])
                S[sensor_id][1]=max(S[sensor_id][1]-1,0)
            S[sensor_id][1]=S[sensor_id][1]+1 if data_prob[sensor_id]>np.random.uniform(0,1) else S[sensor_id][1]
            if S[sensor_id][1]>queue_len:
                S[sensor_id][1]=queue_len
                data_overflow=-1
            else:
                data_overflow=0
   
    else:
        S[sensor_id][1]=S[sensor_id][1]+1 if data_prob[sensor_id]>np.random.uniform(0,1) else S[sensor_id][1]
        if S[sensor_id][1]>queue_len:
            S[sensor_id][1]=queue_len
            data_overflow=-1
        else:
            data_overflow=0

    return data_overflow




#Sensor Node select Env
class nsEnv(gym.Env):
    def __init__(self,sensor_node):
        self.action_space = spaces.Discrete(sensor_node)

        self.lowstate=np.array([0,0,0])
        self.highstate=np.array([battle_level-1,queue_len-1,distance_level-1])

        self.observation_space = spaces.Box(low=self.lowstate,high=self.highstate,shape=(sensor_node,3),dtype=np.uint8)


    def step(self,action):
        done=False
        reward=0.0
        #TO DO
        
        for i in range(sensor_node):
            if i != action:
                reward += calculate_transprob(i,False)
        for i in range(sensor_node):
            if S[i][0]==0:
                done = True
                break
    
        return self._get_obs(),reward,done,{}
    def _get_obs(self):
        return (S)
    def reset(self):
        
        env_init()
        return self._get_obs()

#switch EH or DT Env
class chEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        
        self.lowstate=np.array([0,0,0])
        self.highstate=np.array([battle_level-1,queue_len-1,distance_level-1])
        self.observation_space = spaces.Box(low=self.lowstate,high=self.highstate,shape=(1,3),dtype=np.uint8)
    def step(self,action):
        done=False
        reward=0.0
        
        reward=calculate_transprob(selected_id,True,action)

        if S[selected_id][0] ==0:
            done = True
        return self._get_obs,reward,done,{}

    def _get_obs(self):
        return (S[selected_id])

    def reset(self):
        env_init()
        return self._get_obs()




        