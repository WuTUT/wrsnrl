import numpy as np
import gym
from gym import spaces

#define constant value

queue_len=4
battle_level=4
distance_level=1
sensor_node=4


def env_init():
    
    B=np.random.randint(0,battle_level+1,sensor_node).reshape(-1,1)
    D=np.random.randint(0,queue_len+1,sensor_node).reshape(-1,1)
    H=np.random.randint(0,distance_level,sensor_node).reshape(-1,1)
    S=np.hstack((B,D,H))
    
    #S[sensor_node][0]  B  [1] D [2] H

    data_prob=np.ones(sensor_node)*0.15
    return S,data_prob
def env_init_test():
    B=np.ones(sensor_node,dtype=np.uint8).reshape(-1,1)*battle_level
    D=np.zeros(sensor_node,dtype=np.uint8).reshape(-1,1)
    H=np.zeros(sensor_node,dtype=np.uint8).reshape(-1,1)
    S=np.hstack((B,D,H))
    data_prob=np.ones(sensor_node)*0.1
    return S,data_prob
def funEh(distance):
    Eh=[1,1]
    return Eh[distance]

def funPh(distance):
    Ph=[1,1]
    return Ph[distance]    

def calculate_transprob(S,action,data_prob):
    data_overflow=0
    reward=0
    sensor_id=action//2
    switch=action%2
    for i in range(sensor_node):
        if data_prob[i]>np.random.uniform(0,1):
            S[i][1]=S[i][1]+1 
        if S[i][1]>queue_len:
            S[i][1]=queue_len
            data_overflow+=1
            reward=-1
            
    assert switch==0 or switch ==1
    if switch==0:
        if S[sensor_id][0]+funEh(S[sensor_id][2]) <= battle_level:
            S[sensor_id][0]=S[sensor_id][0]+funEh(S[sensor_id][2])
            reward+=1
        else:
            S[sensor_id][0]=battle_level
    else:
        if funPh(S[sensor_id][2])<=S[sensor_id][0]:
            S[sensor_id][0]=S[sensor_id][0]-funPh(S[sensor_id][2])
            if S[sensor_id][1]==0:
                reward-=1
            else:
                reward+=1
                S[sensor_id][1]-=1
        else:
            reward-=1
            pass
    
            
    return S,reward,data_overflow


    

class wrsn(gym.Env):
    def __init__(self,sensor_node):
        self.action_space = spaces.Discrete(sensor_node*2)
        self.lowstate=np.tile(np.array([0,0,0]),(sensor_node,1))
        self.highstate=np.tile(np.array([battle_level-1,queue_len-1,distance_level-1]),(sensor_node,1))
        
        self.observation_space = spaces.Box(low=self.lowstate,high=self.highstate,dtype=np.uint8)
        self.S,self.data_prob=env_init()
    def step(self,action):
        done=False
        reward=0.0
        self.S,reward,data_overflow=calculate_transprob(self.S,action,self.data_prob)
        #if data_overflow
        return self._get_obs(),reward,done,data_overflow
    def _get_obs(self):
        return tuple(self.S.flatten())
    def reset(self):
        self.S,self.data_prob=env_init()
        return self._get_obs()    
    def reset_test(self):
        self.S,self.data_prob=env_init_test()
        return self._get_obs() 

if __name__ == '__main__':
    env=wrsn(sensor_node)
    state=env.reset_test()
    
    for t in range(15):
        action=np.random.randint(0,2,dtype=np.uint8)
        next_state, reward, done, data_overflow = env.step(action)
        print("step {} state{} action {} reward {} next_state{}".format(t,state,action,reward,next_state),end="\n")
        state = next_state
        
            

