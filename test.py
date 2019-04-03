import numpy as np 



a=1

battle_level=3
sensor_node=2
selected_id=0
data_prob=[]

S=[]
def env_init():
    global S,selected_id,data_prob
    B=np.random.randint(1,battle_level,sensor_node).reshape(-1,1)
    D=np.random.randint(0,10,sensor_node).reshape(-1,1)
    H=np.random.randint(0,5,sensor_node).reshape(-1,1)
    
    S=np.hstack((B,D,H))
    selected_id=2
    #S[sensor_node][0]  B  [1] D [2] H

    data_prob=np.ones(sensor_node)*0.1
    #return S,selected_id,data_prob

def geta(a):
    return a
class C():
    def __init__(self):
        pass
    def f(self):
        
        
        c=a+1
        return c
    def b(self):
        #global a
        a=2
    def init(self):
        
        env_init()

myc=C()
# c=myc.f()
# myc.b()
# print(c)
# print(a)
myc.init()
print(S)
print(selected_id)
print(data_prob)