import json
import numpy as np
from collections import defaultdict
import tensorflow as tf
# with open("nsQ_json.json","r") as json_file:
#     nsQ=json.load(json_file)
# json_file.close()



# for k,v in list(nsQ.items()):
#     print(k)
#     print(v)
#     l=[int(k[x]) for x in range(1,len(k)-1,3)]
#     #v=[float(v[x]) for x in range(1,len(v)-1,3)]
#     nsQ[tuple(np.array(l))] = np.array(v)
#     del nsQ[k]
#     break


# for key in list(nsQ.keys()):
#     print(key)
#     print(nsQ[key])
#     break
a=np.array([1,5,5,5])
d=np.where(a==np.max(a))
b=np.array([0,1,2,0])
c=np.max(b[d])
s=np.where(b==c)[0][0]
print(s)



#print(c)
# c=([2,1])*2
# print(c)
# b= a/c
# print(b)
# print(c[2:])


# H=([0,1,2])*4
# H=np.array(H,dtype=np.uint8).reshape(-1,1)
# print(H)
# print(type(H))