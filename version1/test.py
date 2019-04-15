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
a=np.array([[2,1,3,4],[2,3,4,1]])
c=([2,1])*2
print(c)
b= a/c
print(b)
print(c[2:])