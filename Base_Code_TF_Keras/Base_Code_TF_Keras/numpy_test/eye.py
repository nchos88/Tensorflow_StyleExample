import numpy as np

x = [ 3 , 4 , 1]

x = np.asarray( x )

eye = np.eye(5)

print(eye)

x = x.reshape(-1)
print(x)

res = eye[x]

print(res)




