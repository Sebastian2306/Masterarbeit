import numpy as np
import random as rand
import matplotlib.pyplot as plt

samples = 10000
N = 5000
L = 100
density = 0
for k in range(samples):
    mat = np.ones((L,L))
    pos = np.zeros((N,2))
    for n in range(N): #search iid starting pos for all walkers
        pos[n,:] = np.array([rand.randint(0,L-1), rand.randint(0,L-1)])
        startconf = pos.copy() # save the starting config to calc the msd
    for n in range(N):
        mat[int(pos[n,1]), int(pos[n,0])] = 0 #set entry on which a walker sits to zero, so allways pacman eats all the food
    density += sum(sum(mat))
    #plt.imshow(mat)   
density *= 1.0/samples
print(density/(L*L))