import numpy as np
import random as rand
import matplotlib.pyplot as plt
import sys


def rw(L, N, pot, f): #boxsize, number of walkers, time, food-parameter
    from timeit import default_timer as timer #timer
    start = timer()
    mat = f * np.ones((L,L)) #matrix full of ones, one means position contains food
    pos = np.zeros((N,2)) #x and y coordinate for all N walkers
    
    grid = [] #grid where you save the msds
    for k in list(np.round(np.logspace(0, pot))):
        if k not in grid:
            grid.append(k)
    msd = np.zeros((N, len(grid)))
    fpercentage = np.zeros(len(grid))
    i = 0
    
    for n in range(N): #search iid starting pos for all walkers
        pos[n,:] = np.array([rand.randint(0,L-1), rand.randint(0,L-1)])
    startconf = pos.copy() # save the starting config to calc the msd
    for n in range(N):
        mat[int(pos[n,1]), int(pos[n,0])] = 0 #set entry on which a walker sits to zero, so allways pacman eats all the food
    #print(startconf)
    
    #calc all the prob to move, than move all walkers at the same time, that everyone sees the same food
    prob = np.zeros((N,4)) #up, down, left, right
    for n in range(N):
        prob[n,0] = np.exp(mat[int((pos[n,1] + 1)%L), int(pos[n,0])]) #up
        prob[n,1] = np.exp(mat[int((pos[n,1] - 1)%L), int(pos[n,0])]) #down
        prob[n,2] = np.exp(mat[int(pos[n,1]), int((pos[n,0] + 1)%L)]) #left
        prob[n,3] = np.exp(mat[int(pos[n,1]), int((pos[n,0] - 1)%L)]) #right
        prob[n,:] = prob[n,:] / sum(prob[n,:]) #normalization
        
    #move all walkers and change the food matrix afterwards
    lr_wall = np.zeros(N) #count the moves across the pbc 
    ud_wall = np.zeros(N)
    deltax = np.zeros(N)
    deltay = np.zeros(N)
    for t in range(1, int(10**pot) + 1):
        #calc all the prob to move, than move all walkers at the same time, that everyone sees the same food
        prob = np.zeros((N,4)) #up, down, left, right
        for n in range(N):
            prob[n,0] = np.exp(mat[int((pos[n,1] + 1)%L), int(pos[n,0])]) #up
            prob[n,1] = np.exp(mat[int((pos[n,1] - 1)%L), int(pos[n,0])]) #down
            prob[n,2] = np.exp(mat[int(pos[n,1]), int((pos[n,0] + 1)%L)]) #left
            prob[n,3] = np.exp(mat[int(pos[n,1]), int((pos[n,0] - 1)%L)]) #right
            prob[n,:] = prob[n,:] / sum(prob[n,:]) #normalization
        #print(t, prob)
        

        for n in range(N):
            #move all walkers and change the food matrix afterwards
            rn = rand.random()
            #print(rn)
            if rn < prob[n,0]:
                lr_wall[n] = lr_wall.copy()[n] + ((pos.copy()[n,0] + 1) // L)
                pos[n,0] = (pos[n,0] + 1) % L
            if rn > prob[n,0] and rn < prob[n,0] + prob[n,1]:
                lr_wall[n] = lr_wall.copy()[n] + ((pos.copy()[n,0] - 1) // L)
                pos[n,0] = (pos[n,0] - 1) % L
            if rn > prob[n,0] + prob[n,1] and rn < 1 - prob[n,3]:
                ud_wall[n] = ud_wall.copy()[n] + ((pos.copy()[n,1] + 1) // L)
                pos[n,1] = (pos[n,1] + 1) % L
            if rn > 1 - prob[n,3]:
                ud_wall[n] = ud_wall.copy()[n] + ((pos.copy()[n,1] - 1) // L)
                pos[n,1] = (pos[n,1] - 1) % L
        #print(pos)
        #print('lr:', lr_wall)
        #print('ud:', ud_wall)
        #print('dx:', deltax)
        #print('dy:', deltay)
        #print(msd)
        #now change the matrix
        for n in range(N):
            mat[int(pos[n,1]), int(pos[n,0])] = 0 #set entry on which a walker sits to zero, so allways pacman eats all the food
        #print(mat)
        if t in grid:
            deltax = (pos[:,0] - startconf[:,0]) + lr_wall * L
            deltay = (pos[:,1] - startconf[:,1]) + ud_wall * L          
            msd[:,i] = deltax**2 + deltay**2
            fpercentage[i] = sum(sum(mat)) / (f * L**2)
            i = i + 1
        end = timer()
    print(end - start)
    return msd, fpercentage
            
    
def multiple_walks(L, N, pot, f, n):#n number of walks per matrix
    grid = [] 
    for k in list(np.round(np.logspace(0, pot))):
        if k not in grid:
            grid.append(k)
    msd = np.zeros((N,len(grid)))
    fpercentage = np.zeros(len(grid))
    for i in range(n): #average over the n walks
        msd = msd + rw(L, N, pot, f)[0]
        fpercentage = fpercentage + rw(L, N, pot, f)[1]
    avgmsd = sum(msd) / (N*n)
    fpercentage = fpercentage / n
    return avgmsd, grid, fpercentage            
    
#f = int(sys.argv[1])
    
avgmsd, grid, fpercentage = multiple_walks(10, 10, 4, 10, 100)
plt.subplot(2,1,1)
plt.loglog(grid,avgmsd,label='$F=10$',lw = 0.5)
plt.loglog(grid,grid,label='$reference$',lw = 0.5)
plt.xlabel('$steps$', size = 15)
plt.ylabel('$msd$', size = 15)
plt.legend()
plt.subplot(2,1,2)
plt.plot(grid,fpercentage)
plt.xscale('log')
plt.xlabel('$steps$', size = 15)
plt.ylabel('$food$', size = 15)
plt.savefig('L10test.pdf')
#np.savetxt('crit_food.dat', avgmsd)
