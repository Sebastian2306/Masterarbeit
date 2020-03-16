import numpy as np
import operator as op
import matplotlib
import random as rand
import scipy.optimize as optimization
import newpercfinder as perc 
import matplotlib.pyplot as plt


def rw(mat, N, startconf, pot, f): #matrix, number of walkers, time, food-parameter
    from timeit import default_timer as timer #timer
    start = timer()
    mat = np.sign(mat)
    foodmat = f * mat #matrix full of ones, one means position contains food
    pos = startconf.copy() #x and y coordinate for all N walkers
    L = np.size(mat,1)
    grid = [] #grid where you save the msds
    for k in list(np.round(np.logspace(0, pot))):
        if k not in grid:
            grid.append(k)
    msd = np.zeros((N, len(grid)))
    fpercentage = np.zeros(len(grid))
    fmats = np.zeros((len(grid), L, L))
    i = 0
    
    
    #calc all the prob to move, than move all walkers at the same time, that everyone sees the same food
    prob = np.zeros((N,4)) #up, down, left, right
    for n in range(N):
        prob[n,0] = np.exp(foodmat[int((pos[n,1] + 1)%L), int(pos[n,0])]) #up
        prob[n,1] = np.exp(foodmat[int((pos[n,1] - 1)%L), int(pos[n,0])]) #down
        prob[n,2] = np.exp(foodmat[int(pos[n,1]), int((pos[n,0] + 1)%L)]) #left
        prob[n,3] = np.exp(foodmat[int(pos[n,1]), int((pos[n,0] - 1)%L)]) #right
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
            prob[n,0] = np.exp(foodmat[int((pos[n,1] + 1)%L), int(pos[n,0])]) #up
            prob[n,1] = np.exp(foodmat[int((pos[n,1] - 1)%L), int(pos[n,0])]) #down
            prob[n,2] = np.exp(foodmat[int(pos[n,1]), int((pos[n,0] + 1)%L)]) #left
            prob[n,3] = np.exp(foodmat[int(pos[n,1]), int((pos[n,0] - 1)%L)]) #right
            prob[n,:] = prob[n,:] / sum(prob[n,:]) #normalization
        #print(t, prob)
        

        for n in range(N):
            #move all walkers and change the food matrix afterwards
            rn = rand.random()
            #print(rn)
            if rn < prob[n,0]:
                if mat[int((pos[n,0] + 1)%L), int(pos[n,1])] != 0:
                    lr_wall[n] = lr_wall.copy()[n] + ((pos.copy()[n,0] + 1) // L)
                    pos[n,0] = (pos[n,0] + 1) % L
            if rn > prob[n,0] and rn < prob[n,0] + prob[n,1]:
                if mat[int((pos[n,0] - 1)%L), int(pos[n,1])] != 0:
                    lr_wall[n] = lr_wall.copy()[n] + ((pos.copy()[n,0] - 1) // L)
                    pos[n,0] = (pos[n,0] - 1) % L
            if rn > prob[n,0] + prob[n,1] and rn < 1 - prob[n,3]:
                if mat[int(pos[n,0]), int((pos[n,1] + 1)%L)] != 0:
                    ud_wall[n] = ud_wall.copy()[n] + ((pos.copy()[n,1] + 1) // L)
                    pos[n,1] = (pos[n,1] + 1) % L
            if rn > 1 - prob[n,3]:
                if mat[int(pos[n,0]), int((pos[n,1] - 1)%L)] != 0:
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
            foodmat[int(pos[n,1]), int(pos[n,0])] = 0 #set entry on which a walker sits to zero, so allways pacman eats all the food
        #print(mat)
        if t in grid:
            deltax = (pos[:,0] - startconf[:,0]) + lr_wall * L
            deltay = (pos[:,1] - startconf[:,1]) + ud_wall * L          
            msd[:,i] = deltax**2 + deltay**2
            fpercentage[i] = sum(sum(foodmat)) / (f * sum(sum(mat)))
            fmats[i] = foodmat
            i = i + 1
        end = timer()
    print(end - start)
    return msd, fpercentage



def perc_start(mat, perclabel):
    L = np.size(mat,0)
    x = np.random.choice(L)
    y = np.random.choice(L)
    if mat[x,y] != perclabel:
        x, y = perc_start(mat, perclabel)
    return x, y

def genstart(mat, perclabel, N):
    startconf = np.zeros((N,2))
    for i in range(N):
        startconf[i,:] = perc_start(mat, perclabel)
    return startconf

def multiple_walks(L,p,dens,pot,f,n,m): #boxsize, p value, density of pacmans, length of the walk, food parameter, number of walks, number of matrices
    grid = [] 
    for k in list(np.round(np.logspace(0, pot))):
        if k not in grid:
            grid.append(k)
    
    fpercentage = np.zeros(len(grid))
    avgmsd = np.zeros(len(grid))
    for k in range(m): 
        mat, perclabel, clustersize = perc.generate_percolating_matrix(L,p)
        N = int(dens*clustersize[perclabel])
        if(N == 0):
            N = 1
        msd = np.zeros((N,len(grid)))
        print(N)
        for i in range(n): #average over the n walks
            startconf = genstart(mat, perclabel, N)
            h0, h1 = rw(mat, N, startconf, pot, f)
            msd = msd + h0
            fpercentage = fpercentage + h1
        avgmsd += np.sum(msd, axis=0) /(N*n)
    avgmsd /= m
    fpercentage = fpercentage / (n*m)
    diffmsd = np.zeros((len(grid)))
    for i in range(len(grid) - 1):
        diffmsd[i] = (avgmsd[i+1] - avgmsd[i])/(grid[i+1] - grid[i])
    diffmsd[len(grid) - 1] = diffmsd[len(grid) - 2]
    
    return avgmsd, grid, fpercentage, diffmsd


msd, grid, fpercentage, diff = multiple_walks(100, 0.592746, 0.1, 4, 5, 1, 1)
print(msd)
diff_avg = diff.copy()   
for k in range(2,len(diff)-2):
    diff_avg[k] = (diff[k-2] + diff[k-1] + diff[k] + diff[k+1] + diff[k+2])/5.0

#np.savetxt('msd10F5.dat', msd)
#np.savetxt('food10F5.dat', fpercentage)
#np.savetxt('diff10F5.dat', diff)
plt.figure(num=None, figsize=(6, 10), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(3,1,1)
plt.loglog(grid,msd, label='$simu$')
plt.ylabel('$msd$', size = 15)
plt.loglog(grid,0.6*np.array(grid)**(0.7),label='$reference$')
plt.legend()
plt.subplot(3,1,2)
plt.plot(grid,fpercentage)
plt.xscale('log')
plt.ylabel('$food$', size = 15)
plt.subplot(3,1,3)
plt.plot(grid, diff / msd * grid, label='$central\ difference$')
plt.plot(grid, diff_avg / msd * grid, label='$local\ mean$')
plt.legend()
plt.xscale('log')
plt.xlabel('$steps$', size = 15)
plt.ylabel('$diff\ exponent$', size = 15)



















