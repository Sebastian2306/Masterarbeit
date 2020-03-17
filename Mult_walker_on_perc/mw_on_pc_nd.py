import numpy as np
import operator as op
import matplotlib
import scipy.optimize as optimization
import matplotlib.pyplot as plt
#import newpercfinder as perc
from scipy.ndimage import measurements
from pylab import *

pc = 0.592746




def gen_mat(L,p):
    mat = np.vectorize(lambda x: np.int(x<p))(np.random.random ((L,L)))
    return mat


def perc_finder(mat):
    L = np.size(mat,0)
    labels, num = measurements.label(mat)
    perc_clusters = []
    help_list =[]
    sizes = []
    #check if a label appears on both boundaries, if so, you found a perc cluster
    for x in range(L):
        if labels[0,x] != 0:
            if labels[0,x] == labels[L-1,x]:
                help_list.append(labels[0,x])
                
    for y in range(L):
        if labels[y,0] != 0:
            if labels[y,0] == labels[y,L-1]:
                help_list.append(labels[y,0])
                
    for i in help_list:
        if i not in perc_clusters:
            perc_clusters.append(i)
    perc_clusters = sort(perc_clusters)
    sorted_labels = list(sort(list(labels.flatten())))
    #count how big the perc clusters are    
    for n in perc_clusters:
        if n != 0:
            sizes.append(sorted_labels.count(n))
    #choose biggest perc cluster
    perc = len(perc_clusters)
    perclabel = 0
    clustersize = 0
    if perc > 0:
        biggest_ind = argmax(sizes)
        perclabel = perc_clusters[biggest_ind]
        clustersize = sizes[biggest_ind]          
       
    return perc, perclabel, clustersize
           
def gen_perc_mat(L, p, rejected = 0):
    mat = gen_mat(L,p)
    perc, perclabel, clustersize = perc_finder(mat)
    labels, num = measurements.label(mat)
    if perc > 0:
        #print(mat)
        print('Matrix gefunden')
        #print('Number of rejected matrices', rejected)        
        return labels, perclabel, clustersize
    else:
        return gen_perc_mat(L, p)


def rw(mat, perclabel, clustersize, N, startconf, pot, f): #matrix, number of walkers, time, food-parameter
    from timeit import default_timer as timer #timer
    start = timer()
    L = np.size(mat,1)
    foodmat = np.zeros_like(mat.copy())
    for i in range(L):
        for j in range(L):
            if mat[i,j] == perclabel:
                foodmat[i,j] = f #fill matrix with food on the perc cluster
    #print(foodmat)
    pos = startconf.copy() #x and y coordinate for all N walkers
    for n in range(N):
            foodmat[int(pos[n,1]), int(pos[n,0])] = 0
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
       # print(t, prob)
        

        for n in range(N):
            #move all walkers and change the food matrix afterwards
            rn = random()
            #print(rn)
            if rn < prob[n,0]:
                if mat[int((pos[n,1] + 1)%L), int(pos[n,0])] != 0:
                    lr_wall[n] = lr_wall.copy()[n] + ((pos.copy()[n,1] + 1) // L)
                    pos[n,1] = (pos.copy()[n,1] + 1) % L

            if rn > prob[n,0] and rn < prob[n,0] + prob[n,1]:
                if mat[int((pos[n,1] - 1)%L), int(pos[n,0])] != 0:
                    lr_wall[n] = lr_wall.copy()[n] + ((pos.copy()[n,1] - 1) // L)
                    pos[n,1] = (pos.copy()[n,1] - 1) % L

            if rn > prob[n,0] + prob[n,1] and rn < 1 - prob[n,3]:
                if mat[int(pos[n,1]), int((pos[n,0] + 1)%L)] != 0:
                    ud_wall[n] = ud_wall.copy()[n] + ((pos.copy()[n,0] + 1) // L)
                    pos[n,0] = (pos.copy()[n,0] + 1) % L

            if rn > 1 - prob[n,3]:
                if mat[int(pos[n,1]), int((pos[n,0] - 1)%L)] != 0:
                    ud_wall[n] = ud_wall.copy()[n] + ((pos.copy()[n,0] - 1) // L)
                    pos[n,0] = (pos.copy()[n,0] - 1) % L

#        print(pos)
#        print('lr:', lr_wall)
#        print('ud:', ud_wall)
#        print('dx:', deltax)
#        print('dy:', deltay)
#        print(msd)
        #now change the matrix
        for n in range(N):
            foodmat[int(pos[n,1]), int(pos[n,0])] = 0 #set entry on which a walker sits to zero, so allways pacman eats all the food
        #print(foodmat)
        if t in grid:
            #print(pos)
            deltax = (pos[:,1] - startconf[:,1]) + lr_wall * L
            deltay = (pos[:,0] - startconf[:,0]) + ud_wall * L          
            msd[:,i] = deltax**2 + deltay**2
            fpercentage[i] = sum(sum(foodmat)) / (f * clustersize)
            fmats[i] = foodmat
            i = i + 1
        end = timer()
    print(end - start)
    return msd, fpercentage



def perc_start(mat, perclabel):
    #print("perclabel: ", perclabel)
    #print("drin")
    L = np.size(mat,0)
    x = np.random.choice(L)
    y = np.random.choice(L)
    while(mat[y,x] != perclabel):
        x = np.random.choice(L)
        y = np.random.choice(L)
    return int(x), int(y)

def genstart(mat, perclabel, N):
    startconf = np.zeros((N,2), dtype=int)
    for i in range(N):
        startconf[i,:] = perc_start(mat, perclabel)
    return startconf

def multiple_walks(L,p,d,pot,f,n,m): #boxsize, p value, density of pacmans, length of the walk, food parameter, number of walks, number of matrices
    grid = [] 
    for k in list(np.round(np.logspace(0, pot))):
        if k not in grid:
            grid.append(k)
    fpercentage = np.zeros(len(grid))
    for k in range(m): 
        mat, perclabel, clustersize = gen_perc_mat(L,p)
        #print("Matrix: ",mat)
        #print('perclabel: ',perclabel)
        N = int(round(d*clustersize))
        if(N == 0):
            N = 1
        print("Anzahl der Pacmans: ", N)
        avgmsd = np.zeros(len(grid))
        msd = np.zeros((N,len(grid)))
        for i in range(n): #average over the n walks
            startconf = genstart(mat, perclabel, N)    
            #print('start: ',startconf)
            #print("Label ",mat[startconf[0,:][1], startconf[0,:][0]])
            h0, h1 = rw(mat, perclabel, clustersize, N, startconf, pot, f)
            msd = msd + h0
            #print(msd)
            fpercentage = fpercentage + h1
        for j in range(N):
            avgmsd += msd[j]
        avgmsd = avgmsd / (N*n)
        #print("avgmsd: ", avgmsd)
    fpercentage = fpercentage / (n*m)
    diffmsd = np.zeros((len(grid)))
    for i in range(len(grid) - 1):
        diffmsd[i] = (avgmsd[i+1] - avgmsd[i])/(grid[i+1] - grid[i])
    diffmsd[len(grid) - 1] = diffmsd[len(grid) - 2]
    
    return avgmsd, grid, fpercentage, diffmsd


msd, grid, fpercentage, diff = multiple_walks(100, pc, 0.01, 4, 5, 100, 10)
print(msd)
diff_avg = diff.copy()   
for k in range(2,len(diff)-2):
    diff_avg[k] = (diff[k-2] + diff[k-1] + diff[k] + diff[k+1] + diff[k+2])/5.0

#np.savetxt('msd10F5.dat', msd)
#np.savetxt('food10F5.dat', fpercentage)
#np.savetxt('diff10F5.dat', diff)
#plt.figure(num=None, figsize=(6, 10), dpi=80, facecolor='w', edgecolor='k')
refsingle = np.loadtxt('singleF5.dat')
gridsingle = []
for k in list(np.round(np.logspace(0, 4))):
    if k not in gridsingle:
        gridsingle.append(k)    
        
plt.figure(num=None, figsize=(6, 10), dpi=80, facecolor='w', edgecolor='k')    
plt.subplot(3,1,1)
plt.loglog(grid,msd,label='$F=5$',lw = 1)
plt.loglog(grid,np.array(grid)**(0.7),label='$F=0\ (t^{0.7})$',lw = 1)
plt.loglog(gridsingle[:31], refsingle[:31], label='$single\ Pacman\ F=5$', lw=1)
plt.ylabel('$msd$', size = 15)
plt.legend()
plt.subplot(3,1,2)
plt.plot(grid,fpercentage)
plt.xscale('log')
plt.ylabel('$food$', size = 15)
plt.subplot(3,1,3)
plt.plot(grid, diff / msd * grid, label='$forward\ difference$')
#plt.plot(grid, diff_avg / msd * grid, label='$local\ mean$')
plt.legend()
plt.xscale('log')
plt.xlabel('$steps$', size = 15)
plt.ylabel('$diff\ exponent$', size = 15)
plt.savefig('1percent_on_pc.pdf')




















