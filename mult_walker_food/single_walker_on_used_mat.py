import numpy as np
import random
import matplotlib.pyplot as plt


def walk(fmat,f,pot): #l=boxlength , f=bias, pot = walkduration
    mat = fmat.copy()
    L = np.size(mat,1)
    lr_wall = 0
    ud_wall = 0
    x = random.randint(0,L-1)
    y = random.randint(0,L-1)
    while mat[x,y] != 0:
        x = random.randint(0,L-1)
        y = random.randint(0,L-1)
    x0 = x
    y0 = y
    mat[x, y] = 0
    grid = []
    for k in list(np.round(np.logspace(0, pot))):
        if k not in grid:
            grid.append(k)
    msd = np.zeros(len(grid))
    i = 0
    for t in range(1, 10**pot + 1):
            prob = np.array([np.exp(mat[(x + 1)%L, y]), np.exp(mat[(x - 1)%L, y]), np.exp(mat[x, (y + 1)%L]), np.exp(mat[x, (y - 1)%L])])
            prob = prob / sum(prob) #normalization
            #print(prob)
            rand = random.random()
            if rand < (prob[0] + prob[1]):
                if rand < prob[0]:
                    lr_wall = lr_wall + (x + 1) // L  
                    x = (x + 1)%L
                else:
                    lr_wall = lr_wall + (x - 1) // L  
                    x = (x - 1)%L
            else:
                if rand < (1 - prob[3]):
                    ud_wall = ud_wall + (y + 1) // L
                    y = (y + 1)%L
                else:
                    ud_wall = ud_wall + (y - 1) // L
                    y = (y - 1)%L
            mat[x, y] = 0         
            if t in grid:
                del_x = x - x0 + lr_wall * L
                del_y = y - y0 + ud_wall * L
                msd[i] = del_x**2 + del_y**2
                i += 1
    return msd
  
def mult_walk(f,pot,n,m): #n=number of walks per mat, m=number of mats
    grid = []
    for k in list(np.round(np.logspace(0, pot))):
        if k not in grid:
            grid.append(k)
    msd = np.zeros(len(grid))
    fmats = np.loadtxt('fmats2.dat').reshape(10,100,100)
    for l in range(m):
        fmat = fmats[l]
        #plt.imshow(fmat)
        for k in range(n):
            msd += walk(fmat,f,pot)
    return msd/(n*m) ,grid

msd, grid = mult_walk(5,2,1000,1)

plt.loglog(grid,msd)
plt.loglog(grid,grid)