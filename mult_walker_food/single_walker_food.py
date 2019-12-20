import numpy as np
import random
import matplotlib.pyplot as plt


def walk(L,f,pot): #l=boxlength , f=bias, pot = walkduration
    mat= np.zeros((L,L))
    grid = []
    for k in list(np.round(np.logspace(0, pot))):
        if k not in grid:
            grid.append(k)
    msd = np.zeros(len(grid))
    i = 0
    for l in range(int(4*L/10)-1, int(6*L/10)):
        for j in range(int(4*L/10)-1, int(6*L/10)):
            mat[l,j] = f #fill mat with food
    lr_wall = 0
    ud_wall = 0
    x = int(L/2)
    y = x
    x0 = x
    y0 = y
    mat[x, y] = 0
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
  
def mult_walk(L,f,pot,n): #n=number of walks
    grid = []
    for k in list(np.round(np.logspace(0, pot))):
        if k not in grid:
            grid.append(k)
    msd = np.zeros(len(grid))
    for k in range(n):
        msd += walk(L,f,pot)
    msd /= n
    diffmsd = np.zeros((len(grid)))
    for i in range(len(grid) - 1):
        diffmsd[i] = (msd[i+1] - msd[i])/(grid[i+1] - grid[i])
    diffmsd[len(grid) - 1] = diffmsd[len(grid) - 2]
    return grid, msd ,diffmsd

grid, msd, diffmsd = mult_walk(100,5,3,100000)

plt.figure(num=None, figsize=(6, 10), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(2,1,1)
plt.loglog(grid,msd,label='$F=5$',lw = 0.5)
plt.loglog(grid,grid,label='$reference$',lw = 0.5)
plt.title('$single\ walker\ on\ foodsquare$')
#plt.xlabel('$steps$', size = 15)
plt.ylabel('$msd$', size = 15)
plt.legend()
plt.subplot(2,1,2)
plt.plot(grid, diffmsd / msd * grid, label='$forward\ difference$')
#plt.plot(grid, diff_avg / avgmsd * grid, label='$local\ mean$')
plt.legend()
plt.xscale('log')
plt.xlabel('$steps$', size = 15)
plt.ylabel('$diff\ exponent$', size = 15)
plt.savefig('single_walker_on20x20.pdf')
        
    