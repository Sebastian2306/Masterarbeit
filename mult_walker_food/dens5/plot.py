import numpy as np
import matplotlib.pyplot as plt
import random as rand
#from scipy.optimize import curve_fit

grid = []
for k in list(np.round(np.logspace(0, 3))):
    if k not in grid:
        grid.append(k)
     
        
msd = np.loadtxt('msd1000F5.dat')
food = np.loadtxt('food1000F5.dat')
diff = np.loadtxt('diff1000F5.dat')       
diff_avg = diff.copy()
for k in range(2,len(diff)-2):
    diff_avg[k] = (diff[k-2] + diff[k-1] + diff[k] + diff[k+1] + diff[k+2])/5.0      
        
plt.figure(num=1, figsize=(6, 10), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(3,1,1)
plt.loglog(grid,msd,label='$F=5$',lw = 0.5)
plt.loglog(grid,grid,label='$reference$',lw = 0.5)
plt.title('$1000\ Walker\ on\ percolating\ Cluster\ in\ 100x100\ matrix$')
#plt.xlabel('$steps$', size = 15)
plt.ylabel('$msd$', size = 15)
plt.legend()
plt.subplot(3,1,2)
plt.plot(grid,food/0.95123114**2)
plt.xscale('log')
#plt.xlabel('$steps$', size = 15)
plt.ylabel('$food$', size = 15)
plt.subplot(3,1,3)
plt.plot(grid, diff / msd * grid, label='$central\ difference$')
plt.plot(grid, diff_avg / msd * grid, label='$local\ mean$')
plt.legend()
plt.xscale('log')
plt.xlabel('$steps$', size = 15)
plt.ylabel('$diff\ exponent$', size = 15)
#plt.savefig('.pdf')