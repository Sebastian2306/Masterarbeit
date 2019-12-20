import numpy as np
import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit

grid = []
for k in list(np.round(np.logspace(0, 3))):
    if k not in grid:
        grid.append(k)
        
        
msd = np.loadtxt('msd.dat')
food = np.loadtxt('food.dat')
diff = np.loadtxt('diff.dat')
fdim = np.loadtxt('fdim.dat')       
diff_avg = diff.copy()
for k in range(2,len(diff)-2):
    diff_avg[k] = (diff[k-2] + diff[k-1] + diff[k] + diff[k+1] + diff[k+2])/5.0
        
        
plt.figure(num=1, figsize=(6, 8), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(2,1,1)
plt.loglog(grid,msd,label='$F=5$',lw = 1.5)
plt.loglog(grid,grid,label='$F=0$',lw = 1.5)
plt.title('$10\%\ density\ on\ a\ free\ 100x100\ lattice$')
#plt.xlabel('$steps$', size = 15)
plt.ylabel('$msd$', size = 18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=11)
#plt.subplot(3,1,2)
#plt.plot(grid,food/0.90483934**2)
#plt.xscale('log')
##plt.xlabel('$steps$', size = 15)
#plt.ylabel('$food$', size = 15)
#plt.subplot(4,1,3)
#plt.plot(grid,fdim)
#plt.xscale('log')
#plt.xlabel('$steps$', size = 15)
#plt.ylabel('$fractal\ dim$', size = 15)
plt.subplot(2,1,2)
plt.plot(grid, diff / msd * grid, label='$central\ difference$', lw=1.5)
plt.plot(grid, diff_avg / msd * grid, label='$local\ mean$', lw=1.5)
plt.legend(fontsize=11)
plt.xscale('log')
plt.xlabel('$steps$', size = 18)
plt.ylabel('$diff\ exponent$', size = 18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig('abs1.pdf')