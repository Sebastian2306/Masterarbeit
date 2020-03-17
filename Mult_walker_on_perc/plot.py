import numpy as np
import matplotlib.pyplot as plt
import random as rand
#from scipy.optimize import curve_fit

ref = np.loadtxt('ref.dat')
refgrid = np.loadtxt('refgrid.dat')
refsingle = np.loadtxt('singleF5.dat')

grid = []
for k in list(np.round(np.logspace(0, 4))):
    if k not in grid:
        grid.append(k)

gridsingle = []
for k in list(np.round(np.logspace(0, 4))):
    if k not in gridsingle:
        gridsingle.append(k)     
        
msd = np.loadtxt('msd10F5.dat')
#msd2 = np.loadtxt('msd10F5.dat')
food = np.loadtxt('food10F5.dat')
diff = np.loadtxt('diff10F5.dat')       
diff_avg = diff.copy()
for k in range(2,len(diff)-2):
    diff_avg[k] = (diff[k-2] + diff[k-1] + diff[k] + diff[k+1] + diff[k+2])/5.0      
        
plt.figure(num=1, figsize=(6, 8), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(3,1,1)
plt.loglog(grid,msd,label='$F=5$',lw = 1.5)
plt.loglog(refgrid,ref,label='$F=0$',lw = 1.5)
plt.loglog(gridsingle, refsingle, label='$single\ Pacman\ with\ F=5$', lw=1.5)
plt.title('$10\% Walker\ on\ percolating\ Cluster\ in\ 100x100\ matrix$')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#plt.xlabel('$steps$', size = 15)
plt.ylabel('$msd$', size = 18)
plt.legend(fontsize=11)
plt.subplot(3,1,3)
plt.plot(grid,food)
plt.xscale('log')
#plt.xlabel('$steps$', size = 15)
plt.ylabel('$food$', size = 15)
plt.subplot(3,1,2)
plt.plot(grid, diff / msd * grid, label='$central\ difference$', lw=1.5)
#plt.plot(grid, diff_avg / msd * grid, label='$local\ mean$', lw=1.5)
plt.legend(fontsize=11)
plt.xscale('log')
plt.xlabel('$steps$', size = 18)
plt.ylabel('$diff\ exponent$', size = 18)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#plt.savefig('abs2.pdf')