import numpy as np
import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit

grid = []
for k in list(np.round(np.logspace(0, 4))):
    if k not in grid:
        grid.append(k)
        
        
msd = np.loadtxt('msd10.dat')
food = np.loadtxt('food10.dat')
diff = np.loadtxt('diff10.dat')
corr1 = np.loadtxt('corr1_10.dat').reshape(45,10,10)     
corr2 = np.loadtxt('corr2_10.dat').reshape(45,20,20) 
escp = np.loadtxt('escp10.dat')
diff_avg = diff.copy()
for k in range(2,len(diff)-2):
    diff_avg[k] = (diff[k-2] + diff[k-1] + diff[k] + diff[k+1] + diff[k+2])/5.0
        
        
#plt.figure(num=1, figsize=(6, 10), dpi=80, facecolor='w', edgecolor='k')
#plt.subplot(4,1,1)
#plt.loglog(grid,msd,label='$F=5$',lw = 1.5)
#plt.loglog(grid,grid,label='$F=0$',lw = 1.5)
#plt.title('$10\%\ density\ on\ a\ free\ 100x100\ lattice$')
##plt.xlabel('$steps$', size = 15)
#plt.ylabel('$msd$', size = 18)
#plt.xticks(fontsize=15)
#plt.yticks(fontsize=15)
#plt.legend(fontsize=11)
#plt.subplot(4,1,2)
#plt.plot(grid,food/0.90483934**2)
#plt.xscale('log')
#plt.xlabel('$steps$', size = 15)
#plt.ylabel('$food$', size = 18)
#plt.xticks(fontsize=15)
#plt.yticks(fontsize=15)
#plt.subplot(4,1,3)
#plt.plot(grid,corr1[:,0,0]/corr1[0,0,0])
#plt.xscale('log')
#plt.xlabel('$steps$', size = 18)
#plt.ylabel('$corr[0,0]$', size = 18)
#plt.xticks(fontsize=15)
#plt.yticks(fontsize=15)
#plt.subplot(4,1,4)
#plt.plot(grid, diff / msd * grid, label='$central\ difference$', lw=1.5)
#plt.plot(grid, diff_avg / msd * grid, label='$local\ mean$', lw=1.5)
#plt.legend(fontsize=11)
#plt.xscale('log')
#plt.xlabel('$steps$', size = 18)
#plt.ylabel('$diff\ exponent$', size = 18)
#plt.xticks(fontsize=15)
#plt.yticks(fontsize=15)
##plt.savefig('abs1.pdf')

plt.figure(num=2, dpi=80, facecolor='w', edgecolor='k')
plt.plot(range(10),corr1[7,:,0]/corr1[0,0,0], label='$t=10$')
plt.plot(range(10),corr1[20,:,0]/corr1[0,0,0], label='$t=110$')
plt.plot(range(10),corr1[29,:,0]/corr1[0,0,0], label='$t=596$')
plt.plot(range(10),corr1[32,:,0]/corr1[0,0,0], label='$t=1048$')
plt.plot(range(10),corr1[44,:,0]/corr1[0,0,0], label='$t=10000$')
plt.xlim(0,5)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=11)
plt.ylabel('$corr\ at\ fixed\ time$', size = 18)
plt.xlabel('$\Delta x$', size=18)
plt.title('$10x10\ Box$')

plt.figure(num=3, dpi=80, facecolor='w', edgecolor='k')
plt.plot(range(20),corr2[7,:,0]/corr2[0,0,0], label='$t=10$')
plt.plot(range(20),corr2[20,:,0]/corr2[0,0,0], label='$t=110$')
plt.plot(range(20),corr2[29,:,0]/corr2[0,0,0], label='$t=596$')
plt.plot(range(20),corr2[32,:,0]/corr2[0,0,0], label='$t=1048$')
plt.plot(range(20),corr2[44,:,0]/corr2[0,0,0], label='$t=10000$')
plt.xlim(0,10)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=11)
plt.ylabel('$corr\ at\ fixed\ time$', size = 18)
plt.xlabel('$\Delta x$', size=18)
plt.title('$5x5\ Box$')
plt.savefig('corr5.pdf')

plt.figure(num=4, figsize=(6, 10), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(2,1,1)
plt.title('$10\%\ density\ on\ a\ free\ 100x100\ lattice$')
plt.loglog(grid,msd,label='$F=5$',lw = 1.5)
plt.loglog(grid,grid,label='$F=0$',lw = 1.5)
plt.ylabel('$msd$', size = 18)
plt.legend()
plt.subplot(2,1,2)
plt.plot(grid, escp, label='$g_{0}(t)$')
plt.xscale('log')
plt.legend()
plt.ylabel('$autocorrelation\ of\ steps$', size = 18)
plt.xlabel('$steps$', size = 18)
plt.savefig('autocorr.pdf')