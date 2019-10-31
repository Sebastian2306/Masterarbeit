import numpy as np
import random as rand
import matplotlib.pyplot as plt
from scipy.ndimage import measurements

def perc_finder(mat):
    L = np.size(mat,0)
    labels, num = measurements.label(mat)
    plt.imshow(labels)
    xlink = []
    ylink = []
    perclabels =[]
    for y in range(L):
        if(labels[y,0]>0) and (labels[y,L-1]>0):
            ylink.append(sorted([labels[y,0],labels[y,L-1]]))
            #print(ylink)
            
    for x in range(L):
        if(labels[0,x]>0) and (labels[L-1,x]>0):
            xlink.append(sorted([labels[0,x],labels[L-1,x]]))
            #print(xlink)
            
    for a in xlink:
        if a in ylink:
            for label in a:
                perclabels.append(label)
            
    return perclabels
            

tstmat1 = np.array([[0,0,0,0,1,0],[1,1,0,0,1,1],[0,1,0,0,0,0],[0,1,1,1,1,0],[1,1,0,1,0,0],[0,0,0,1,1,0]])

print(perc_finder(tstmat1))