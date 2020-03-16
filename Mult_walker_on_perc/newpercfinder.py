import numpy as np
#import operator as op
#import matplotlib
import random
#from scipy.ndimage.interpolation import shift
#import scipy.optimize as optimization

# This code works for a 2D square lattice.
# Most parts should easily extend to 3D, if one adapts the list of
# directions below. This applies in particular to the generation of
# the random walks (but is untested).
# The algorithm to detect clusters will also extend in principle, but
# the extension will require extra coding.


debug_matrix=0

##
## helpers
##

# Quick-and-dirty implementation of periodic boundary conditions.
def pbcwrap (i, w):
  return (i % w)
##
## generation of random matrix: methods
##

# generate_matrix ((w,h),p):
# This generates a random matrix of width w and height h, where
# sites are marked with probability p -- those will be the sites the
# random walker can access.
# A cluster counting algorihm is employed, and a percolating
# cluster will be identified (assuming p.b.c. in both directions).
# Return value: a tuple of the form (matrix,label,clustersizes), where
# - matrix = the random matrix
#   Entries are: 0 = unmarked site (inaccessible)
#                >0 = marked, unique label per cluster
# - label = The label identifying the largest percolating cluster.
#   If no percolating cluster is found, this return value is zero.
# - clustersizes = array of cluster sizes.
def generate_matrix (L,p):
  # Generation of the random matrix (zeros and ones):
  # This is code one is likely to understand: create a matrix of zeros,
  # walk through the array, put in a 1 if a drawn random number is less than p.
  # Instead of two for loops, using an iterator is a python way of doing it:
  #mat = np.zeros ((w,h),dtype=np.int)
  #for x in np.nditer(mat, op_flags=['readwrite']):
  #  if (np.random.random() < p):
  #    x[...] = 1
  # This is the even more python-ish and supposedly significantly faster
  # (not that it matters here):
  print("random matrix ...")
  mat = np.vectorize(lambda x: np.int(x<p))(np.random.random ((L,L)))
  tstperc = None
  if (debug_matrix):
    tstmat = None
    # for debugging, I found the following to be nice matrix test cases
    if (debug_matrix==1):
      # debug_matrix 1:
      # it contains a single cluster that crosses all periodic boundaries,
      # yet does not percolate
      # if you close the gap in the middle, it will percolate
      tstmat = [[0,1,0,1,0,0,0],
                [1,1,1,0,0,0,1],
                [0,0,1,1,0,0,1],
                [0,0,0,0,1,1,1],
                [0,0,0,1,1,0,0],
                [0,1,0,1,0,0,0]]
      tstperc = False
    if (debug_matrix==2):
      tstmat = [[0,1,0,1,0,0,0],
                [1,1,1,0,0,0,1],
                [0,0,1,1,0,0,1],
                [0,0,0,1,1,1,1],
                [0,0,0,1,1,0,0],
                [0,1,0,1,0,0,0]]
      tstperc = True
    if (debug_matrix==3):
      # this test matrix contains a single spanning cluster that also
      # crosses all four boundaries, but still does not percolate
      tstmat = [[0,0,0,1,0,1,0,0],
                [0,0,0,0,0,1,0,0],
                [1,0,0,0,0,1,1,1],
                [0,0,0,1,1,1,0,0],
                [1,1,1,1,0,0,0,1],
                [0,0,0,1,0,0,0,0],
                [0,0,0,1,0,1,0,0]]
      tstperc = False
    if (debug_matrix==4):
      # this is a very simple matrix containing two percolating clusters
      tstmat = [[0,0,1,0,1,1],
                [0,0,1,0,1,1],
                [0,1,1,0,1,0],
                [0,1,0,0,1,1],
                [0,1,1,0,0,1]]
      tstperc = True
    if (debug_matrix==5):
      # first test matrix discussed with Sebastian Steinhaeuser Oct 2019
      # this percolates diagonally across boundaries
      tstmat = [[0,0,0,0,1,0],
                [1,1,0,0,1,1],
                [0,1,0,0,0,0],
                [0,1,1,1,1,0],
                [1,1,0,1,0,0],
                [0,0,0,1,1,0]]
      tstperc = True
    if (debug_matrix==6):
      # second test matrix discussed with Sebastian Steinhaeuser Oct 2019
      # does not percolate
      tstmat = [[0,0,0,1,0,1,0,0],
                [0,0,0,0,0,1,0,0],
                [1,0,0,0,0,1,1,1],
                [0,0,0,1,1,1,0,0],
                [1,1,1,1,0,0,0,1],
                [0,0,0,1,0,0,0,0],
                [0,0,0,1,0,1,0,0]]
      tstperc = False
    if (debug_matrix==7):
      # third test matrix discussed with Sebastian Steinhaeuser Oct 2019
      # percolates
      tstmat = [[1,0,0,0,0,1],
                [1,1,1,0,0,0],
                [0,0,1,1,0,0],
                [1,0,0,1,1,1],
                [1,1,1,0,0,0],
                [0,0,1,1,1,1]]
      tstperc = True
    if (debug_matrix==8):
      # does not percolate!
      tstmat = [[1,1,0,0,1,1],
                [1,1,0,0,1,1],
                [0,0,0,0,0,0],
                [0,0,0,0,0,0],
                [1,1,0,0,1,1],
                [1,1,0,0,1,1]]
      tstperc = False
    if (debug_matrix==9):
      # does not percolate!
      tstmat = [[0,1,0,0,1,0],
                [0,1,1,1,1,0],
                [0,0,0,0,0,0],
                [0,0,0,0,0,0],
                [0,1,1,1,1,0],
                [0,1,0,0,1,0]]
      tstperc = False
    if (debug_matrix==10):
      # does percolate!
      tstmat = [[0,1,0,0,1,0],
                [0,1,1,1,1,0],
                [0,0,0,0,0,0],
                [0,0,0,0,0,0],
                [1,1,0,0,1,1],
                [0,1,0,0,1,0]]
      tstperc = True
    # select either one for debugging
    if not tstmat:
      raise SystemExit
    mat = np.zeros((L,L))
    mat = np.transpose(np.vectorize(lambda x: x)(tstmat))
    w,h = mat.shape[:]
  num_occupied = sum(mat.reshape(-1))
  print(num_occupied, "occupied sites (",1.*num_occupied/(L*L),")")
  diag = 1 #to check for diagonal percolation
  # We now have a matrix of 0's and 1's (1 = blocked site).
  # Next steps: identify clusters, find percolating cluster.
  # There is a Wolfram Demonstrations project "percolation on a square grid"
  # that uses a deceptively simple algorithm, which may or may not work;
  # it is however recursive, and blows the recursion limit on large matrices.
  # Here's a proper Hoshen-Kopelman algorithm.
  # [https://www.ocf.berkeley.edu/~fricke/projects/hoshenkopelman/hoshenkopelman.html]
  # Idea: walk through the grid and check for every occupied site,
  # whether the site to the left or above is also occupied;
  # if so, mark them with the same label; if both left and above are
  # occupied, mark the two (likely different) labels as belonging to
  # the same equivalence class.
  # To setup equivalence classes, we need a linked list,
  # here implemented through a mapping array called labels. This needs
  # to be big enough to hold all possible labels -- their number is unknown
  # and the only strict upper bound is the number of lattice sites...
  # This is (temporarily) nasty on memory, and could be improved
  # (allocate memory in smaller blocks, extending if needed); perhaps there
  # is a python way of doing this.
  labels = list(range(L**2))
  # Helper to return a representative of an equivalence class:
  # works since we initialize the array labels to be an identity mapping
  # (at the beginning, each label is in its own equivalence class, of
  # which it is the representative)
  def same_as (l):
    while (labels[l]) != l:
      l = labels[l]
    return l
  # Helper to mark two labels as belonging to the same equivalence class:
  # maintains the property labels[l] = l for the chosen representative l
  def link (label1,label2):
    labels[same_as(label1)] = same_as(label2)
  # main part of the H.-K. algorithm:
  # Instead of keeping a separate array for the labels, we use the
  # convention 0: unoccupied, 1: occupied but not processed,
  # >=2: cluster label
  label = 1
  for i in range(L):
    for j in range(L):
      if mat[i,j]>0:
        left = mat[i-1,j] if (i>0) else 0
        #left = mat[pbcwrap(i-1,w),j]
        above = mat[i,j-1] if (j>0) else 0
        #above = mat[i,pbcwrap(j-1,h)]
        if (left<=1 and above<=1):
          # site belongs to a new cluster
          label+=1
          mat[i,j] = label
        elif (left>1 and above<=1):
          # same cluster as left
          mat[i,j] = same_as(left)
        elif (left<=1 and above>1):
          # same cluster as above
          mat[i,j] = same_as(above)
        else:
          link(left,above)
          mat[i,j] = same_as(left)
  def printmat (mat):
    mat2 = np.copy(mat)
    for i in range(L):
      for j in range(L):
        l = same_as(mat2[i,j])
        mat2[i,j] = l
    print(np.flip(np.transpose(mat2),0))
  if (debug_matrix):
    printmat(mat)
  # We now look at the boundaries:
  # make a list of cluster labels to be linked across boundaries
  # but keep the original equivalence classes from above
  # if we see equivalent labels on both ends across one dimension
  # in the same place now, we have a cluster that percolates through
  # the inside of our box
  # else we just keep a list of cross-boundary links that we traverse later
  def same_as_tmp (l):
    while (labels_tmp[l]) != l:
      l = labels_tmp[l]
    return l
  def link_tmp (label1,label2):
    labels_tmp[same_as_tmp(label1)] = same_as_tmp(label2)
  labels_tmp = np.copy(labels)
  percolates = np.zeros(label+1,dtype=np.int)
  loopstarts = np.zeros(label+1,dtype=np.int)
  looplist = []
  for j in range(L):
    if mat[L-1,j]>1 and mat[0,j]>1:
      l1 = mat[0,j]
      l2 = mat[L-1,j]
      link_tmp (l1,l2)
      if same_as(l1)==same_as(l2):
        percolates[same_as(l1)]=1
      else:
        looplist.append((same_as(l1),same_as(l2),(1,0),0))
        looplist.append((same_as(l2),same_as(l1),(-1,0),0))
        loopstarts[same_as(l1)]=1
        loopstarts[same_as(l2)]=1
  for i in range(L):
    if mat[i,L-1]>1 and mat[i,0]>1:
      l1 = mat[i,0]
      l2 = mat[i,L-1]
      link_tmp (l1,l2)
      if same_as(l1)==same_as(l2):
        percolates[same_as(l1)]=1
      else:
        looplist.append((same_as(l1),same_as(l2),(0,1),0))
        looplist.append((same_as(l2),same_as(l1),(0,-1),0))
        loopstarts[same_as(l1)]=1
        loopstarts[same_as(l2)]=1
  if sum(percolates):
    print("found simply percolating cluster")
    diag = 0 #to check for diagonal percolation
  else:
    # if we did not yet find a percolating cluster inside the box,
    # we could have percolation across periodic boundary images
    # to find them, go through all cross-boundary links and try
    # to find a closed loop among them
    # if such a loop exists, *and* this loop circles around such that
    # it comes back to its start in a periodic image that is offset
    # from the original by at least one in one direction,
    # we have a percolating cluster (else, just an isolated loop that
    # happens to lie across a box boundary); to figure this out, we
    # store with the cross-boundary links also a sense of direction
    sadd=lambda xs,ys: tuple(x + y for x, y in zip(xs, ys))
    def seekpath (path,pathdir,l,lstart,looplist):
      path = list(path)
      mylooplist = list(looplist)
      path.append(l)
      for n in range(len(mylooplist)):
        (l1,l2,linkdir,inpath)=mylooplist[n]
        if not inpath:
          if l==l1: # and not (len(path)>1 and l2==path[-2]):
            mylooplist[n]=(l1,l2,linkdir,1)
            if l2==lstart:
              #print "found loop: ",path,sadd(pathdir,linkdir)
              if not sadd(pathdir,linkdir)==(0,0):
                return 1
            else:
              return seekpath (path,sadd(pathdir,linkdir),l2,lstart,mylooplist)
      return 0
    for l in range(len(loopstarts)):
      if loopstarts[l]:
        percolates[l]=seekpath ([],(0,0),l,l,looplist)
  #
  # now we're free to associate cluster labels also across boundaries
  labels = np.copy(labels_tmp)
  # The matrix now contains for each site a label corresponding to the cluster
  # this site belongs to, but the label is not always the same representative
  # of the equivalence class. We replace all matrix entries by the
  # "same as" representative of the cluster equivalence classes, to make
  # further computations easier
  maxl = 0
  for i in range(L):
    for j in range(L):
      l = same_as(mat[i,j])
      mat[i,j] = l
      if (l>maxl):
        maxl = l
  # count cluster sizes
  size = np.zeros(maxl+1,dtype=np.int)
  for i in range(L):
    for j in range(L):
      size[mat[i,j]] += 1
  # determine the largest percolating cluster
  percl = 0
  percsize = 0
  for l in range(1,len(percolates)):
    if (percolates[l]):
      ll = same_as(l)
      if (size[ll]>percsize):
        percsize = size[ll]
        percl = ll
  if debug_matrix and ((tstperc and not percl) or (percl and not tstperc)):
    print("ERROR: percfind FAILED for given test matrix")
    raise SystemExit
  return mat,percl,size,diag

# generate_percolating_matrix ((w,h),p):
# Repeatedly calls generate_matrix until a percolating matrix is obtained.
# May not return... If it does, returns the same values as the last
# successful call to generate_matrix.
def generate_percolating_matrix (L,p):
  if debug_matrix:
    return generate_matrix (L,p)
  perclabel = 0
  diag = 0
  while (perclabel == 0):
    print ("generating matrix...")
    mat,perclabel,clustersize, diag = generate_matrix (L,p)
  print(mat)
  print("percolating cluster label:",perclabel)
  return mat,perclabel,clustersize

#mat, perclabel, clustersize = generate_percolating_matrix(100,0.592)
#print(clustersize[perclabel])
