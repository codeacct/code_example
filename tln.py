import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import random

# implements toy example of threshold-linear network from "pattern completion in symmetric threshold-linear networks" https://arxiv.org/abs/1512.00897 by Curto and Morrison

# wrote this a couple of years ago just to practice with python and get a better sense of the topic. 
# hobby level code, just to demonstrate very amateur-level outside of work interests.
# one project that may be fun to look at at some point is: https://github.com/facebookresearch/poincare-embeddings


x_dot = 0
x_i = 0


wall =  np.zeros((1,1),dtype=np.int)

for x in range(n):
    print wall
    t = wall.shape[0] + 1
    wall_new = np.zeros((t,t), dtype=np.float)
    wall_new[0:wall.shape[0], 0:wall.shape[1]] = wall
    wall = wall_new
    # fix the edges, either too from nothing.
    foo = ['a', 'b', 'c']

    for r in range(wall.shape[0] - 1):
        pp = random.choice(foo)
        if pp == 'a':
            #to
            wall[r][-1] = 1
            wall[-1][r] = 0
        if pp == 'b':
            #from, everyone must have one outgoing
            wall[r][-1] = 0
            wall[-1][r] = 1
        if pp == 'c':
            #none
            wall[r][-1] = 0
            wall[-1][r] = 0
#print W_ij
W_ij = wall
x_j = 0
theta = 1.0 # constant external drive, greater than 0
delta = .2 # > 0
#epsilon = .20 # 0 < epsilon < (delta)/(delta + 1)
epsilon = delta/(delta + 1.0)
def nonlinearity(value):
    if value<0:
        return 0
    else:
        return value

W_ij[ W_ij > 0 ] = -1.0 + epsilon 
#np.where(W_ij > .5, W_ij, -1+epsilon)
W_ij[ W_ij == 0 ] = -1.0 + delta 

np.fill_diagonal(W_ij, 0)
print W_ij


def new_model(y, t):
    #dy_i= -y[here] + nonlinearity(sum(weights)*other neurons + drive)
    xo = np.empty_like(y)
    #y will be an array with 4 neurons
    for x in range(16):
        t1 = -y[x]
        t2 = W_ij[x]
        m = np.inner(t2, y.T)
        t3 = nonlinearity(m + theta)
        xo[x] = t1 + t3
    print xo
    return xo
#"X0=.01*rand(n,1)"
init_state = np.random.rand(17)*.01
print "here"
print init_state
#init_state = np.array([.2,.3,.1,.4])
time = np.linspace(0.0, 20.0, 100)
#new_model(init_state, 0)
y = odeint(new_model, init_state, time)
plt.plot(time, y[:,0], time, y[:,1], time, y[:,2], time, y[:,3],
         time, y[:,4], time, y[:,5], time, y[:,6], time, y[:,7],
         time, y[:,8], time, y[:,9], time, y[:,10], time, y[:,11])#,
         #time, y[:,12], time, y[:,13], time, y[:,14], time, y[:,15]) # y[:,0] is the first column of y
plt.show() 
    
     


