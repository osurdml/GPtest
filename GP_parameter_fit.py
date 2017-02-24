import GPy
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
#from IPython.display import display

mean_value=3
def explore_cost_function(a, b):
    cost = mean_value 
    cost += 10*math.exp(-math.sqrt((a-40)**2 + (b-40)**2)/16)
    cost += 7*math.exp(-math.sqrt((a-10)**2 + (b-90)**2)/12)
    cost += 4*math.exp(-math.sqrt((a-80)**2 + (b-60)**2)/32)
    cost += 7*math.exp(-math.sqrt((a+20)**2 + (b-50)**2)/32)
    cost += 7*math.exp(-math.sqrt((a-120)**2 + (b-50)**2)/32)
    cost += 12*math.exp(-math.sqrt((a-80)**2 + (b-20)**2)/8)
    cost += 5*math.exp(-math.sqrt((a-60)**2 + (b-80)**2)/10)
    cost += 3*math.exp(-math.sqrt((a-90)**2 + (b-90)**2)/20)
    return cost

N_POINTS = 200
X = np.random.uniform(0.,100.,(N_POINTS,2))
Y = np.zeros((N_POINTS,1))

for ii in range(N_POINTS):
    Y[ii] = explore_cost_function(X[ii][0], X[ii][1])
Y += np.random.randn(N_POINTS,1)*0.25

kernel = GPy.kern.RBF(input_dim=2, variance=10., lengthscale=20.)
gpm = GPy.models.GPRegression(X, Y, kernel)
gpm.optimize(messages=True)
gpm.optimize_restarts(num_restarts = 10)
#display(gpm)

    
fig1,ax1 = plt.subplots(1,2,sharex=True, sharey=True)
true_mat = np.zeros((100, 100))

for x in range(true_mat.shape[0]):
    for y in range(true_mat.shape[1]):
        true_mat[x,y] = explore_cost_function(x,y)
cmap = plt.cm.terrain
cmap.set_bad(color='black')
ax1[0].matshow(true_mat.transpose(), interpolation='none', cmap=cmap, vmin = 3, vmax=19)

GP_mat = np.zeros((100, 100))
Xtemp, Ytemp = np.meshgrid(np.arange(GP_mat.shape[0]), np.arange(GP_mat.shape[1]))
Xfull = np.vstack([Xtemp.ravel(), Ytemp.ravel()]).transpose()

def plot_gp(axes, model):
    Yfull, varYfull = model.predict(Xfull)
    Yfull += mean_value
    gmat = np.reshape(Yfull, (GP_mat.shape[1], GP_mat.shape[0])).transpose()
    plot_obj =  [axes.matshow(gmat.transpose(), interpolation='none', cmap=cmap, vmin = 3, vmax=19)]
    plot_obj.append(axes.plot(model.X[:,0], model.X[:,1], 'rx')[0])
    return plot_obj

video_frames=[]
# Make video
for ii in range(40):
    gpm.set_XY(X[0:ii+1,:], Y[0:ii+1]-mean_value)
    cframe = plot_gp(ax1[1], gpm)
    video_frames.append(cframe)
    #for item in cframe:
     #   item.remove()
# plt.show()
fig1.set_size_inches(10, 5)
ax1[0].set_title('True field')
ax1[1].set_title('GP estimate')

ax1[0].set_xlim([0, true_mat.shape[0]]); ax1[0].set_ylim([0, true_mat.shape[1]])
for axn in ax1:
    axn.set_aspect('equal', 'datalim')
    axn.tick_params(labelbottom='on',labeltop='off')
    axn.set_xlabel('x')
    axn.set_ylabel('y') 
    axn.autoscale(tight=True)

vid = ani.ArtistAnimation(fig1, video_frames, interval=500, repeat_delay=0)
vid.save('../GP_vid2.mp4', writer = 'avconv', fps=2, bitrate=1500)
plt.show()