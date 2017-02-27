import numpy as np
import matplotlib.pyplot as plt

plt.rc('font',**{'family':'serif','sans-serif':['Computer Modern Roman']})
plt.rc('text', usetex=True)

def k(x,xp, l=1.0):
    d = (x-xp)**2
    return np.exp(-d/(l**2))
    
xp = 0
x = np.linspace(0,3,101)

fk = k(x,xp)

hf,ha = plt.subplots()
ha.plot(x, fk)
ha.set_xlabel('$|x-y|/\sigma$')
ha.set_ylabel('$k(x,y)$')
plt.show()