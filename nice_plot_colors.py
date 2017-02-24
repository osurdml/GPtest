import numpy as np

lines = np.array([[57,106,177], [218,124,48], [62,150,81], [204,37,41], [83,81,84], [107,76,154], [146,36,40], [148,139,61]], dtype='float')/255
bars = np.array([[114,147,203], [225,151,76], [132,186,91], [211,94,96], [128,133,133], [144,103,167], [171,104,87], [204,194,16]], dtype='float')/255

# These look better in print
vlines = ['cornflowerblue', 'green', 'firebrick', 'orange', 'black', 'indigo']
vbars = ['steelblue', 'darkgreen', 'darkred', 'darkorange', 'grey', 'mediumvioletred']

lines = np.hstack((lines, np.ones((lines.shape[0],1))))
bars = np.hstack((bars, np.ones((bars.shape[0],1))))

def darken(c,power=2):
    co = np.array(c).copy()
    co = np.clip(co**power, 0, 1.0)
    co[-1] = c[-1]
    return co

def lighten(c, power=2):
    co = 1.0-np.array(c).copy()
    co = darken(co, power)
    return 1.0-co