import numpy
import matplotlib.pyplot as plt

buv = numpy.loadtxt('bpts.dat')
bedg = numpy.loadtxt('bedg.dat', dtype=int) - 1
tri = numpy.loadtxt('tri.dat', dtype=int) - 1
uv = numpy.loadtxt('uv.dat')



fig, ax = plt.subplots()

ax.triplot(uv[:,0], uv[:,1], tri, color='b')

for e in bedg:
    ax.plot(buv[e,0], buv[e,1], 'r.-')

ax.set_aspect('equal')
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)

plt.show()
