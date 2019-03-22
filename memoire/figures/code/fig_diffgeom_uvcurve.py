import numpy
from numpy.polynomial.chebyshev import chebval
import matplotlib.pyplot as plt

import sys
sys.path.append('/stck/bandrieu/Bureau/Python/mylibs/')
sys.path.append('/d/bandrieu/GitHub/These/memoire/figures/code/')
import chebyshev as cheb
import my_lib1 as mylb
import chebyshev_lib as cheblib

bpsi = numpy.array([[0.2,-.8],[1.0,-0.5],[1.0,0.4],[0.4,0.7]])
bpsi[:,0] = -bpsi[:,0] + 0.1
ang = -0.3*numpy.pi
c = numpy.cos(ang)
s = numpy.sin(ang)
rot = numpy.array([[c,s],[-s,c]])
bpsi = mylb.mymatmul(bpsi, rot)
bpsi = 0.85*bpsi
bpsi[:,0] = bpsi[:,0] + 0.1
bpsi[:,1] = bpsi[:,1] - 0.1
bpsi = numpy.flip(bpsi, axis=0)
print bpsi
M = cheb.B2Cmatrix(len(bpsi))
cpsi = mylb.mymatmul(M, bpsi)
dwcpsi = cheblib.diff(cpsi)

mylb.writeCoeffs1(cpsi,'../data/fig_diffgeom_cpsi.cheb')

uv = chebval(numpy.linspace(-1.,1.,100), cpsi)
numpy.savetxt('../data/fig_diffgeom_uvcurve.dat', uv.T)



fig, ax = plt.subplots()

ax.plot(uv[0], uv[1], 'k-')


w = -0.2
uv = chebval(w, cpsi)
duv = chebval(w, dwcpsi)
print uv, duv
ax.plot(uv[0], uv[1], 'k*')
ax.plot([uv[0], uv[0] + duv[0]], [uv[1], uv[1] + duv[1]], 'k')
#ax.quiver(uv[0], uv[1], duv[0], duv[1], 'k')




uv = -1.0 + 2.*numpy.array([6,2])/9.
duv = [0.3,0.3]
ax.plot([uv[0], uv[0] + duv[0]], [uv[1], uv[1]], 'r')
ax.plot([uv[0], uv[0]], [uv[1], uv[1] + duv[1]], 'b')
ax.plot([uv[0], uv[0] + duv[0]], [uv[1], uv[1] + duv[1]], 'g')





ax.set_aspect('equal')
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
plt.show()


