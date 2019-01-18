import numpy as np
from numpy.polynomial.chebyshev import *
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt

##########################################
def cgl_nodes(N):
    return np.cos(np.arange(0,N+1)*np.pi/float(N))
##########################################
def fcht(f):
    N = len(f) - 1
    v = np.hstack((f, f[N-1:0:-1]))
    F = fft(v)
    F = F.real / float(N)
    c = F[0:N+1]
    c[0] *= 0.5
    c[N] *= 0.5
    return c
##########################################
def diff( c ):
    shp = c.shape
    degr = shp[0]-1
    if len( shp ) == 1:
	d = np.zeros( degr+1 )
    else:
	d = np.zeros( (degr+1, shp[1]) ) 
    
    if degr < 1:
	return d
    d[degr-1] = 2.0 * float(degr) * c[degr]
    
    if degr > 1:
	d[degr-2] = 2.0 * float(degr-1) * c[degr-1]
    
    if degr > 2:
	for i in range( degr-3, -1, -1 ):
	    d[i] = d[i+2] + 2.0 * float( i+1 ) * c[i+1]
    d[0] *= 0.5
    return d
##########################################
def fun(x):
    f0 = 1./(1. + 25.*np.power(x,2))
    f1 = -50.*x*np.power(f0,2)
    f2 = -50.*f0*(f0 + 2.*x*f1)
    return f0, f1, f2
##########################################
def norminf(a):
    return np.amax(np.absolute(a))
##########################################

eps = 2.2e-15

x = np.linspace(-1.0,1.0,10000)
f0, f1, f2 = fun(x)
amx0 = norminf(f0)
amx1 = norminf(f1)
amx2 = norminf(f2)

Nvec = np.arange(10,200,10)
err = np.zeros((3,len(Nvec)))
for i, N in enumerate(Nvec):
    xN = cgl_nodes(N)
    f0N, f1N, f2N = fun(xN)
    
    d0 = fcht(f0N)
    if True:#i < len(Nvec)-1:
        d0[np.where(np.absolute(d0) < eps)] = 0.0

    d1 = diff(d0)
    if True:#i < len(Nvec)-1:
        d1[np.where(np.absolute(d1) < eps)] = 0.0
    d2 = diff(d1)

    y0 = chebval(x, d0)
    y1 = chebval(x, d1)
    y2 = chebval(x, d2)
    
    err[0,i] = norminf(y0 - f0)/amx0
    err[1,i] = norminf(y1 - f1)/amx1
    err[2,i] = norminf(y2 - f2)/amx2

r = np.sqrt(1./25. + 1.) + 1./5.

mrk = ['s','o','^']
clr = ['b','r','g']
fig, ax = plt.subplots()
for k in range(3):
    ax.semilogy(Nvec, err[k], '.-', marker=mrk[k], color=clr[k], label='k='+str(k))

ax.plot(Nvec, np.power(r,-Nvec)/10., 'k--')

ax.set_xlabel('N')
ax.set_ylabel('max |D^(k)_Nf - f| / max |f|')
ax.set_ylim([1e-16,1e1])
plt.legend()
plt.grid()
plt.show()


c = np.absolute(np.vstack([d0,d1,d2]))

fig, ax = plt.subplots()
for k in range(3):
    ax.semilogy(c[k], '.', marker=mrk[k], color=clr[k], label='k='+str(k))

ax.plot(Nvec, np.power(r,-Nvec)/10., 'k--')

ax.set_xlabel('n')
ax.set_ylabel('|\\tilde{f}_n|')
ax.set_ylim([1e-16,1e1])
plt.legend()
plt.grid()
plt.show()
