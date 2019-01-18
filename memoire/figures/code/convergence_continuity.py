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
def norminf(a):
    return np.amax(np.absolute(a))
##########################################
def fun(x, j):
    if j == 0:
        f = np.sign(x)
    elif j == 1:
        f = np.absolute(x)
    elif j == 2:
        f = np.zeros(len(x))
        l = np.where(x > 0)
        f[l] = np.power(x[l],2)
    elif j == 3:
        f = np.zeros(len(x))
        l = np.where(x > 0)
        f[l] = np.exp(1.-1./x[l])
    elif j == 4:
        f = np.exp(np.sin(3.*np.power(x,3)) - 1.)
        #eps = 1e-16
        #a = -np.log(eps)
        #f = np.exp(-a*np.power(x,2))
    return f
##########################################
def spectral_filter(c, p):
    n = len(c)
    w = np.arange(n)
    eps = 2.2e-16
    a = np.log(eps)
    f = np.exp(a*np.power(w,2*p))
    return c*f
##########################################
x = np.linspace(-1.0,1.0,10000)
y = np.zeros((5,len(x)))

#fig, ax = plt.subplots()
for j in range(5):
    y[j] = fun(x, j)
    #ax.plot(x, y[j], label='j='+str(j+1))
#ax.set_xlim([-1,1])
#plt.grid()
#plt.legend()
#plt.show()



lbl = ['discontinu','C^0','C^1','C^inf n.a.','analytique']
g = 3


#Nvec = np.rint(np.power(2, np.arange(1,10,0.5))).astype(int)
Nvec = np.hstack([np.arange(10,200,10), np.power(2,np.arange(8,11))])
err = np.zeros((5,len(Nvec)))
coef = np.zeros((5,np.amax(Nvec)+1))
for i, N in enumerate(Nvec):
    xN = cgl_nodes(N)
    for j in range(5):
        fN = fun(xN, j)
        c = fcht(fN)
        if j < 2:
            c = spectral_filter(c, 1)
        z = chebval(x, c)
        err[j,i] = norminf(y[j] - z)
        if i == len(Nvec)-1:
            coef[j] = np.absolute(c)

b = np.log(np.absolute(np.log(err[3,-2])))/np.log(Nvec[-2])
print 'b=',b
X = np.power(Nvec[3:-2],b)
Y = np.log(err[3,3:-2])
p = np.polyfit(X, Y, 1)
q = p[0]
print "q=",q

X = Nvec[3:6]
Y = np.log(err[4,3:6])
p = np.polyfit(X, Y, 1)
r = np.exp(-p[0])
print 'r=',r

fig, ax = plt.subplots(nrows=1, ncols=2)
for j in range(5):
    ax[0].loglog(Nvec, err[j], '.-', label=lbl[j])

ax[0].loglog(Nvec,np.power(Nvec,-1.),'k:', label='N^-1')
ax[0].loglog(Nvec,np.power(Nvec,-2.),'k--', label='N^-2')
ax[0].loglog(Nvec,np.exp(q*np.power(Nvec,b)), 'k-.', label='exp(qN^b)')

ax[0].set_xlabel('N')
ax[0].set_ylim([1e-16,1e1])
ax[0].legend(loc='best')
ax[0].grid()

for j in range(5):
    ax[1].semilogy(Nvec, err[j], '.-', label=lbl[j])

#ax[1].semilogy(Nvec,np.exp(q*np.power(Nvec,b)), 'k:', label='exp(qN^b)')
ax[1].semilogy(Nvec,np.power(r,-Nvec), 'k--', label='r^-N')

ax[1].set_xlabel('N')
ax[1].set_xlim([0,100])
ax[1].set_ylim([1e-16,1e1])
ax[1].legend(loc='best')
ax[1].grid()

plt.show()


"""
fig, ax = plt.subplots(nrows=1, ncols=2)
for j in range(5):
    ax[0].loglog(coef[j], '.', label=lbl[j])

ax[0].set_xlabel('n')
ax[0].set_ylim([1e-16,1e1])
ax[0].legend(loc='best')
ax[0].grid()

for j in range(5):
    ax[1].semilogy(coef[j], '.', label=lbl[j])
 
ax[1].set_xlabel('n')
ax[1].set_xlim([0,100])
ax[1].set_ylim([1e-16,1e1])
ax[1].legend(loc='best')
ax[1].grid()

plt.show()
"""
"""
for g in range(2,5):
    fig, ax = plt.subplots()
    X = np.power(np.log(Nvec),g)
    Y = np.log(err[3])
    
    ax.plot(X,Y,'.-')
    ax.grid()
    plt.show()
"""

"""
X = np.power(Nvec,b)
Y = np.log(err[3])
p = np.polyfit(X,Y, 1)
q = p[0]
print "q=",q

fig, ax = plt.subplots()
ax.plot(X, Y,'.-')
ax.plot(X,q*X,'k--')
ax.grid()
plt.show()
"""

"""
fig, ax = plt.subplots()
plt.semilogx(Nvec, np.log(np.absolute(np.log(err[3])))/np.log(Nvec))
plt.grid()
plt.show()
"""


