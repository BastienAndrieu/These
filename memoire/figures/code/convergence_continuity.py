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
        #f = np.zeros(len(x))
        #l = np.where(x > 0)
        #f[l] = np.sqrt(x[l])
        #f = np.sqrt(0.5*(x + 1.0))
    elif j == 2:
        #f = np.zeros(len(x))
        #l = np.where(x > 0)
        #f[l] = np.power(x[l],1.5)
        #f[l] = np.power(x[l],2)#3)
        f = np.sign(x)*np.power(x,2)
        #f = 0.25*(1.0 + np.sign(x))*(1.0 - np.cos(np.pi*x))
        #f = np.power(x + 1.0,1.5)
    elif j == 3:
        f = np.zeros(len(x))
        l = np.where(x > 0)
        f[l] = np.exp(1.-1./x[l])
        #a = -0.99
        #l = np.where(x > a)
        #f[l] = np.exp(-1./(x[l]-a))
        """
        a = 1.212
        f = np.sqrt(1 - np.power(x/a, 2))
        """
        #f = x / np.sqrt(1.0 + np.power(x, 2))
    elif j == 4:
        #f = np.exp(np.sin(3.*np.power(x,3)) - 1.)
        #f = np.sqrt(1 - np.power(x/1.212, 2))
        f = 1.0 / np.sqrt(1.0 + np.power(x, 2))
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
eps = 2.2e-15

x = np.linspace(-1.0,1.0,10000)
y = np.zeros((5,len(x)))

fig, ax = plt.subplots()
for j in range(5):
    y[j] = fun(x, j)
    ax.plot(x, y[j], label='j='+str(j+1))
ax.set_xlim([-1,1])
plt.grid()
plt.legend()
plt.show()



lbl = ['discontinu','C^0','C^1','C^inf n.a.','analytique']
g = 3


#Nvec = np.rint(np.power(2, np.arange(1,10,0.5))).astype(int)
Nvec = np.hstack([np.arange(10,200,10), np.power(2,np.arange(8,11))])
err = np.zeros((5,len(Nvec)))
est = np.zeros((5,len(Nvec)))
coef = np.zeros((5,np.amax(Nvec)+1))
for i, N in enumerate(Nvec):
    xN = cgl_nodes(N)
    for j in range(5):
        fN = fun(xN, j)
        c = fcht(fN)
        #if j == 2 and N < 40 and N > 20:
        #    for l,v in enumerate(c):
        #        print l, v
        #if j < 2:
        #    c = spectral_filter(c, 1)
        z = chebval(x, c)
        err[j,i] = norminf(y[j] - z)
        for k in range(N,-1,-1):
            if np.absolute(c[k]) > eps:
                est[j,i] = np.absolute(c[k])
                if j < 3:
                    est[j,i] *= k
                break
        if i == len(Nvec)-1:
            coef[j] = np.absolute(c)

b = np.log(np.absolute(np.log(err[3,-2])))/np.log(Nvec[-2])
print 'b=',b
X = np.power(Nvec[3:-2],b)
Y = np.log(err[3,3:-2])
p = np.polyfit(X, Y, 1)
q = p[0]
print "q=",q

#X = Nvec[3:6]
#Y = np.log(err[4,3:6])
X = Nvec[:3]
Y = np.log(err[4,:3])
p = np.polyfit(X, Y, 1)
r = np.exp(-p[0])
print 'r=',r

cl = ['b','g','r','c','m']
lwd = 1.5

fig, ax = plt.subplots(nrows=1, ncols=2)
for j in range(5):
    ax[0].loglog(Nvec, err[j], '.-', label=lbl[j], color=cl[j])
    ax[0].loglog(Nvec, est[j], '+--', color=cl[j])

ax[0].loglog(Nvec,np.power(Nvec,-1.),'k:', label='N^-1', lw=lwd)
ax[0].loglog(Nvec,np.power(Nvec,-2.),'k--', label='N^-2', lw=lwd)
ax[0].loglog(Nvec,np.exp(q*np.power(Nvec,b)), 'k-.', label='exp(qN^b)', lw=lwd)

ax[0].set_xlabel('N')
ax[0].set_ylim([1e-16,1e1])
ax[0].legend(loc='best')
ax[0].grid()

for j in range(5):
    ax[1].semilogy(Nvec, err[j], '.-', label=lbl[j], color=cl[j])
    ax[1].semilogy(Nvec, est[j], '+--', color=cl[j])

ax[1].semilogy(Nvec,np.exp(q*np.power(Nvec,b)), 'k:', label='exp(qN^b)', lw=lwd)
ax[1].semilogy(Nvec,np.power(r,-Nvec), 'k--', label='r^-N', lw=lwd)


ax[1].set_xlabel('N')
ax[1].set_xlim([0,100])
ax[1].set_ylim([1e-16,1e1])
ax[1].legend(loc='best')
ax[1].grid()

plt.show()



fig, ax = plt.subplots(nrows=1, ncols=2)
for j in range(5):
    ax[0].loglog(coef[j], '.', label=lbl[j])


ax[0].loglog(Nvec,np.power(Nvec,-1.),'k:', label='N^-1', lw=lwd)
ax[0].loglog(Nvec,np.power(Nvec,-2.),'k--', label='N^-2', lw=lwd)
ax[0].loglog(Nvec,np.power(Nvec,-3.), 'k-.', label='N^-3', lw=lwd)

ax[0].set_xlabel('n')
ax[0].set_ylim([1e-16,1e1])
ax[0].legend(loc='best')
ax[0].grid()

for j in range(5):
    ax[1].semilogy(coef[j], '.', label=lbl[j])

ax[1].semilogy(Nvec,np.power(r,-Nvec), 'k--', label='r^-N', lw=lwd)

ax[1].set_xlabel('n')
ax[1].set_xlim([0,100])
ax[1].set_ylim([1e-16,1e1])
ax[1].legend(loc='best')
ax[1].grid()

plt.show()

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

"""
mrk = ['.','+','x']
fig, ax = plt.subplots()
Nvec = 10.*np.power(2,np.arange(14))
for i, N in enumerate(Nvec):
    xN = cgl_nodes(N)
    fN = fun(xN, 2)
    c = fcht(fN)
    ax.loglog(np.absolute(c), '.', marker=mrk[i%len(mrk)])
ax.loglog(Nvec, 10.*np.power(Nvec,-3.), 'k--')
ax.loglog(Nvec, np.power(Nvec,-4.), 'k:')
#ax.loglog(np.array([11,33,99,297]),
#          np.array([0.001554001554, 0.0000558581204, 0.0000020620619, 0.000000076345]),
#          'ro-')
hpi = np.pi/2.0
x = np.arange(11,Nvec[-1],2)
w = ( np.sin(x*hpi)/x +
    0.5*np.sin((x+2)*hpi)/(x+2) +
    0.5*np.sin((x-2)*hpi)/(x-2) )/np.pi
q = 4.0/(np.pi*(np.power(x,3) + 4.0*x))
#ax.loglog(x, np.absolute(w), 'r.-')
ax.loglog(x, q, 'r.-')
                        
ax.set_ylim([1e-16,1e1])
ax.grid()
plt.show()
"""
