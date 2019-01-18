import numpy as np
import matplotlib.pyplot as plt

m = 4

print 'err ~ exp(-q*dof^b)'

bm = 0
qm = 0
for ig, grandeur in enumerate(['position','aire','volume']):
    data = np.loadtxt('../data/vortex_erreur_'+grandeur+'_vs_dof_RK4.dat', skiprows=1)
    x = data[:m,1]
    y = data[:m,2]
    b = np.log(np.absolute(np.log(y[-1])))/np.log(x[-1])
    X = np.power(x,b)
    Y = np.log(y)
    p = np.polyfit(X, Y, 1)
    q = -p[0]
    print grandeur[0],' : b = ',b,'\tq = ',q
    bm += b
    qm += q
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].loglog(x, y)
    ax[0].loglog(x, np.exp(p[1] - q*np.power(x,b)), 'k--')
    ax[0].set_xlabel('dof')
    ax[0].set_ylabel('err')
    ax[0].grid()

    ax[1].plot(X, Y)
    ax[1].plot(X, p[1] - q*X, 'k--')
    ax[1].set_xlabel('dof^b')
    ax[1].set_ylabel('log(err)')
    ax[1].grid()
    plt.show()

bm /= 3.
qm /= 3.
print 'b_avg = ',bm,'\tqavg = ',qm
