import numpy as np
import math
import matplotlib.pyplot as plt
import pylab
import matplotlib.patches as mpatches


###############################################
def chebfit( x, y, M ):
    n = len( x )
    F = np.zeros( ( M, n ) )
    F[0,:] = 1.0
    F[1,:] = x
    for i in range(2,M):
        F[i,:] = 2.0 * x * F[i-1,:] - F[i-2,:]
    A = np.matmul( F, np.transpose(F) )
    b = np.matmul( F, y )
    c = np.linalg.solve( A, b )
    z = np.polynomial.chebyshev.chebval(x, c)
    if len(y.shape) == 1:
        err = np.sqrt(np.sum(np.power(y - z,2))/float(n))
        c[abs(c) < err] = 0.0
    else:
        err = 0.0
        for i in range(y.shape[1]):
            err += np.sqrt(np.sum(np.power(y[:,i] - z[:,i],2))/float(n))
        for i in range(y.shape[1]):
            c[abs(c[:,i]) < err,i] = 0.0
    print "err =",err
    return c
###############################################
def chebdiff( c ):
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
###############################################
def arclength( xy ):
    n = len( xy )
    s = np.zeros( n )
    for i in range(1,n):
        s[i] = s[i-1] + np.hypot( xy[i][0] - xy[i-1][0],
                                  xy[i][1] - xy[i-1][1] )
    return s
###############################################
def nchoosek(n, k):
    if k == 0 or k == n:
        return 1
    else:
        return nchoosek(n-1, k) + nchoosek(n-1, k-1)
###############################################
def eval_bernstein( B, t ):
    n = len( t )
    dim = len( B.shape )
    M = len(B)-1
    f = np.zeros( (n,dim) )
    for m in range(M+1):
        p = nchoosek( M, m ) * ( np.power( 1.0-t, M-m ) * np.power( t, m ) )
        for d in range(dim):
            f[:,d] += p*B[m,d]
    return f
###############################################
def high_order_parameterization(xy):
    n = len(xy)
    s = arclength( xy )
    #s = np.linspace(0.0, 2.0, n)
    t = -1.0 + 2.0*s/s[-1]
    M = min(64,int(math.ceil(0.25*n)))
    cx = chebfit(t, xy[:,0], M)
    cy = chebfit(t, xy[:,1], M)
    c = np.transpose( np.vstack((cx, cy)) )
    return c, s
###############################################
def high_order_resampling(xy, h=0.0):
    c, s = high_order_parameterization(xy)
    d = chebdiff(c)
    L = s[-1]
    if h < 1e-15:
        n = len(xy)
    else:
        n = 1 + int(round(L/h))
    print L, h, n
    t = np.linspace(-1.0, 1.0, n)
    xyr = np.transpose(np.polynomial.chebyshev.chebval(t, c))
    #dxyr = np.transpose(np.polynomial.chebyshev.chebval(t, d) * 2.0/L)
    dxyr = np.transpose(np.polynomial.chebyshev.chebval(t, d))
    for i in range(n):
        dxyr[i] = dxyr[i] / np.hypot(dxyr[i,0], dxyr[i,1])
    return xyr, dxyr
###############################################
###############################################
def derivative_curve( f, xy ):
    shp = f.shape
    dim = len( shp )
    
    n = len( xy )
    s = np.zeros( n )
    for i in range(1,n):
        s[i] = s[i-1] + np.hypot( xy[i][0] - xy[i-1][0],
                                  xy[i][1] - xy[i-1][1] )
    df = np.zeros( (n,dim) )
    for i in range( n ):
        if i == 0:
            df[i] = (f[i+1] - f[i]) / (s[i+1] - s[i])
        elif i == n-1:
            df[i] = (f[i] - f[i-1]) / (s[i] - s[i-1])
        else:
            df[i] = 0.5 * ( (f[i+1] - f[i]) / (s[i+1] - s[i]) +
                             (f[i] - f[i-1]) / (s[i] - s[i-1]) )
    return df
###############################################
def tangents_and_normal( xy ):
    n = len( xy )
    s = arclength( xy )
    tng = derivative_curve( xy, xy )
    nor = np.zeros( (n,2) )
    for i in range( n ):
        nor[i] = [ -tng[i][1], tng[i][0] ] / np.hypot( tng[i][0], tng[i][1] )
    return tng, nor
###############################################
def offset( xy, d ):
    tng, nor = tangents_and_normal( xy )
    return xy + d*nor
###############################################
def clip_curve(curve, xhead, xtail):
    d1 = 1.e6
    d2 = 1.e6
    i1 = 0
    i2 = 0
    for i, x in enumerate(curve):
        di1 = np.hypot(x[0] - xhead[0], x[1] - xhead[1])
        if di1 < d1:
            d1 = di1
            i1 = i
        di2 = np.hypot(x[0] - xtail[0], x[1] - xtail[1])
        if di2 < d2:
            d2 = di2
            i2 = i
    return i1, i2
###############################################


yscl = 0.85

h0 = 9.34439344612/11
# front style
clf = 'k'
lwf = 1.5

# normal vector style
scl = 0.25*h0
hle = 0.04
wid = 0.75*hle
cln = "#f93802"#'r'

## Initial front
n = 1000
#t = np.linspace(-0.25*np.pi,2.25*np.pi,n)
t = np.linspace(0.0,2.0*np.pi,n)
front1 = np.transpose(np.vstack((t, yscl*np.cos(t))))
"""
xyr1, dxyr1 = high_order_resampling(front1, h0)
nor1 = np.transpose( np.vstack([
    np.transpose(xyr1[1:-1,:]), -scl*dxyr1[1:-1,1], scl*dxyr1[1:-1,0]]))

front2 = offset( front1, 0.5/yscl )
xyr2, dxyr2 = high_order_resampling(front2, h0)
nor2 = np.transpose( np.vstack([
    np.transpose(xyr2[1:-1,:]), -scl*dxyr2[1:-1,1], scl*dxyr2[1:-1,0]]))

front3 = offset( front1, 1.0/yscl )
xyr3, dxyr3 = high_order_resampling(front3, h0)
nor3 = np.transpose( np.vstack([
    np.transpose(xyr3[1:-1,:]), -scl*dxyr3[1:-1,1], scl*dxyr3[1:-1,0]]))

fig, ax = plt.subplots()

ax.plot(front1[:,0], front1[:,1], '-', color=clf, lw=lwf)
#for i in range(len(xyr1)):
#    pylab.arrow(xyr1[i,0], xyr1[i,1], -scl*dxyr1[i,1], scl*dxyr1[i,0],
#                fc=cln, ec=cln, head_width=wid, head_length=hle)
for i in range(len(nor1)):
    pylab.arrow(nor1[i,0], nor1[i,1], nor1[i,2], nor1[i,3],
                fc=cln, ec=cln, head_width=wid, head_length=hle)

ax.plot(front2[:,0], front2[:,1], '-', color=clf, lw=lwf)
#for i in range(len(xyr2)):
#    pylab.arrow(xyr2[i,0], xyr2[i,1], -scl*dxyr2[i,1], scl*dxyr2[i,0],
#                fc=cln, ec=cln, head_width=wid, head_length=hle)
for i in range(len(nor2)):
    pylab.arrow(nor2[i,0], nor2[i,1], nor2[i,2], nor2[i,3],
                fc=cln, ec=cln, head_width=wid, head_length=hle)
    
ax.plot(front3[:,0], front3[:,1], '-', color=clf, lw=lwf)
#ax.plot(xyr3[:,0], xyr3[:,1], 'r.')
for i in range(len(nor3)):
    pylab.arrow(nor3[i,0], nor3[i,1], nor3[i,2], nor3[i,3],
                fc=cln, ec=cln, head_width=wid, head_length=hle)

ax.set_aspect('equal')
plt.axis('off')
plt.show()

"""

#np.savetxt('../data/singularpointformation_ini1.dat', front)

for i in range(3):
    if i == 0:
        xy = front1
    else:
        xy = offset(front1, 0.5*i/yscl)
    xyr, dxyr = high_order_resampling(xy, h0)
    nor = np.transpose( np.vstack([
        np.transpose(xyr[1:-1,:]), -scl*dxyr[1:-1,1], scl*dxyr[1:-1,0]
    ]))
    #
    i1, i2 = clip_curve(xy, nor[0,0:2], nor[-1,0:2])
    s = str(i)
    np.savetxt('../data/singularpointformation_front_'+s+'_1.dat', xy[:i1+1,:])
    np.savetxt('../data/singularpointformation_front_'+s+'_2.dat', xy[i1:i2+1,:])
    np.savetxt('../data/singularpointformation_front_'+s+'_3.dat', xy[-1:i2-1:-1,:])
    np.savetxt('../data/singularpointformation_nor_'+s+'.dat', nor)


f = open('../data/singularpointformation_xysing.dat','w')
f.write(str(np.pi) + ' ' + str(-yscl+1.0/yscl) + '\n')
f.close()
