import numpy as np
import math
import matplotlib.pyplot as plt
import pylab
import matplotlib.patches as mpatches

###############################################
def find_gcd(x, y): 
    while(y): 
        x, y = y, x % y 
    return x 
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
def envelope_of_spheres( xy, nu ):
    n = len( xy )
    dnu = derivative_curve( nu, xy )
    tng, nor = tangents_and_normal( xy )
    xye = np.zeros( (n,2) )
    for i in range( n ):
        dxy = np.hypot( tng[i][0], tng[i][1] )
        xye[i] = xy[i] + nu[i] / dxy * (
            np.sqrt( dxy**2 - dnu[i]**2 )*nor[i] -
            dnu[i]*tng[i]/dxy )
    return xye
###############################################
def remove_cusps( x ):
    n = len( x )
    s = arclength( x )
    inter = []
    branches = []
    for i in range( n-1 ):
        tng = x[i+1] - x[i]
        nor = [-tng[1], tng[0]] / np.hypot( tng[0], tng[1] )
        dotp = np.zeros( n )
        for j in range(n):
            dotp[j] = np.dot( x[j] - x[i], nor )
        for j in range( i+1, n-1 ):
            if j < i-1 or j > i+1:
                if dotp[j] * dotp[j+1] <= 0.0:
                    tngj = x[j+1] - x[j]
                    norj = [-tngj[1], tngj[0]] / np.hypot( tngj[0], tngj[1] )
                    dotpj = np.array( (np.dot(x[i]-x[j],norj), np.dot(x[i+1]-x[j],norj)) )
                    if dotpj[0] * dotpj[1] <= 0.0:
                        t = dotp[j] / ( dotp[j] - dotp[j+1] )
                        xi = (1.0 - t)*x[j] + t*x[j+1]
                        si = (1.0 - t)*s[j] + t*s[j+1]
                        inter.append( [i,j,si,xi] )
    if len( inter ) < 1:
        branches.append( x )
    else:
        for ib in range(len(inter)+1):
            if ib == 0:
                bi = np.vstack( [ x[0:inter[ib][0],:], inter[ib][3] ] )
            elif ib == len(inter):
                bi = np.vstack( [ inter[ib-1][3], x[inter[ib-1][1]+1:n,:] ] )
            else:
                bi = np.vstack( [ inter[ib-1][3],
                                  x[inter[ib-1][1]+1:inter[ib][0],:],
                                  inter[ib][3] ] )
            branches.append( bi )
    return branches
###############################################
def envelope_of_balls(branches_xy, dt, h):
    EPSxyz = 1e-2*h
    eostmp = []
    for i, bxy in enumerate(branches_xy):
        nui = dt*fnu(bxy)
        eosi = envelope_of_spheres( bxy, dt*nui )
        eostmp.append(eosi)
    #
    eos = []
    skipnext = False
    for i, bxy in enumerate(branches_xy):
        if skipnext:
            skipnext = False
            continue
        nui = dt*fnu(bxy)
        eos.append(eostmp[i])
        if i+1 < len(branches_xy):
            norL = eostmp[i][-1] - bxy[-1]
            norL = norL / np.hypot(norL[0], norL[1])
            norR = eostmp[i+1][0] - branches_xy[i+1][0]
            norR = norR / np.hypot(norR[0], norR[1])
            nLdotnR = np.dot(norL, norR)
            if nLdotnR < TOLcos:
                if norL[0]*norR[1] - norL[1]*norR[0] > 0.0: # concave vertex
                    aux = np.vstack([eostmp[i], eostmp[i+1]])
                    auxr = remove_cusps(aux)
                    eos.pop()
                    skipnext = True
                    for baux in auxr:
                        eos.append(baux)
                else: # convex vertex
                    nuip1 = dt*fnu(branches_xy[i+1])
                    nLdotnR = np.dot(norL, norR)
                    if nLdotnR < TOLcos:
                        L = np.arccos(nLdotnR)*dt*nui[-1]
                        n = int(math.ceil(100.0*L/h))
                        t = np.linspace(np.arctan2(norL[1],norL[0]), np.arctan2(norR[1],norR[0]),n)
                        xv = bxy[-1,0] + dt*nui[-1]*np.cos(t)
                        yv = bxy[-1,1] + dt*nui[-1]*np.sin(t)
                        xyv = np.transpose(np.vstack((xv,yv)))
                        eos.append(xyv)
    #
    eob = []
    for i, x in enumerate(eos):
        y = remove_cusps(x)
        for z in y:
            eob.append(z)
    return eob, eos
###############################################
def fnu( xy ):
    #return 0.3 * ( 1.0 + 0.2*np.cos(2.0*xy[:,0]-1.0) + 0.1*xy[:,0] )#1.3*np.ones(len(xy)) #
    return 0.35 * ( 1.0 + 0.2*np.cos(2.0*xy[:,0]-1.0) + 0.1*xy[:,0] )
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


h0 = 0.65#0.2505
# normal vector style
scl = 0.25*h0
hle = 0.04
wid = 0.75*hle
cln = "#f93802"#'r'


# sphere/ball style
clc = "#5c98d1" #'b'#(0.5,0.5,0.5)
clfc = "#dbedff" #"#d1e8ff"
lsc =  '--'
dashc = [5,2]#[2,2]
lwc = 0.5

# front style
clf = 'k'
lwf = 1.5

# domain style
cld = "w"#"#fff2db"

tc = np.linspace(0.0, 2.0*np.pi, 200)

TOLcos = np.cos(np.radians(10.0))


## Initial branches as Bezier curves
branches_cp = []


#branches_cp.append(np.array( [(0.0,0.0), (1.0,0.0), (2.0,0.5), (3.0, 1.5)] ))
branches_cp.append(np.array( [(0.9449,0.1488), (1.6300,0.3646), (2.3150,0.8150), (3.0,1.5)] ))
branches_cp.append(np.array( [(3.0, 1.5), (3.5,0.5), (4.5,-1.0), (4.0,0.0), (4.5,1.0), (5.0,2.0), (6.0, 2.25)] ))

#branches_cp.append(np.array( [(0.0,0.0), (1.2,0.0), (2.3,1.2), (3.5, 1.5)] ))
#branches_cp.append(np.array( [(1.0,0.0), (2.0,0.0), (2.8,0.9), (3.5, 1.5)] ))
#branches_cp.append(np.array( [(3.5, 1.5), (4.2,-1.4), (4.6,0.8), (5.8,1.7), (6.2,1.8)] ))
"""
#branches_cp.append(np.array( [(0.0,-5.0), (0.1,-2.5), (0.0,0.0)] ))
branches_cp.append(np.array( [(0.0,0.0), (2.25,-0.2), (4.5,0.0)] ))
branches_cp.append(np.array( [(4.5,0.0), (4.75, -0.9), (5.0,-2.0)] ))
branches_cp.append(np.array( [(5.0,-2.0), (4.0,-3.0), (1.0,1.0), (1.0,-5.0), (0.0,-5.0)] ))
#branches_cp.append(np.array( [(5.0,-2.0), (2.5,-3.0), (0.0,-5.0)] ))
"""

## Initial sampling
f = 10.0
n = 1000
branches_xy = []
lengths = []
t = np.linspace( 0.0, 1.0, n )
for i, b in enumerate(branches_cp):
    z = eval_bernstein(b, t)
    s = arclength(z)
    lengths.append(int(f*s[-1]))
    branches_xy.append(z)


cl = ['r','b']

fig, ax = plt.subplots()
b = branches_xy

ini_curv = np.empty((0,2))
for bxy in b:
    ini_curv = np.vstack((ini_curv, bxy))
ax.plot(ini_curv[:,0], ini_curv[:,1], '-', color=clf, lw=lwf, zorder=10)
    #ax.plot(bxy[:,0], bxy[:,1], 'k-', lw=lwf, color=clf, zorder=10 )

timestep = [1.0]#, 0.5, 0.5]
for k, dt in enumerate(timestep):
    if k == 0:
        circ_ctr = np.empty((0,2))
        tng_orig = np.empty((0,2))
        tng_vect = np.empty((0,2))
        for j, bxy in enumerate(b):
            xyr, dxyr = high_order_resampling(bxy, h0)
            if j == len(b)-1:
                circ_ctr = np.vstack([circ_ctr, xyr])
            else:
                circ_ctr = np.vstack([circ_ctr, xyr[:-1,:]])
            tng_orig = np.vstack([tng_orig, xyr])
            tng_vect = np.vstack([tng_vect, scl*dxyr])
            for i in range(len(xyr)):
                pylab.arrow(xyr[i,0], xyr[i,1], -scl*dxyr[i,1], scl*dxyr[i,0],
                            fc=cln, ec=cln, head_width=wid, head_length=hle, zorder=3)
            circ_rad = dt*fnu(circ_ctr)
            for i in range(len(circ_ctr)):
                ax.plot(circ_ctr[i,0] + circ_rad[i]*np.cos(tc), circ_ctr[i,1] + circ_rad[i]*np.sin(tc), 'k--')
                ball = mpatches.Circle( circ_ctr[i], circ_rad[i], ec="none", fc=clfc, zorder=1 )
                ax.add_patch( ball )
    b, eos = envelope_of_balls(b, dt, h0)
    eos_curv = np.empty((0,2))
    for bxy in eos:
        eos_curv = np.vstack((eos_curv, bxy))
    ax.plot(eos_curv[:,0], eos_curv[:,1], '-', color=clf, lw=lwf, zorder=4)
    
    eob_curv = np.empty((0,2))
    for bxy in b:
        eob_curv = np.vstack((eob_curv, bxy))
    ax.plot(eob_curv[:,0], eob_curv[:,1], '-', color=cl[k%len(cl)], lw=lwf, zorder=10)
    
ax.set_aspect('equal')
#ax.set_xlim(0.5,5.0)
#ax.set_ylim(-0.25,2.5)
plt.axis('off')
plt.show()


circ_data = np.transpose(np.vstack([np.transpose(circ_ctr[1:-1,:]), circ_rad[1:-1]]))
nor_data = np.transpose(np.vstack([np.transpose(tng_orig[1:-1,:]), -tng_vect[1:-1,1], tng_vect[1:-1,0]]))



i1, i2 = clip_curve(ini_curv, circ_data[0], circ_data[-1])
np.savetxt('../data/huygens_principle_ini1.dat', ini_curv[:i1+1,:])
np.savetxt('../data/huygens_principle_ini2.dat', ini_curv[i1:i2+1,:])
np.savetxt('../data/huygens_principle_ini3.dat', ini_curv[-1:i2-1:-1,:])


i1, i2 = clip_curve(eos_curv, circ_data[0], circ_data[-1])
np.savetxt('../data/huygens_principle_eos1.dat', eos_curv[:i1+1,:])
np.savetxt('../data/huygens_principle_eos2.dat', eos_curv[i1:i2+1,:])
np.savetxt('../data/huygens_principle_eos3.dat', eos_curv[-1:i2-1:-1,:])

i1, i2 = clip_curve(eob_curv, circ_data[0], circ_data[-1])
np.savetxt('../data/huygens_principle_eob1.dat', eob_curv[:i1+1,:])
np.savetxt('../data/huygens_principle_eob2.dat', eob_curv[i1:i2+1,:])
np.savetxt('../data/huygens_principle_eob3.dat', eob_curv[-1:i2-1:-1,:])



np.savetxt('../data/huygens_principle_ini.dat', ini_curv)
np.savetxt('../data/huygens_principle_eos.dat', eos_curv)
np.savetxt('../data/huygens_principle_eob.dat', eob_curv)
np.savetxt('../data/huygens_principle_circ.dat', circ_data)
np.savetxt('../data/huygens_principle_nor.dat', nor_data)
