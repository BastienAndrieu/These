# -*- coding: utf-8 -*-

import numpy
from numpy import *

import matplotlib.pyplot as plt

import sys
#sys.path.append('/d/bandrieu/GitHub/Code/Python/')
sys.path.append('/home/bastien/GitHub/Code/Python/')
import lib_bezier as lbez
#from lib_compgeom import angle_between_vectors_2d


EPSfp = 1e-15
HUGEfp = 1e15

EPSalpha = 1e-6
EPSspt = 1e-4



#####################################
def radius_function(x, y):
    """
    return r, dr_dx, dr_dy
    """
    if False:
        a = 0.05
        b, c, d, e, f = 0
    else:
        a = 0.08#0.28#
        b = 0.2
        c = 2
        d = 0.13
        e = 1.5
        f = 0.25
    ox = 0.1
    oy = 0.2
    f = 0.25#0.01#
    sx = 2
    r = a*(1 + b*sin(c*x + ox) + d*cos(e*y + oy) + f*(x + sx))
    dr_dx = a*(b*c*cos(c*x + ox) + f)
    dr_dy = -a*d*e*sin(e*y + oy)
    d2r_dx2 = -a*b*c*c*sin(c*x + ox)
    d2r_dy2 = -a*d*e*e*cos(e*y + oy)
    d2r_dxdy = 0*x
    return r, dr_dx, dr_dy, d2r_dx2, d2r_dy2, d2r_dxdy
#####################################









#####################################
def norm2(u):
    return numpy.hypot(u[0], u[1])
#####################################
def perp(u):
    return numpy.array([-u[1], u[0]])
#####################################
def angle_between_vectors_2d(u, v):
    return numpy.arccos(
        min(1.0,
            max(-1.0,
                u.dot(v) / numpy.sqrt(u.dot(u)*v.dot(v))
            )
        )
    )
#####################################
class Curve:
    def __init__(self, x):
        self.x = x
        self.update()
        return

    def eval(self, t):
        return lbez.eval_bezier_curve(self.x, t)

    def evald(self, t):
        return lbez.eval_bezier_curve(self.xt, t)
    
    def evald2(self, t):
        return lbez.eval_bezier_curve(self.xtt, t)

    def curvature(self, t, xt=None, xtt=None):
        if xt is None: xt = self.evald(t)
        if xtt is None: xtt = self.evald2(t)
        sqr_xt = sum(xt**2, axis=1)
        denom = maximum(EPSfp, minimum(HUGEfp, sqr_xt*sqrt(sqr_xt)))
        det = xt[:,0]*xtt[:,1] - xt[:,1]*xtt[:,0]
        return det/denom

    def update(self):
        self.xt = lbez.diff(self.x)
        self.xtt = lbez.diff(self.xt)
        return

    def aabb(self):
        return lbez.AABB_2d_bezier_curve(self.x)

    def obb(self):
        return lbez.OBB_2d_bezier_curve(self.x)

    def make_envelope_propre(self, t):
        xy = self.eval(t)
        dxy_dt = self.evald(t)
        sqr_norm_dxy_dt = sum(dxy_dt**2, axis=1)
        inv_sqr_norm_dxy_dt = 1/sqr_norm_dxy_dt
        normal = vstack([dxy_dt[:,1], -dxy_dt[:,0]]).T

        r, dr_dx, dr_dy, d2r_dx2, d2r_dy2, d2r_dxdy = radius_function(xy[:,0], xy[:,1])
        dr_dt = dr_dx*dxy_dt[:,0] + dr_dy*dxy_dt[:,1]

        n = len(t)
        a = curve.alpha(t, dxy_dt, r, dr_dx, dr_dy, d2r_dx2, d2r_dy2, d2r_dxdy)
        
        self.eoc = [numpy.zeros((n,2)), numpy.zeros((n,2))]
        self.eod = [[], []]
        for iside in range(2):
            qn = sqrt(sqr_norm_dxy_dt - dr_dt**2)*(-1)**iside
            for j in range(2):
                self.eoc[iside][:,j] = (qn*normal[:,j] - dr_dt*dxy_dt[:,j])*r*inv_sqr_norm_dxy_dt
            self.eoc[iside] = self.eoc[iside] + xy
            # look for local self-intersections (folding)
            if numpy.amin(a[iside]) > EPSalpha:
                self.eod[iside] = [self.eoc[iside].copy()]
            else:
                flipped = []
                ihead = 0
                """
                while ihead < n:
                    #print 'ihead = %d/%d' % (ihead,n)
                    if a[iside][ihead] > EPSalpha:
                        ihead += 1
                        continue
                    for itail in range(ihead,n-1):
                        if a[iside][itail+1] > EPSalpha:
                            break
                    if itail == n-2: itail += 1
                    flipped.append([ihead,itail])
                    ihead = itail + 1
                    if ihead >= n-1: break
                """
                while ihead < n:
                    if a[iside][ihead] > EPSalpha:
                        ihead += 1
                        continue
                    else:
                        itail = ihead + 1
                        while True:#itail < n:
                            if a[iside][itail] > EPSalpha or itail >= n-1:
                                flipped.append([ihead,itail-1])
                                break
                            else:
                                itail += 1
                                continue
                        ihead = itail + 1
                #
                print '\t\t\tside ', iside, ', flipped = ', flipped
                if flipped[0][0] > 0:
                    self.eod[iside].append(self.eoc[iside][:flipped[0][0]])
                for j in range(len(flipped)-1):
                    self.eod[iside].append(self.eoc[iside][flipped[j][1]+1:flipped[j+1][0]])
                if flipped[-1][1]+1 < n-1:
                    self.eod[iside].append(self.eoc[iside][flipped[-1][1]+1:n])
        return
    #####################
    def alpha(self, t, dxy_dt=None, r=None, dr_dx=None, dr_dy=None, d2r_dx2=None, d2r_dy2=None, d2r_dxdy=None):
        # position and derivatives
        if dxy_dt is None: dxy_dt = self.evald(t)
        d2xy_dt2 = self.evald2(t)

        # curvature
        k = self.curvature(t, dxy_dt, d2xy_dt2)

        # radius function and derivatives
        if r is None or dr_dx is None or dr_dy is None or d2r_dx2 is None or d2r_dy2 is None or d2r_dxdy is None:
            xy = self.eval(t)
            r, dr_dx, dr_dy, d2r_dx2, d2r_dy2, d2r_dxdy = radius_function(xy[:,0], xy[:,1])
        dr_dt = dr_dx*dxy_dt[:,0] + dr_dy*dxy_dt[:,1]
        d2r_dt2 = dr_dx*d2xy_dt2[:,0] + dr_dy*d2xy_dt2[:,1] + (d2r_dx2 + d2r_dxdy)*dxy_dt[:,0] + (d2r_dy2 + d2r_dxdy)*dxy_dt[:,1]
        
        sqrtterm = numpy.sqrt(1 - dr_dt**2)
        commonterm = sqrtterm - dr_dt*d2r_dt2/sqrtterm
        return [commonterm + k*r, commonterm - k*r]
#####################################
def intersect_polylines(curves):
    n0 = len(curves[0])
    n1 = len(curves[1])
    intersections = []
    for i in range(n0-1,0,-1):
        tngi = curves[0][i] - curves[0][i-1]
        nori = perp(tngi)
        dii = nori.dot(curves[0][i])
        for j in range(n1-1):
            dji = nori.dot(curves[1][j]) - dii
            djp1i = nori.dot(curves[1][j+1]) - dii
            if dji*djp1i < 0:
                tngj = curves[1][j+1] - curves[1][j]
                norj = perp(tngj)
                djj = norj.dot(curves[1][j])
                dij = norj.dot(curves[0][i]) - djj
                dim1j = norj.dot(curves[0][i-1]) - djj
                if dij*dim1j < 0:
                    det = nori[0]*norj[1] - norj[0]*nori[1]
                    if abs(det) > 1e-14:
                        invdet = 1/det
                        x = (dii*norj[1] - djj*nori[1])*invdet
                        y = (nori[0]*djj - norj[0]*dii)*invdet
                        intersections.append([i, j, x, y])
                    else:
                        print '/!\ det = %s' % det
    return intersections
#####################################
def read_connected_path(filename):
    segments = []
    f = open(filename, 'r')
    n = int(f.readline())
    for i in range(n):
        ni = int(f.readline())
        bp = zeros((ni,2))
        for j in range(ni):
            bp[j] = [float(x) for x in f.readline().split()]
        segments.append(Curve(bp))
    f.close()
    return segments
#####################################


def newton_circle_packing(path, t, oprev, rprev, tol, itmax=20):
    converged = False
    for it in range(itmax):
        o = path.eval(t)
        do_dt = path.evald(t)
        r, dr_dx, dr_dy, d2r_dx2, d2r_dy2, d2r_dxdy = radius_function(o[0], o[1])
        if r < 0:
            #exit('newton_circle_packing: r < 0')
            print '\t\t**** r < 0 at (x, y) = (%s, %s)' % (o[0], o[1])
            return converged, t, o, do_dt, r
        dr_dt = do_dt[0]*dr_dx + do_dt[1]*dr_dy

        res = o - oprev

        f = dot(res, res) - (r + rprev + SPACE_BETWEEN_CIRCLES)**2
        #print '\t\tit.#%d, resf = %s' % (it, sqrt(abs(f))/r)
        if abs(f) < (tol*r)**2:
            converged = True
            break

        df_dt = 2.0*(dot(do_dt, res) - dr_dt*(r + rprev))

        dt = -f/df_dt
        #print '\t\t        dt = %s' % dt
        t += dt
    return converged, t, o, do_dt, r
#####################################
def diff_angle(u, v):
    return arctan2(u[1]*v[0] - u[0]*v[1], u[0]*v[0] + u[1]*v[1])
#####################################
def circle_arc_between_two_points(center, xy0, xy1, tolchord=1e-2):
    p0 = xy0 - center
    p1 = xy1 - center
    angle = angle_between_vectors_2d(p0, p1)%(2*numpy.pi)

    npts = max(2, int(0.5*abs(angle)/sqrt(tolchord*(2-tolchord))))
    
    t = linspace(0,1,npts)
    s = sin(angle)
    if abs(s) < 1e-15:
        #print 'forcing linear interpolation'
        p = outer(1 - t, p0) + outer(t, p1)
    else:
        # slerp
        p = (outer(sin((1 - t)*angle), p0) + outer(sin(t*angle), p1))/s
    for i in range(npts):
        p[i] = p[i] + center
    return p
#####################################




########################################################
# READ PATHS
if True:
    elems = []
    #for name in ['body', 'spout', 'handle_out', 'handle_in']:
    #for name in ['body', 'spout', 'handle_out']:
    for name in ['outer', 'inner']:
        elems.append(read_connected_path('teapot_simple_'+name+'_bcp.dat'))
else:
    elems = [read_connected_path('teapot_simple_outer_bcp.dat')]


# **************
if True:
    elems[0][3].x = lbez.reparameterize_bezier_curve(elems[0][3].x, end=0.4)
    elems[0][3].update()
    elems[0][4].x[0] = elems[0][3].x[-1]
    elems[0][4].update()
# **************

########################################################
# PLOT CURRENT INTERFACE
t = linspace(0,1,100)
tc = linspace(0,1,5)

im = int(len(t)/2)

if False:
    fig, ax = plt.subplots()

    for ipath, path in enumerate(elems):
        for icurve, curve in enumerate(path):
            xy = curve.eval(t)
            xyc = curve.eval(tc)
            ax.plot(xy[:,0], xy[:,1])#, 'k')
            ax.text(xy[im,0], xy[im,1], str(icurve))
            """
            for i in range(len(tc)):
                r, dr_dx, dr_dy, d2r_dx2, d2r_dy2, d2r_dxdy = radius_function(xyc[i][0], xyc[i][1])
                ax.add_artist(
                    plt.Circle(
                        xyc[i],
                        r,
                        ec='g',
                        fill=False
                    )
                )
            """
    ax.set_aspect('equal')
    plt.show()
    exit()
########################################################



########################################################
# PACK CIRCLES
TOL = 1e-1
FRAC_DT_PREDICTOR = 1
START_FROM_ENDPOINT = False#True
SPACE_BETWEEN_CIRCLES = 0#1e-3#

centers = []
centers_xy = []

# Find 'key' points along each path
# type: 0: G1 discontinuity, 1: alpha (locally) maximally negative
key_pts = []
tsample = numpy.linspace(0,1,100)
TOLangleG1 = 10.0 # degrees
for path in elems:
    ncurves = len(path)
    key_pts_i = []
    for icurve in range(ncurves):
        # potential type-0 key point at first knot (t=0)
        jcurve = (icurve-1)%ncurves # index of previous curve on same path
        if numpy.degrees(
                angle_between_vectors_2d(
                    path[icurve].x[1] - path[icurve].x[0],
                    path[jcurve].x[3] - path[jcurve].x[2]
                )
        ) > TOLangleG1:
            newpoint = True
            for typ, kcurve, tval, xy in key_pts_i:
                if norm2(path[icurve].x[0] - xy) < EPSspt:
                    newpoint = False
                    break
            if newpoint: key_pts_i.append([0, icurve, 0.0, path[icurve].x[0].copy()])
        #
        # potential type-1 key points
        """
        a = path[icurve].alpha(tsample)
        for iside in range(2):
            imin = numpy.argmin(a[iside])
            if a[iside][imin] < EPSalpha:
                newpoint = True
                xyi = path[icurve].eval(tsample[imin])
                for typ, kcurve, tval, xy in key_pts_i:
                    if norm2(xyi - xy) < EPSspt:
                        newpoint = False
                        break
                if newpoint: key_pts_i.append([1, icurve, tsample[imin], xyi])
        """
    key_pts.append(key_pts_i)
########################################################


########################################################
# ENVELOPPES (PROPRES) DES CERCLES
npts = 100
t = linspace(0,1,npts)

eoc_paths = []
for ipath, path in enumerate(elems):
    #print 'ipath = %d' % ipath
    eoc_branches = [[], []]
    for icurve, curve in enumerate(path):
        #print '\ticurve = %d' % icurve
        curve.make_envelope_propre(t)
########################################################


########################################################
# PREPARE ENVELOPES OF CIRCLES/DISKS
arcs = []
intersections = []
eod = [[[] for iside in range(2)] for path in elems]
for ipath, path in enumerate(elems):
    ncurves = len(path)
    for icurve, curvei in enumerate(path):
        for iside in range(2):
            eod[ipath][iside].append(curvei.eoc[iside])
        # look for self-intersections
        # ...
        # look for intersections at sharp joints
        jcurve = (icurve + 1)%ncurves
        curvej = path[jcurve]
        dxyi = curvei.evald(1.0)
        dxyj = curvej.evald(0.0)
        angle = numpy.degrees(angle_between_vectors_2d(dxyi, dxyj))
        if angle > 2.0: # sharp joint
            # MAKE ARCS
            arc = Curve(numpy.zeros((1,2)))
            arc.x[0] = curvei.x[-1]
            arc.eoc = []
            arc.eod = [[], []]
            for iside in range(2):
                xy_arc = circle_arc_between_two_points(
                    center=arc.x[0].copy(),
                    xy0=curvei.eoc[iside][-1],
                    xy1=curvej.eoc[iside][0],
                    tolchord=1e-4
                )
                arc.eoc.append(xy_arc)
            arcs.append([ipath, icurve, arc])
            # FIND INTERSECTION
            sgn = dxyi[0]*dxyj[1] - dxyi[1]*dxyj[0]
            if sgn < 0:
                inter_ij = intersect_polylines([c.eoc[0] for c in [curvei, curvej]])
                intersections.append([ipath, 0, icurve, jcurve, inter_ij])
                # add arc on side 1 to eod...
                eod[ipath][1].append(arc.eoc[1])
                arc.eod[1] = [arc.eoc[1].copy()]
            else:
                inter_ij = intersect_polylines([c.eoc[1] for c in [curvei, curvej]])
                intersections.append([ipath, 1, icurve, jcurve, inter_ij])
                # add arc on side 0 to eod...
                eod[ipath][0].append(arc.eoc[0])
                arc.eod[0] = [arc.eoc[0].copy()]

########################################################
# ARCS (PSEUDO-EDS DES COINS)
for i in range(len(arcs)):
    ipath, icurve, arc = arcs[i]
    elems[ipath].insert(icurve+1, arc)
    for j in range(i+1,len(arcs)):
        if arcs[j][0] == ipath:
            arcs[j][1] += 1
########################################################





########################################################
eoc = []
for path in elems:
    eoc_path = [[], []]
    for curve in path:
        for iside, xy in enumerate(curve.eoc):
            eoc_path[iside].append(xy.copy())
    eoc.append(eoc_path)
########################################################

########################################################
# REMOVE SELF-INTERSECTIONS
eod = []
for path in elems:
    eod_path = [[], []]
    for curve in path:
        for iside, eod_side in enumerate(curve.eod):
            for xy in eod_side:
                eod_path[iside].append(xy)
    eod.append(eod_path)


for ipath in range(len(eod)):
    for iside in range(2):
        icurve = 0
        while True:
            ncurves = len(eod[ipath][iside])
            if ncurves < 2: break
            #print '%d/%d' % (icurve, ncurves)
            curvei = eod[ipath][iside][icurve]
            jcurve = (icurve + 1)%ncurves
            curvej = eod[ipath][iside][jcurve]
            #
            if norm2(curvei[-1] - curvej[0]) > EPSspt:
                #print ipath, iside, icurve, norm2(curvei[-1] - curvej[0])
                if icurve+1 == len(eod[ipath][iside]):
                    break
                else:
                    icurve = jcurve
                    continue
            else:
                eod[ipath][iside][icurve] = numpy.vstack(
                    [
                        eod[ipath][iside][icurve][:],
                        eod[ipath][iside][jcurve][1:]
                    ]
                )
                del eod[ipath][iside][jcurve]
                if icurve == len(eod[ipath][iside]): break
                continue


xy_inter = []
for ipath in range(len(eod)):
    for iside in range(2):
        ncurves = len(eod[ipath][iside])
        for icurve in range(ncurves):
            curvei = eod[ipath][iside][icurve]
            jcurve = (icurve + 1)%ncurves
            curvej = eod[ipath][iside][jcurve]
            if norm2(curvei[-1] - curvej[0]) > EPSspt:
                intersections = intersect_polylines([curvei, curvej])
                if len(intersections) < 1:
                    print ipath, iside, icurve
                    # # #
                    """
                    fig, ax = plt.subplots()
                    ax.plot(curvei[:,0], curvei[:,1], 'r.-')
                    ax.plot(curvej[:,0], curvej[:,1], 'b.-')
                    ax.set_aspect('equal')
                    plt.show()
                    """
                    # # #
                for i, j, x, y in intersections:
                    xy_inter.append([x,y])
                if len(intersections) == 1:
                    i, j, x, y = intersections[0]
                    eod[ipath][iside][icurve] = curvei[:i+1]
                    eod[ipath][iside][jcurve] = curvej[j:]
                    eod[ipath][iside][icurve][-1] = [x,y]
                    eod[ipath][iside][jcurve][0] = [x,y]
                elif len(intersections) == 2:
                    ipoints = [intersections[k][0] for k in range(2)]
                    if ipoints[0] < ipoints[1]:
                        eod[ipath][iside][icurve] = curvei[ipoints[0]:ipoints[1]+1]
                        eod[ipath][iside][icurve][0] = intersections[0][2:4]
                        eod[ipath][iside][icurve][-1] = intersections[1][2:4]
                    else:
                        eod[ipath][iside][icurve] = curvei[ipoints[1]:ipoints[0]+1]
                        eod[ipath][iside][icurve][0] = intersections[1][2:4]
                        eod[ipath][iside][icurve][-1] = intersections[0][2:4]
                    #
                    jpoints = [intersections[k][1] for k in range(2)]
                    if jpoints[0] < jpoints[1]:
                        eod[ipath][iside][jcurve] = curvej[jpoints[0]:jpoints[1]+1]
                        eod[ipath][iside][jcurve][0] = intersections[0][2:4]
                        eod[ipath][iside][jcurve][-1] = intersections[1][2:4]
                    else:
                        eod[ipath][iside][jcurve] = curvej[jpoints[1]:jpoints[0]+1]
                        eod[ipath][iside][jcurve][0] = intersections[1][2:4]
                        eod[ipath][iside][jcurve][-1] = intersections[0][2:4]
########################################################



########################################################
# VISUALIZE
cl = ['r', 'b']
#centers_xy = numpy.array([path[ipath].eval(t) for ipath, t in centers])
centers_xy = numpy.asarray(centers_xy)

t = linspace(0,1,100)
im = int(len(t)/2)

"""
fig, ax = plt.subplots()
for ipath, path in enumerate(elems):
    for icurve, curve in enumerate(path):#outer + inner:
        xy = curve.eval(t)
        ax.plot(xy[:,0], xy[:,1], 'k')
        #
        critical = []
        a = curve.alpha(t)
        for iside in range(2):
            ia = numpy.where(a[iside] < EPSalpha)
            critical.append(ia)
            ax.plot(xy[ia,0], xy[ia,1], '.', color=cl[iside])
        #ax.text(xy[im,0], xy[im,1], '%d|%d' % (ipath, icurve), fontsize=8, color='g')
        #
        for i, xy in enumerate(curve.eoc):
            ax.plot(xy[:,0], xy[:,1], '-', color=cl[i])
            ax.plot(xy[critical[i],0], xy[critical[i],1], 'k.')
if False:
    for i in range(len(centers_xy)):
        r, dr_dx, dr_dy, d2r_dx2, d2r_dy2, d2r_dxdy = radius_function(centers_xy[i][0], centers_xy[i][1])
        ax.add_artist(
            plt.Circle(
                centers_xy[i],
                r,
                ec='g',
                fc='g',
                alpha=0.5
            )
        )

if False:
    for path in eoc_paths:
        for i in range(2):
            for xy in path[i]:
                ax.plot(xy[:,0], xy[:,1], color=cl[i])

#ax.plot(xy_inter[:,0], xy_inter[:,1], 'go')
                
ax.set_aspect('equal')
plt.show()
"""
########################################################

########################################################
# center and scale everything
aabb = 1e10*numpy.ones((2,2))
aabb[1] = -aabb[1]
for path in elems:
    for curve in path:
        aabbi = curve.aabb()
        aabb[0] = numpy.minimum(aabbi[0], aabb[0])
        aabb[1] = numpy.maximum(aabbi[1], aabb[1])

ctr = 0.5*(aabb[0] + aabb[1])
rng = 0.5*(aabb[1] - aabb[0])
scale = 1/numpy.amax(rng)

for path in elems:
    for curve in path:
        for i in range(2):
            curve.x[:,i] = (curve.x[:,i] - ctr[i])*scale
        curve.update()

for path in eod:
    for iside in range(2):
        for xy in path[iside]:
            for i in range(2):
                xy[:,i] = (xy[:,i] - ctr[i])*scale

for path in eoc:
    for iside in range(2):
        for xy in path[iside]:
            for i in range(2):
                xy[:,i] = (xy[:,i] - ctr[i])*scale
"""
for i in range(len(centers_xy)):
    centers_xy[i] = (centers_xy[i] - ctr)*scale
"""
xy_inter = numpy.asarray(xy_inter)
for i in range(2):
    xy_inter[:,i] = (xy_inter[:,i] - ctr[i])*scale
########################################################

fig, ax = plt.subplots(1,2)

for iax in range(2):
    for ipath, path in enumerate(elems):
        for curve in path:#
            xy = curve.eval(t)
            ax[iax].plot(xy[:,0], xy[:,1], 'k-')
        for typ, icurve, tval, xy in key_pts[ipath]:
            p = (xy - ctr)*scale
            r, dr_dx, dr_dy, d2r_dx2, d2r_dy2, d2r_dxdy = radius_function(xy[0], xy[1])
            ax[iax].add_artist(
                plt.Circle(
                    p,
                    scale*r,
                    ec='g',
                    fc='g',
                    alpha=0.5
                )
            )

if True:
    for iax in range(2):
        for i in range(len(centers_xy)):
            xy = (centers_xy[i] - ctr)*scale
            r, dr_dx, dr_dy, d2r_dx2, d2r_dy2, d2r_dxdy = radius_function(centers_xy[i][0], centers_xy[i][1])
            ax[iax].add_artist(
                plt.Circle(
                    xy,
                    scale*r,
                    ec='g',
                    fc='g',
                    alpha=0.5
                )
            )

for path in eoc:
    for iside in range(2):
        for xy in path[iside]:
            ax[0].plot(xy[:,0], xy[:,1], color=cl[iside])
ax[0].set_title('EdS')
 
for path in eod:
    for iside in range(2):
        for xy in path[iside]:
            if len(xy) < 2:
                ax[1].plot(xy[:,0], xy[:,1], '.')
            else:
                ax[1].plot(xy[:,0], xy[:,1], color=cl[iside])
ax[1].set_title('EdB')

xy_inter = numpy.asarray(xy_inter)
#ax.plot(xy_inter[:,0], xy_inter[:,1], 'g.')

for axe in ax:
    axe.set_aspect('equal')
plt.show()




########################################################
# TIKZ EXPORT
               
##############
def bezier_path_to_string(path):
    s = ''
    for i, curve in enumerate(path):
        b = curve.x
        if b.shape[0] < 2: continue
        if b.shape[0] > 4: continue
        if b.shape[0] < 4:
            print '!!!'
        s += '(%s, %s) .. controls (%s,%s) and (%s,%s) .. (%s,%s) --\n' % (b[0][0], b[0][1], b[1][0], b[1][1], b[2][0], b[2][1], b[3][0], b[3][1])
    s += 'cycle'
    return s
##############
def closed_polyline_to_string(poly):
    s = ''
    for xy in poly:
        for p in xy[:-1]:
            s += '(%s, %s) -- ' % (p[0], p[1])
    s += 'cycle'
    return s
##############

f = open('teapot_tikzcode.tex', 'w')
f.write('\\draw[solide]\n')
f.write('\t%s\n' % (bezier_path_to_string(elems[0])))
f.write('\t%s;' % (bezier_path_to_string(elems[1])))
f.close()



f = open('teapot_EdS_plus_tikzcode.tex', 'w')
f.write('\\draw[EdS, plus]\n')
for path in eoc:
    f.write('\t%s\n' % (closed_polyline_to_string(path[0])))
f.write(';')
f.close()

f = open('teapot_EdS_moins_tikzcode.tex', 'w')
f.write('\\draw[EdS, moins]\n')
for path in eoc:
    f.write('\t%s\n' % (closed_polyline_to_string(path[1])))
f.write(';')
f.close()




f = open('teapot_EdB_plus_tikzcode.tex', 'w')
f.write('\\draw[EdB, plus]\n')
for path in eod:
    f.write('\t%s\n' % (closed_polyline_to_string(path[0])))
f.write(';')
f.close()

f = open('teapot_EdB_moins_tikzcode.tex', 'w')
f.write('\\draw[EdB, moins]\n')
for path in eod:
    f.write('\t%s\n' % (closed_polyline_to_string(path[1])))
f.write(';')
f.close()





########################################################
