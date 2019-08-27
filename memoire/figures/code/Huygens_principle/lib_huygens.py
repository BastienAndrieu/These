# -*- coding: utf-8 -*-
import numpy

import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python/')
#sys.path.append('/home/bastien/GitHub/Code/Python/')
import lib_bezier as lbez


EPSfp = 1e-15
HUGEfp = 1e15
EPSalpha = 1e-6
EPSspt = 1e-4

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
        sqr_xt = numpy.sum(xt**2, axis=1)
        denom = numpy.maximum(EPSfp, numpy.minimum(HUGEfp, sqr_xt*numpy.sqrt(sqr_xt)))
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

    def make_envelope_propre(self, t, radfun):
        xy = self.eval(t)
        dxy_dt = self.evald(t)
        sqr_norm_dxy_dt = numpy.sum(dxy_dt**2, axis=1)
        inv_sqr_norm_dxy_dt = 1/sqr_norm_dxy_dt
        normal = numpy.vstack([dxy_dt[:,1], -dxy_dt[:,0]]).T
        #
        r, dr_dx, dr_dy, d2r_dx2, d2r_dy2, d2r_dxdy = radfun(xy[:,0], xy[:,1])#radius_function(xy[:,0], xy[:,1])
        dr_dt = dr_dx*dxy_dt[:,0] + dr_dy*dxy_dt[:,1]
        #
        n = len(t)
        a = self.alpha(t, dxy_dt, r, dr_dx, dr_dy, d2r_dx2, d2r_dy2, d2r_dxdy)
        #
        self.eoc = [numpy.zeros((n,2)), numpy.zeros((n,2))]
        self.eod = [[], []]
        for iside in range(2):
            qn = numpy.sqrt(sqr_norm_dxy_dt - dr_dt**2)*(-1)**iside
            for j in range(2):
                self.eoc[iside][:,j] = (qn*normal[:,j] - dr_dt*dxy_dt[:,j])*r*inv_sqr_norm_dxy_dt
            self.eoc[iside] = self.eoc[iside] + xy
            # look for local self-intersections (folding)
            if numpy.amin(a[iside]) > EPSalpha:
                self.eod[iside] = [self.eoc[iside].copy()]
            else:
                flipped = []
                ihead = 0
                while ihead < n:
                    if a[iside][ihead] > EPSalpha:
                        ihead += 1
                        continue
                    else:
                        itail = ihead + 1
                        while True:
                            if itail > n-1:
                                flipped.append([ihead,itail-1])
                                break
                            if a[iside][itail] > EPSalpha:# or itail > n-1:
                                flipped.append([ihead,itail-1])
                                break
                            else:
                                itail += 1
                                continue
                        ihead = itail + 1
                #
                if flipped[0][0] > 0:
                    self.eod[iside].append(self.eoc[iside][:flipped[0][0]])
                for j in range(len(flipped)-1):
                    self.eod[iside].append(self.eoc[iside][flipped[j][1]+1:flipped[j+1][0]])
                if flipped[-1][1]+1 < n-1:
                    self.eod[iside].append(self.eoc[iside][flipped[-1][1]+1:n])
        return
    #####################
    def alpha(self, t, dxy_dt=None, r=None, dr_dx=None, dr_dy=None, d2r_dx2=None, d2r_dy2=None, d2r_dxdy=None, radfun=None):
        # cf. 'Data transfer and interface propagation in multicomponent simulations', Jiao (2001)
        # position and derivatives
        if dxy_dt is None: dxy_dt = self.evald(t)
        d2xy_dt2 = self.evald2(t)
        # curvature
        k = self.curvature(t, dxy_dt, d2xy_dt2)
        # radius function and derivatives
        if r is None or dr_dx is None or dr_dy is None or d2r_dx2 is None or d2r_dy2 is None or d2r_dxdy is None:
            if radfun is None: exit('alpha: you must specify a radfun')
            xy = self.eval(t)
            r, dr_dx, dr_dy, d2r_dx2, d2r_dy2, d2r_dxdy = radfun(xy[:,0], xy[:,1])#radius_function(xy[:,0], xy[:,1])
        dr_dt = dr_dx*dxy_dt[:,0] + dr_dy*dxy_dt[:,1]
        d2r_dt2 = dr_dx*d2xy_dt2[:,0] + dr_dy*d2xy_dt2[:,1] + (d2r_dx2 + d2r_dxdy)*dxy_dt[:,0] + (d2r_dy2 + d2r_dxdy)*dxy_dt[:,1]
        #
        sqrtterm = numpy.sqrt(1 - dr_dt**2)
        commonterm = sqrtterm - dr_dt*d2r_dt2/sqrtterm
        return [commonterm + k*r, commonterm - k*r]
#####################################


#####################################
def norm2(u):
    return numpy.hypot(u[0], u[1])
#####################################
def perp(u):
    return numpy.array([-u[1], u[0]])
#####################################


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

def circle_arc_between_two_points(center, xy0, xy1, tolchord=1e-2):
    p0 = xy0 - center
    p1 = xy1 - center
    angle = angle_between_vectors_2d(p0, p1)%(2*numpy.pi)
    #
    npts = max(2, int(0.5*abs(angle)/numpy.sqrt(tolchord*(2-tolchord))))
    #
    t = numpy.linspace(0,1,npts)
    s = numpy.sin(angle)
    if abs(s) < 1e-15:
        #print 'forcing linear interpolation'
        p = numpy.outer(1 - t, p0) + numpy.outer(t, p1)
    else:
        # slerp
        p = (numpy.outer(numpy.sin((1 - t)*angle), p0) + numpy.outer(numpy.sin(t*angle), p1))/s
    for i in range(npts):
        p[i] = p[i] + center
    return p
#####################################


#####################################
def circle_packing(curves, radfun, nsample, itmax):
    tsample = numpy.linspace(0,1,nsample)
    #
    spacing = 0
    #
    for it in range(itmax):
        print '\tCP: it #%d' % it
        xyc = numpy.empty((0,2))
        rc = numpy.empty(0)
        for curve in curves:
            xy = curve.eval(tsample)
            r, dr_dx, dr_dy, d2r_dx2, d2r_dy2, d2r_dxdy = radfun(xy[:,0], xy[:,1])#radius_function(xy[:,0], xy[:,1])
            #
            xyc = numpy.vstack([xyc, xy[:-1]])
            rc = numpy.hstack([rc, r[:-1]])
        #
        j = 1
        while j < len(rc)-1:
            n = len(rc)
            i = (j-1)%n
            if norm2(xyc[j] - xyc[i]) < rc[j] + rc[i] + spacing:
                xyc = numpy.delete(xyc, j, 0)
                rc = numpy.delete(rc, j)
            else:
                j += 1
        #
        if norm2(xyc[-2] - xyc[-1]) < rc[-2] + rc[-1] + spacing:
            xyc = numpy.delete(xyc, -2, 0)
            rc = numpy.delete(rc, -2)
        #
        if norm2(xyc[-2] - xyc[-1]) > rc[-2] + rc[-1] + 2*spacing:
            spacing += 1e-3
            #delta = (norm2(xyc[-2] - xyc[-1]) - (rc[-2] + rc[-1] + spacing))/len(rc)
            #spacing += min(1e-3, delta)
            continue
        else:
            print '\tconverged, %d points' % len(xyc)
            break
    return [xyc, rc]
#####################################




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

#####################################
def read_connected_path(filename):
    segments = []
    f = open(filename, 'r')
    n = int(f.readline())
    for i in range(n):
        ni = int(f.readline())
        bp = numpy.zeros((ni,2))
        for j in range(ni):
            bp[j] = [float(x) for x in f.readline().split()]
        segments.append(Curve(bp))
    f.close()
    return segments
#####################################



def get_key_points(paths, TOLangleG1, nsample=0, radfun=None):
    # type: 0: G1 discontinuity, 1: alpha (locally) maximally negative
    if nsample > 0 and radfun is not None: tsample = numpy.linspace(0,1,nsample)
    key_pts = []
    for path in paths:
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
            if nsample > 0 and radfun is not None:
                a = path[icurve].alpha(tsample, radfun=radfun)
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
            #
        key_pts.append(key_pts_i)
    return key_pts

#######################################################
def make_eoc_eod(paths, radfun, npts=100):
    t = numpy.linspace(0,1,npts)
    #
    # ENVELOPPES (PROPRES) DES CERCLES
    for path in paths:
        for curve in path:
            curve.make_envelope_propre(t, radfun)
    #
    # PREPARE ENVELOPES OF CIRCLES/DISKS
    arcs = []
    for ipath, path in enumerate(paths):
        ncurves = len(path)
        for icurve, curvei in enumerate(path):
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
                # DETERMINE LOCAL CONVEXITY
                sgn = dxyi[0]*dxyj[1] - dxyi[1]*dxyj[0]
                if sgn < 0:
                    # add arc on side 1 to eod...
                    arc.eod[1] = [arc.eoc[1].copy()]
                else:
                    # add arc on side 0 to eod...
                    arc.eod[0] = [arc.eoc[0].copy()]
    #
    # INSERT ARCS (PSEUDO-EDS DES COINS)
    for i in range(len(arcs)):
        ipath, icurve, arc = arcs[i]
        paths[ipath].insert(icurve+1, arc)
        for j in range(i+1,len(arcs)):
            if arcs[j][0] == ipath:
                arcs[j][1] += 1
    #
    # (GLOBAL) EOC
    eoc = []
    for path in paths:
        eoc_path = [[], []]
        for curve in path:
            for iside, xy in enumerate(curve.eoc):
                eoc_path[iside].append(xy.copy())
        eoc.append(eoc_path)
    #
    # (GLOBAL) EOD (REMOVE INTERSECTIONS)
    eod = []
    for path in paths:
        eod_path = [[], []]
        for curve in path:
            for iside, eod_side in enumerate(curve.eod):
                for xy in eod_side:
                    eod_path[iside].append(xy)
        eod.append(eod_path)
    #
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
    #
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
    #
    return paths, eoc, eod, xy_inter
