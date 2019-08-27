# -*- coding: utf-8 -*-
ROOT = '/d/bandrieu/'#'/home/bastien/'#

import numpy
from numpy import *

import matplotlib.pyplot as plt

import sys
sys.path.append(ROOT + 'GitHub/Code/Python/')
import lib_bezier as lbez
sys.path.append(ROOT + 'GitHub/These/memoire/figures/code/Huygens_principle/')
from lib_huygens import *
sys.path.append(ROOT + 'GitHub/These/memoire/figures/code/')
from lib_py2tikz import *


#####################################
def radius_function(x, y):
    a = 0.08
    b = 0.2
    c = 2
    d = 0.13
    e = 1.5
    f = 0.25
    ox = 0.1
    oy = 0.2
    f = 0.25
    sx = 2
    r = a*(1 + b*sin(c*x + ox) + d*cos(e*y + oy) + f*(x + sx))
    dr_dx = a*(b*c*cos(c*x + ox) + f)
    dr_dy = -a*d*e*sin(e*y + oy)
    d2r_dx2 = -a*b*c*c*sin(c*x + ox)
    d2r_dy2 = -a*d*e*e*cos(e*y + oy)
    d2r_dxdy = 0*x
    return r, dr_dx, dr_dy, d2r_dx2, d2r_dy2, d2r_dxdy
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
########################################################



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
# Find 'key' points along each path
# type: 0: G1 discontinuity, 1: alpha (locally) maximally negative
TOLangleG1 = 20.0 # degrees
key_pts = get_key_points(elems, TOLangleG1, nsample=0)

######

circles = []
for ipath in range(len(key_pts)):
    nkey = len(key_pts[ipath])
    xys = numpy.empty((0,2))
    rs = numpy.empty(0)
    for ikey in range(nkey):
        #print '%d/%d' % (ikey, nkey-1)
        ntmp = len(rs)
        jkey = (ikey+1)%nkey
        typi, icurve, ti, xyi = key_pts[ipath][ikey]
        typj, jcurve, tj, xyj = key_pts[ipath][jkey]
        #print '\t[%d, %d[ /%d' % (icurve, jcurve, len(elems[ipath])-1)
        if jcurve < icurve:
            list_curves = [k for k in range(icurve,len(elems[ipath]))] + [k for k in range(jcurve)]
        else:
            list_curves = [k for k in range(icurve,jcurve)]
        #
        xyc, rc = circle_packing(
            curves=[elems[ipath][k] for k in list_curves],
            radfun=radius_function,
            nsample=200,
            itmax=30
        )
        xys = numpy.vstack([xys, xyc[:-1]])
        rs = numpy.hstack([rs, rc[:-1]])
    circles.append([xys, rs])
########################################################


########################################################
elems, eoc, eod, xy_inter = make_eoc_eod(
    paths=elems,
    radfun=radius_function,
    npts=100
)
########################################################



########################################################
# CENTER AND RESCALE EVERYTHING
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

xy_inter = numpy.asarray(xy_inter)
for i in range(2):
    xy_inter[:,i] = (xy_inter[:,i] - ctr[i])*scale
########################################################



########################################################
# VISUALIZE
cl = ['r', 'b']

t = linspace(0,1,100)

fig, ax = plt.subplots(1,2)

for iax in range(2):
    for ipath, path in enumerate(elems):
        for curve in path:#
            xy = curve.eval(t)
            ax[iax].plot(xy[:,0], xy[:,1], 'k-')
        """
        for typ, icurve, tval, xy in key_pts[ipath]:
            p = (xy - ctr)*scale
            r, dr_dx, dr_dy, d2r_dx2, d2r_dy2, d2r_dxdy = radius_function(xy[0], xy[1])
            ax[iax].add_artist(
                plt.Circle(
                    p,
                    scale*r,
                    ec='y',
                    fc='y',
                    alpha=0.8
                )
            )
        """
        for i, xy in enumerate(circles[ipath][0]):
            p = (xy - ctr)*scale
            r = circles[ipath][1][i]
            ax[iax].add_artist(
                plt.Circle(
                    p,
                    scale*r,
                    ec='g',
                    fc='g',
                    alpha=0.4
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

"""
xy_inter = numpy.asarray(xy_inter)
for iax in range(2):
    ax[iax].plot(xy_inter[:,0], xy_inter[:,1], 'k.')
"""

for axe in ax:
    axe.set_aspect('equal')
plt.tight_layout()
plt.show()



########################################################
# TIKZ EXPORT

#
f = open('teapot_def_paths_tikzcode.tex', 'w')

f.write('\\def\pathSigma{')
f.write('%s ' % (bezier_path_to_string(elems[0])))
f.write('%s' % (bezier_path_to_string(elems[1])))
f.write('}\n')

f.write('\\def\pathEdSplus{')
for path in eoc:
    f.write('%s' % (closed_polyline_to_string(path[0])))
f.write('}\n')

f.write('\\def\pathEdSmoins{')
for path in eoc:
    f.write('%s' % (closed_polyline_to_string(path[1])))
f.write('}\n')

f.write('\\def\pathEdBplus{')
for path in eod:
    f.write('%s' % (closed_polyline_to_string(path[0])))
f.write('}\n')

f.write('\\def\pathEdBmoins{')
for path in eod:
    f.write('%s' % (closed_polyline_to_string(path[1])))
f.write('}\n')

f.close()
#


f = open('teapot_circles.dat', 'w')
for ipath in range(len(circles)):
    for i, xy in enumerate(circles[ipath][0]):
        p = (xy - ctr)*scale
        r = circles[ipath][1][i]
        f.write('%s, %s, %s\n' % (p[0], p[1], r*scale))
f.close()
########################################################
