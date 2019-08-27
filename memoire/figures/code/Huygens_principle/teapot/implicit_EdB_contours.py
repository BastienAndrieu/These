# -*- coding: utf-8 -*-
ROOT = '/d/bandrieu/'#'/home/bastien/'#

import numpy
from numpy import sin, cos
import copy 

import matplotlib.pyplot as plt

import sys
sys.path.append(ROOT + 'GitHub/Code/Python/')
import lib_bezier as lbez
sys.path.append(ROOT + 'GitHub/These/memoire/figures/code/Huygens_principle/')
from lib_huygens import *
sys.path.append(ROOT + 'GitHub/These/memoire/figures/code/')
from lib_py2tikz import *

#####################################
def implicit_sphere_function(xyc, r, x, y):
    return (x - xyc[0])**2 + (y - xyc[1])**2 - r**2
#####################################

#####################################
def radius_function(x, y):
    a = 0.08*tau
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
paths = [read_connected_path('teapot_simple_'+name+'_bcp.dat')
         for name in ['outer', 'inner']]#for name in ['outer']]#
########################################################


########################################################
# SAMPLE PATHS
t = numpy.linspace(0,1,100)

xy = numpy.empty((0,2))
for path in paths:
    for curve in path:
        xyc = curve.eval(t)
        xy = numpy.vstack((xy, xyc[:-1]))

tau = 2
r, dr_dx, dr_dy, d2r_dx2, d2r_dy2, d2r_dxdy = radius_function(xy[:,0], xy[:,1])
########################################################

########################################################
# DOMAIN GRID
xymin = numpy.amin(xy, axis=0)
xymax = numpy.amax(xy, axis=0)

xymid = 0.5*(xymax + xymin)
xyrng = 0.5*(xymax - xymin)

extend = 20

npts = 200
h = 2*numpy.amin(xyrng)/float(npts - 1)
Y, X = numpy.meshgrid(
    numpy.arange(xymin[1]-extend*h, xymax[1]+(extend+1)*h, h),
    numpy.arange(xymin[0]-extend*h, xymax[0]+(extend+1)*h, h),
    indexing='ij'
)
########################################################

########################################################
# IMPLICIT FUNCTION SAMPLING
F = 1e6*numpy.ones(X.shape)
n = len(r)
for i in range(n):
    F = numpy.minimum(
        F,
        implicit_sphere_function(xy[i], r[i], X, Y)
    )
########################################################

########################################################
# VISUALIZE CONTOURS
t = numpy.linspace(0,1,100)

fig, ax = plt.subplots()

for ipath, path in enumerate(paths):
    for curve in path:
        xy = curve.eval(t)
        ax.plot(xy[:,0], xy[:,1], 'k-')

M = numpy.amax(numpy.absolute(F))
#lv = M*numpy.linspace(0,1,8)**2
eps = 1e-3
lv = 2*M*numpy.array([-1, 0, 1])
#lv = numpy.hstack([-lv[-1::-1], lv[1:]])
#lv = numpy.hstack([-M, lv])
  
C = ax.contourf(X, Y, F, cmap='RdBu_r', vmin=-M*eps, vmax=M*eps, levels=lv)
ax.contour(C, colors='k', linestyles='dashed')

                
cbar = plt.colorbar(C)

ax.set_aspect('equal')
plt.tight_layout()
plt.show()
########################################################


"""
########################################################
eods = []
tau = 0.5
step_tau = 0.5
for istep in range(3):
    print 'step #%d' % (istep+1)
    paths, eoc, eod, xy_inter = make_eoc_eod(
        paths=copy.deepcopy(paths),
        radfun=radius_function,
        npts=100
    )
    eods.append(eod)
    tau += step_tau
########################################################


########################################################
# VISUALIZE
cl = ['r', 'b']
t = numpy.linspace(0,1,100)

fig, ax = plt.subplots()

for ipath, path in enumerate(paths):
    for curve in path:
        xy = curve.eval(t)
        ax.plot(xy[:,0], xy[:,1], 'k-')

for eod in eods:
    for path in eod:
        for iside in range(2):
            for xy in path[iside]:
                ax.plot(xy[:,0], xy[:,1], color=cl[iside])

ax.set_aspect('equal')
plt.tight_layout()
plt.show()
"""
