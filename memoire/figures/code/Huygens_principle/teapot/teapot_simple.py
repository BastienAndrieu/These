# -*- coding: utf-8 -*-

import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python/')
#sys.path.append('/home/bastien/GitHub/Code/Python/')
import lib_bezier as lbez



######################################################
# OPTIONS
SHARP_SPOUT = True#False#
######################################################






######################################################
def Newton_intersection(curves, t, itmax=30, tol=1e-6, verbose=False):
    deriv = [lbez.diff(b) for b in curves]
    for it in range(itmax):
        if verbose:print 'it. #%d, t = (%s, %s)' % (it, t[0], t[1])
        xy = [lbez.eval_bezier_curve(curves[i], t[i]) for i in range(2)]
        if verbose:
            for i in range(2):
                print '   xy_%d = (%s, %s)' % (i, xy[i][0], xy[i][1])
        res = xy[1] - xy[0]
        if verbose: print '   |res| = %s' % (numpy.hypot(res[0], res[1]))
        #
        if sum(res**2) < tol**2:
            return t
        #
        dxy = [lbez.eval_bezier_curve(deriv[i], t[i]) for i in range(2)]
        dxy[1] = -dxy[1]
        det = dxy[0][0]*dxy[1][1] - dxy[0][1]*dxy[1][0]
        if verbose: print'   |det| = %s\n' % (abs(det))
        if abs(det) < 1e-15:
            return None
        t[0] += (res[0]*dxy[1][1] - res[1]*dxy[1][0])/det
        t[1] += (dxy[0][0]*res[1] - dxy[0][1]*res[0])/det
    return None
######################################################
def Newton_bezier(t, b, ivar, val, itmax, tol):
    d = lbez.diff(b)
    for it in range(itmax):
        f = lbez.eval_bezier_curve(b, t) - val
        if abs(f[ivar]) < tol:
            return t
        df = lbez.eval_bezier_curve(d, t)
        if abs(df[ivar]) < 1e-15:
            return None
        dt = -f[ivar]/df[ivar]
        t += dt
    return None
######################################################
def bezier_extremal_values(b, ivar):
    d = lbez.diff(b)
    A = d[2][ivar] - 2*d[1][ivar] + d[0][ivar]
    if abs(A) < 1e-15:
        return [0.5*d[0][ivar]/(d[0][ivar] - d[1][ivar])]
    else:
        B = d[1][ivar] - d[0][ivar]
        C = d[0][ivar]
        sqrtdelta = numpy.sqrt(B**2 - A*C)
        t = []
        for sign in [-1,1]:
            ts = (sign*sqrtdelta - B)/A
            if ts >= 0 and ts <= 1: t.append(ts)
        if len(t) < 1:
            fig, ax = plt.subplots()
            u = numpy.linspace(0,1,100)
            xy = lbez.eval_bezier_curve(d, u)
            ax.plot(u, xy[:,ivar])
            ax.plot([0,1], [0,0], 'k--')
            ax.set_aspect('equal')
            plt.show()
        return t
######################################################
def export_connected_path(path, filename):
    f = open(filename, 'w')
    f.write('%d\n' % len(path))
    for curve in path:
        f.write('%d\n' % len(curve))
        for b in curve:
            f.write('%s %s\n' % (b[0], b[1]))
    f.close()
    return
######################################################

def joint(segm1, segm2, tension):
    if isinstance(tension, int) or isinstance(tension, float):
        tension = [tension, tension]

    b = numpy.zeros((4,2))
    
    b[0] = segm1[3]

    if tension[0] == 0:
        b[1] = (2*segm1[3] + segm2[0])/3.0
    else:
        b[1] = segm1[3] + tension[0]*(segm1[3] - segm1[2])

    if tension[1] == 0:
        b[2] = (segm1[3] + 2*segm2[0])/3.0
    else:
        b[2] = segm2[0] + tension[1]*(segm2[0] - segm2[1])
        
    b[3] = segm2[0]
    return b

######################################################

def insert_joint(path, after, tension=1):
    path.insert(
        after+1,
        joint(
            path[after],
            path[(after+1)%len(path)],
            tension
        )
    )
    return path

######################################################
######################################################

t = numpy.linspace(0,1,100)
k = int(len(t)/2)
b = numpy.zeros((4,2))

if False:
    fig, ax = plt.subplots()

    f = open('teapot_profile_bcp.dat', 'r')
    j = 0
    while True:
        j +=1 
        line = f.readline()
        if ("" == line): break # end of file
        b[0] = [float(a) for a in line.split()]
        for i in range(1,4):
            b[i] = [float(a) for a in f.readline().split()]
            line = f.readline()
            #
        p = lbez.eval_bezier_curve(b, t)
        ax.plot(p[:,0], p[:,1])
        ax.text(p[k,0], p[k,1], str(j))
    f.close()

    ax.set_aspect('equal')
    plt.show()



##############################
# READ
f = open('teapot_elems_profile_bcp.dat', 'r')
npaths = int(f.readline())
paths = []
for ipath in range(npaths):
    segms = []
    nsegms = int(f.readline())
    for isegm in range(nsegms):
        b = numpy.zeros((4,2))
        for i in range(4):
            b[i] = [float(a) for a in f.readline().split()]
        segms.append(b)
        f.readline()
    paths.append(segms)
f.close()
##############################



##############################
bpolys = []
polys = []
for path in paths:
    poly = numpy.empty((0,2))
    bpoly = numpy.empty((0,2))
    for i, segm in enumerate(path):
        xy = lbez.eval_bezier_curve(segm, t)
        poly = numpy.vstack([poly, xy])
        bpoly = numpy.vstack([bpoly, segm])
    polys.append(poly)
    bpolys.append(bpoly)
##############################


##############################
# EDIT
# body (right)
d = lbez.diff(paths[0][3])
t0 = 0.5*d[0][1]/(d[0][1] - d[1][1])

paths[0][3] = lbez.reparameterize_bezier_curve(paths[0][3], end=t0)
xy0 = paths[0][3][3]

t1 = Newton_bezier(0, paths[0][4], 1, xy0[1], 20, 1e-6)
paths[0][4] = lbez.reparameterize_bezier_curve(paths[0][4], start=t1)

paths[0] = insert_joint(paths[0], after=3, tension=0)

# body (left)
paths[0] = paths[0][:7]
for i in range(7):
    bsym = paths[0][6-i][-1::-1].copy()
    bsym[:,0] = -bsym[:,0]
    paths[0].append(bsym)

# spout
if SHARP_SPOUT:
    # sharp tip
    """
    #   find y-extremal value
    t2 = bezier_extremal_values(paths[1][1], 0)
    print 't2 = ', t2
    if len(t2) < 1:
        print 'error t2'
    else:
        paths[1][1] = lbez.reparameterize_bezier_curve(paths[1][1], end=t2[0])
    #
    t3 = bezier_extremal_values(paths[1][2], 1)
    print 't3 = ', t3
    if len(t3) < 1:
        print 'error t3'
    else:
        paths[1][2] = lbez.reparameterize_bezier_curve(paths[1][2], start=t3[0])
    #
    paths[1] = insert_joint(paths[1], after=1, tension=(0,1))
    """
    paths[1][1] = lbez.reparameterize_bezier_curve(paths[1][1], end=0.4)
    paths[1][2][0] = paths[1][1][-1]
        
else:
    # smooth tip
    b3 = 0.5*(paths[1][1][2] + paths[1][2][1])
    b3[1] += 0.01
    b2 = paths[1][1][2]
    paths[1][1][2] = b2
    paths[1][1][3] = b3
    
    paths[1][2][0] = b3
    paths[1][2][1] = b3 + 0.85*(b3 - b2)

    

# handle
if True:
    paths.append(paths[2][2:])
    paths[3] = paths[3][-1::-1]
    for i in range(len(paths[3])):
        paths[3][i] = paths[3][i][-1::-1]
paths[2] = paths[2][:2]
##############################

TOL = 1e-6
itmax = 20

# # # # # # # # # # #
b = paths[0][11][-1::-1].copy()
ti = Newton_intersection(
    [b, paths[3][0]],
    [0.5,0]
)
if ti is not None:
    b = lbez.reparameterize_bezier_curve(b, end=ti[0])
    paths[3][0] = lbez.reparameterize_bezier_curve(paths[3][0], start=ti[1])
paths[3].append(b)

    
# # # # # # # # # # #
ti = Newton_intersection(
    [paths[0][1], paths[1][0]],
    [0.5,0]
)
if ti is not None:
    paths[0][1] = lbez.reparameterize_bezier_curve(paths[0][1], end=ti[0])
    paths[1][0] = lbez.reparameterize_bezier_curve(paths[1][0], start=ti[1])

# # # # # # # # # # #
if SHARP_SPOUT:
    i1 = 3#4
else:
    i1 = 3
ti = Newton_intersection(
    [paths[1][i1], paths[0][2]],
    [0.9,0.5]
)
if ti is not None:
    paths[1][i1] = lbez.reparameterize_bezier_curve(paths[1][i1], end=ti[0])
    paths[0][2] = lbez.reparameterize_bezier_curve(paths[0][2], start=ti[1])

# # # # # # # # # # #
ti = Newton_intersection(
    [paths[0][11], paths[2][0]],
    [0,0]
)
if ti is not None:
    paths[0][11] = lbez.reparameterize_bezier_curve(paths[0][11], end=ti[0])
    paths[2][0] = lbez.reparameterize_bezier_curve(paths[2][0], start=ti[1])
    
# # # # # # # # # # #
ti = Newton_intersection(
    [paths[2][1], paths[0][12]],
    [1,0.3]
)
if ti is not None:
    paths[2][1] = lbez.reparameterize_bezier_curve(paths[2][1], end=ti[0])
    paths[0][12] = lbez.reparameterize_bezier_curve(paths[0][12], start=ti[1])

outer = paths[0][:2]
outer.extend(paths[1])
outer.extend(paths[0][2:12])
outer.extend(paths[2])
outer.extend(paths[0][12:])

inner = [curve[-1::-1] for curve in paths[3][-1::-1]]
##############################
# PLOT
control = False
indices = False
knots = False

fig, ax = plt.subplots()

for poly in polys:
    ax.plot(poly[:,0], poly[:,1], 'k-', lw=0.25)
"""
for poly in bpolys:
    ax.plot(poly[:,0], poly[:,1], 'r:o', lw=0.25)
"""

if False:
    for path in paths:
        poly = numpy.empty((0,2))
        for i, segm in enumerate(path):
            if knots: ax.plot(segm[[0,-1],0], segm[[0,-1],1], 'ko')
            if control: ax.plot(segm[:,0], segm[:,1], 'k.:')
            if indices: 
                xy = lbez.eval_bezier_curve(segm, 0.5)
                ax.text(xy[0], xy[1], str(i))
            xy = lbez.eval_bezier_curve(segm, t)
            poly = numpy.vstack([poly, xy])
        ax.plot(poly[:,0], poly[:,1], lw=1.5)
else:
    for path in [outer, inner]:
        for i, segm in enumerate(path):
            if knots: ax.plot(segm[[0,-1],0], segm[[0,-1],1], 'ko')
            if control: ax.plot(segm[:,0], segm[:,1], 'k.:')
            if indices:
                xy = lbez.eval_bezier_curve(segm, 0.5)
                ax.text(xy[0], xy[1], str(i))
            xy = lbez.eval_bezier_curve(segm, t)
            ax.plot(xy[:,0], xy[:,1], 'b', lw=1.5)
    
ax.set_aspect('equal')
plt.show()
##############################



##############################
# EXPORT
export_connected_path(
    outer,
    'teapot_simple_outer_bcp.dat'
)
export_connected_path(
    inner,
    'teapot_simple_inner_bcp.dat'
)

#for i, name in enumerate(['body', 'spout', 'handle_out']):
for i, name in enumerate(['body', 'spout', 'handle_out', 'handle_in']):
    export_connected_path(
        paths[i],
        'teapot_simple_'+name+'_bcp.dat'
    )
##############################


##############################
# EXPORT SVG
# flip y, shift and rescale
aabb = 1e10*numpy.ones((2,2))
aabb[1] = -aabb[1]
for path in [outer, inner]:
    for curve in path:
        aabbi = lbez.AABB_2d_bezier_curve(curve)
        aabb[0] = numpy.minimum(aabbi[0], aabb[0])
        aabb[1] = numpy.maximum(aabbi[1], aabb[1])
shift_x = -aabb[0][0]
shift_y = -aabb[1][1]

scale = 100
for path in [outer, inner]:
    for curve in path:
        curve[:,0] = scale*(curve[:,0] + shift_x )
        curve[:,1] = -scale*(curve[:,1] + shift_y)

width = scale*(aabb[1][0] - aabb[0][0])
height = scale*(aabb[1][1] - aabb[0][1])

f = open('teapot_simple_from_python.svg', 'w')
f.write('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n')
f.write('<svg\n')
f.write('  xmlns:dc="http://purl.org/dc/elements/1.1/"\n')
f.write('  xmlns:cc="http://creativecommons.org/ns#"\n')
f.write('  xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"\n')
f.write('  xmlns:svg="http://www.w3.org/2000/svg"\n')
f.write('  xmlns="http://www.w3.org/2000/svg"\n')
f.write('  xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"\n')
f.write('  xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape"\n')
f.write('  width="%s"\n' % width)
f.write('  height="%s"\n' % height)
f.write('  id="svg4200"\n')
f.write('  version="1.1"\n')
f.write('  inkscape:version="0.91 r13725"\n')
f.write('  sodipodi:docname="teapot_simple_from_python.svg">\n')

for path in [outer, inner]:
    f.write('  <path\n')
    f.write('    stroke="black"\n')
    f.write('    fill="transparent"\n')
    f.write('    style="fill:none"\n')
    f.write('    d="')
    f.write('M %s, %s ' % (path[0][0][0], path[0][0][1]))
    for i, curve in enumerate(path):
        if i == 0: f.write('C ')
        for bp in curve[1:]:
            f.write('%s, %s ' % (bp[0], bp[1]))
    f.write('z" />\n')
f.write('</svg>')
f.close()
##############################
