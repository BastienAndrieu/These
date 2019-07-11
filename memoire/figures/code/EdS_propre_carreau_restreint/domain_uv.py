"""
1) Make TikZ code 
2) Make input files for meshgen
3) Run meshgen
"""
import os
import numpy
import matplotlib.pyplot as plt

import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python/')
import lib_bezier as lbez
import lib_chebyshev as lcheb


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

    def update(self):
        self.xt = lbez.diff(self.x)
        return
#####################################


#####################################
def wire_polygon(wire, n=100):
    t = numpy.linspace(0,1,n)
    xy = numpy.empty((0,2), dtype=float)
    for curve in wire:
        p = curve.eval(t)
        xy = numpy.vstack([xy, p[:-1]])
    return xy
#####################################
def run_meshgen(file_coef, file_bpts, file_bedg, file_info, file_tri, file_uv, file_xyz):
    cmd = '/stck/bandrieu/Bureau/MeshGen/./meshgen.out'
    cmd += ' ' + file_coef
    cmd += ' ' + file_bpts
    cmd += ' ' + file_bedg
    cmd += ' ' + file_info
    cmd += ' ' + file_tri
    cmd += ' ' + file_uv
    cmd += ' ' + file_xyz
    os.system(cmd)
    return
#####################################
def run_meshconv(file_tri, file_uv, file_xyz, file_output):
    cmd = 'python /d/bandrieu/GitHub/FFTsurf/meshconv.py'
    cmd += ' -f ' + file_tri
    cmd += ' -u ' + file_uv
    cmd += ' -x ' + file_xyz
    cmd += ' -o ' + file_output
    os.system(cmd)
    return
#####################################

pth = '../../data/EdS_propre_carreau_restreint/'


########################################################
# IMPORT BEZIER CURVES (INKSCAPE OUTPUT)
outer = []
f = open(pth + 'domaine_uv_bcp_outer.dat', 'r')
while True:
    line = f.readline()
    if ("" == line): break # end of file
    nbp = int(line)
    
    bp = numpy.zeros((nbp,2))
    for i in range(nbp):
        bp[i] = [float(x) for x in f.readline().split()]
    outer.append(Curve(bp))
f.close()

inner = []
f = open(pth + 'domaine_uv_bcp_inner.dat', 'r')
while True:
    line = f.readline()
    if ("" == line): break # end of file
    nbp = int(line)
    
    bp = numpy.zeros((nbp,2))
    for i in range(nbp):
        bp[i] = [float(x) for x in f.readline().split()]
    inner.append(Curve(bp))
f.close()
########################################################


########################################################
# ADJUST (FLIP Y, RESCALE, CENTER)
t = numpy.linspace(0,1,100)
huge = 1e9
xymin = huge*numpy.ones(2)
xymax = -huge*numpy.ones(2)
for curve in outer + inner:
    curve.x[:,1] = -curve.x[:,1]
    curve.update()
    #
    xy = curve.eval(t)
    xymin = numpy.minimum(xymin, numpy.amin(xy, axis=0))
    xymax = numpy.maximum(xymax, numpy.amax(xy, axis=0))

xymid = 0.5*(xymin + xymax)
xyrng = 0.5*numpy.amax(xymax - xymin)
scale = 0.7/xyrng

for curve in outer + inner:
    for j in range(2):
        curve.x[:,j] = curve.x[:,j] - xymid[j]
    curve.x = curve.x*scale
    curve.update()
########################################################


########################################################
# PLOT UV DOMAIN
t = numpy.linspace(0,1,100)

fig, ax = plt.subplots()

for curve in outer + inner:
    xy = curve.eval(t)
    ax.plot(xy[:,0], xy[:,1])

ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_aspect('equal')
plt.show()
########################################################


########################################################
# WRITE TikZ CODE
# (...)
########################################################


########################################################
# MAKE meshgen INPUT FILES
uv_outer = wire_polygon(outer)
uv_inner = wire_polygon(inner)

n_outer = len(uv_outer)
n_inner = len(uv_inner)

edg_outer = [(i, (i+1)%n_outer) for i in range(n_outer)]
edg_inner = [(i, (i+1)%n_inner) for i in range(n_inner)]

file_bpts = pth + 'bpts.dat'
f = open(file_bpts, 'w')
for u, v in uv_outer:
    f.write('%s %s\n' % (u, v))
for u, v in uv_inner:
    f.write('%s %s\n' % (u, v))
f.close()

file_bedg = pth + 'bedg.dat'
f = open(file_bedg, 'w')
for a, b in edg_outer:
    f.write('%d %d\n' % (a+1, b+1))
for a, b in edg_inner:
    f.write('%d %d\n' % (a+n_outer+1, b+n_outer+1))
f.close()

hmin = 1e-3
hmax = 1e-2
tolchord = 1e-3
file_info = pth + 'info.dat'
f = open(file_info, 'w')
f.write('%s\n%s\n%s' % (hmin, hmax, tolchord))
f.close()


os.system('cp /d/bandrieu/GitHub/FFTsurf/test/demo_EoS_MAT/c_skeleton.cheb ' + pth + 'c_skeleton.cheb')

c = lcheb.read_polynomial2(pth+'c_skeleton.cheb')
c = lcheb.flip_polynomial2(c, flip_u=False, flip_v=True)
lcheb.write_polynomial2(c, pth+'c_skeleton.cheb')


os.system('cp /d/bandrieu/GitHub/FFTsurf/test/demo_EoS_MAT/surfnew_c_01.cheb ' + pth + 'c_eos.cheb')

print 'Running meshgen...'
run_meshgen(pth+'c_skeleton.cheb', file_bpts, file_bedg, file_info, pth+'tri0.dat', pth+'uv0.dat', pth+'xyz0.dat')
run_meshgen(pth+'c_eos.cheb',      file_bpts, file_bedg, file_info, pth+'tri1.dat', pth+'uv1.dat', pth+'xyz1.dat')
print 'done.'

print 'Running meshconv...'
run_meshconv(pth+'tri0.dat', pth+'uv0.dat', pth+'xyz0.dat', pth+'mesh0.obj')
run_meshconv(pth+'tri1.dat', pth+'uv1.dat', pth+'xyz1.dat', pth+'mesh1.obj')
print 'done.'

########################################################
