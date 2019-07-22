import numpy
from numpy import *
import matplotlib.pyplot as plt

import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python/')
import lib_bezier as lbez

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
def norm2(u):
    return sqrt(sum(u**2))
#####################################
def radius_function(x, y):
    """
    return r, dr_dx, dr_dy
    """
    a = 0.28
    b = 0.15
    c = 3
    d = 0.22
    e = 2
    r = a*(1 + b*cos(c*x) + d*sin(e*y))
    dr_dx = -a*b*c*sin(c*x)
    dr_dy = a*d*e*cos(e*x)
    return r, dr_dx, dr_dy
#####################################
def circle_arc_between_two_points(center, xy0, xy1, tolchord=1e-4):
    p0 = asarray(xy0) - asarray(center)
    p1 = asarray(xy1) - asarray(center)
    p090d = array([-p0[1], p0[0]])

    angle = arctan2(dot(p1, p090d), dot(p1, p0))
    if angle < 0: angle += 2*pi
    #angle = arccos(min(1.0, max(-1.0, dot(p0,p1)/sqrt(dot(p0,p0)*dot(p1,p1)))))

    npts = max(2, int(0.5*angle/sqrt(tolchord*(2-tolchord))))
    
    t = linspace(0,1,npts)
    p = zeros((npts,2))
    #p = (outer(sin((1 - t)*angle), p0) + outer(sin(t*angle), p1))/sin(angle)
    for i in range(npts):
        ai = angle*t[i]
        p[i] = center + cos(ai)*p0 + sin(ai)*p090d
    return p
#####################################

pth = '../../data/pseudo_EdS_arete/'

########################################################
# IMPORT BEZIER CURVES (INKSCAPE OUTPUT)
nappes = []
f = open(pth + 'vue_coupe_bcp.dat', 'r')
while True:
    line = f.readline()
    if ("" == line): break # end of file
    nbp = int(line)
    
    bp = numpy.zeros((nbp,2))
    for i in range(nbp):
        bp[i] = [float(x) for x in f.readline().split()]
    nappes.append(Curve(bp))
f.close()
########################################################

########################################################
# ADJUST (FLIP Y, SCALE DOWN, FIX SINGULAR ENDPOINTS)
SPT = 1e-4
HUGE = 1e9
xymin = HUGE*numpy.ones(2)
xymax = -HUGE*numpy.ones(2)
for curve in nappes:
    curve.x[:,1] = -curve.x[:,1]
    if norm2(curve.x[0] - curve.x[1]) < SPT: curve.x = curve.x[1:]
    if norm2(curve.x[-1] - curve.x[-2]) < SPT: curve.x = curve.x[:-1]
    curve.update()
    xymin = numpy.minimum(xymin, numpy.amin(curve.x, axis=0))
    xymax = numpy.maximum(xymax, numpy.amax(curve.x, axis=0))

xymid = 0.5*(xymin + xymax)
xyrng = 0.5*numpy.amax(xymax - xymin)
scale = 1/xyrng

for curve in nappes:
    for j in range(2):
        curve.x[:,j] = curve.x[:,j] - xymid[j]
    curve.x = curve.x*scale
    curve.update()
########################################################



########################################################
# PLOT & WRITE TikZ CODE
#sidelabel = ['R','L']
sidelabel = ['Right','Left']

pthcode = '../../code/pseudo_EdS_arete/'
f = open(pthcode + 'vue_coupe_tikzcode.tex', 'w')

#f.write('{\\transparent{0.3}\n');
f.write('\\begin{scope}[blend group = overlay]\n')

colors = [
    (1,0,0),
    (0,1,0),
    (0,0,1)
]

npts = 40
t = numpy.linspace(0,1,npts)
q = zeros((npts,2))

fig, ax = plt.subplots()

zones = []
EdS = []
for inappe, curve in enumerate(nappes):
    #
    xy = curve.eval(t)
    ax.plot(xy[:,0], xy[:,1], 'k')
    #
    dxy_dt = curve.evald(t)
    sqr_norm_dxy_dt = sum(dxy_dt**2, axis=1)
    inv_sqr_norm_dxy_dt = 1/sqr_norm_dxy_dt
    normal = vstack([dxy_dt[:,1], -dxy_dt[:,0]]).T
    #
    r, dr_dx, dr_dy = radius_function(xy[:,0], xy[:,1])
    dr_dt = dr_dx*dxy_dt[:,0] + dr_dy*dxy_dt[:,1]
    #
    eoc = []
    for side in range(2):
        qn = sqrt(sqr_norm_dxy_dt - dr_dt**2)*(-1)**side
        for j in range(2):
            q[:,j] = (qn*normal[:,j] - dr_dt*dxy_dt[:,j])*r*inv_sqr_norm_dxy_dt
        eoc.append(xy + q)
    #
    EdS.append(eoc)
    # 
    if inappe == 0:
        xy0 = eoc[0][-1]
        xylabel = eoc[0][int(0.6*npts)]
    else:
        xy1 = eoc[0][0]
        xylabel = eoc[0][int(0.33*npts)]
    arc0 = circle_arc_between_two_points(
        xy[0],
        eoc[1][0],
        eoc[0][0]
    )
    arc1 = circle_arc_between_two_points(
        xy[-1],
        eoc[0][-1],
        eoc[1][-1]
    )
    zone = vstack([eoc[0][:-1], arc1[:-1], eoc[1][:0:-1], arc0[:-1]])
    f.write('\\draw[styleEdS, draw=colorContourEdSnappe%d, fill=colorInterieurEdSnappe%d] \n' % (inappe, inappe))
    for x, y in zone:
        f.write('(%s, %s) -- \n' % (x, y))
    f.write('cycle; \n')
    f.write('\\coordinate (EdS%d) at (%s, %s);\n' % (inappe, xylabel[0], xylabel[1]))
    ax.plot(zone[:,0], zone[:,1], color=colors[inappe])
    
        
o = 0.5*(nappes[0].x[-1] + nappes[1].x[0])
r, dr_dx, dr_dy = radius_function(o[0], o[1])
ax.add_artist(
    plt.Circle(
        o,
        r,
        ec=colors[2],
        fc=colors[2],
        alpha=0.2
    )
)

arc = circle_arc_between_two_points(
        o,
        xy0,
        xy1
)
ax.plot(arc[:,0], arc[:,1], color=colors[2], lw=2)
f.write('\\end{scope}\n')

f.write('\\draw[styleEdS, draw=colorContourEdSarete, fill=colorInterieurarete] (%s, %s) circle (%s);\n' % (o[0], o[1], 0.985*r))

#f.write('} % fin transparent\n'); # fin \transparent


factor = 1.02
xystart = o + factor*(xy0 - o)
astart = degrees(arctan2(xy0[1] - o[1], xy0[0] - o[0]))
astop = degrees(arctan2(xy1[1] - o[1], xy1[0] - o[0]))
f.write('\\draw[stylePseudoEdS] (%s, %s) arc (%s:%s:%s) ' % (xystart[0], xystart[1], astart, astop, factor*r))
f.write('node[above, pos=0.5, inner sep=2pt] {$\\pseudoEdS{\\Gamma}{\\rho}$};\n')

f.write('\\path[decoration={text along path, raise={1ex}, text color=colorContourEdSarete, text={{$\\implicitEdB{\\Gamma}$} {$=$} {$0$}{}}, text align={center}}, decorate] ')
#f.write('(%s, %s) circle (%s);\n' % (o[0], o[1], r))
astart = 225#-160
astop = 310#-210
xystart = o + r*array([cos(radians(astart)), sin(radians(astart))])
f.write('(%s, %s) arc (%s:%s:%s);\n' % (xystart[0], xystart[1], astart, astop, r))

sideanchor = ['west', 'east']
for inappe, curve in enumerate(nappes):
    b = curve.x
    pos = 0.5 - 0.15*(-1)**inappe
    #
    f.write('\\draw[\n')
    f.write('    styleNappe,\n')
    f.write('    postaction={\n')
    f.write('        decoration={\n')
    #f.write('            text along path, raise={1ex}, text={{$\\Sigma_%s$}{}}, text align=center, reverse path\n' % sidelabel[inappe])
    f.write('            text along path, raise={1ex}, text={{$\\%s{\\Sigma}$}{}}, text align=center, reverse path\n' % sidelabel[inappe])
    f.write('        },\n')
    f.write('        decorate\n')
    f.write('    }\n')
    f.write(']\n')
    f.write('(%s, %s) .. controls (%s, %s) and (%s, %s) .. (%s, %s) node[pos=%s, above] (labelNappe%d) {};\n' %
            (b[0][0], b[0][1],
             b[1][0], b[1][1],
             b[2][0], b[2][1],
             b[3][0], b[3][1],
             pos,
             inappe)
    )
    #
    #f.write('\\node[colorContourEdSnappe%d, inner sep=0.15\\imagewidth, below] at (labelNappe%d) {$\\implicitEdB{\\Sigma_%s} < 0$};\n' %
    f.write('\\node[colorContourEdSnappe%d, inner sep=0.15\\imagewidth, below] at (labelNappe%d) {$\\implicitEdB{\\%s{\\Sigma}} < 0$};\n' %
            (inappe,
             inappe,
             sidelabel[inappe])
    )
    #
    #f.write('\\path[decoration={text along path, raise={1ex}, text color=colorContourEdSnappe%d, text={{$\\implicitEdB{\\Sigma_%s}$} {$=$} {$0$}{}}, text align={center}}, decorate] \n' %
    f.write('\\path[decoration={text along path, raise={1ex}, text color=colorContourEdSnappe%d, text={{$\\implicitEdB{\\%s{\\Sigma}}$} {$=$} {$0$}{}}, text align={center}}, decorate] \n' %
            (inappe,
             sidelabel[inappe])
    )
    for i, xy in enumerate(EdS[inappe][0][::-1]):
        f.write('(%s, %s)' % (xy[0], xy[1]))
        if i < len(EdS[inappe][0]) - 1:
            f.write(' -- \n')
        else:
            f.write(';\n')

f.write('\\coordinate (Gamma) at (%s, %s);\n' % (o[0], o[1]))
f.write('\\fill[black] (Gamma) circle (1.2pt);\n')
f.write('\\node[above] at (Gamma) {$\\Gamma$};\n')

f.close()

ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_aspect('equal')
plt.show()
########################################################
