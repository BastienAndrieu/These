import numpy 
import matplotlib.pyplot as plt
import sys

####################################################################
def make_box(center, ranges, hole=False):
    verts = numpy.zeros((4,2))
    for j in range(2):
        for i in range(2):
            verts[2*j+i,0] = center[0] - ranges[0]*(-1)**i
            verts[2*j+i,1] = center[1] - ranges[1]*(-1)**j
    edges = numpy.array([[0,1],[1,3],[3,2],[2,0]])
    if hole: edges = numpy.flip(edges, axis=1)
    return verts, edges
####################################################################
def is_inside_polygon(point, verts, edges):
    M = 10.0
    a = 0.5*numpy.pi*numpy.random.rand()
    c = numpy.cos(a)
    s = numpy.sin(a)
    uvb = point + M*numpy.array([c, s])

    inside = False
    for e in edges:
        v1 = verts[e[0]]
        v2 = verts[e[1]]
        if (((v1[0] - point[0])*s > (v1[1] - point[1])*c) != ((v2[0] - point[0])*s > (v2[1] - point[1])*c)) and ((point[0] - v1[0])*(v1[1] - v2[1]) > (point[1] - v1[1])*(v1[0] - v2[0])) != ((uvb[0] - v1[0])*(v1[1] - v2[1]) > (uvb[1] - v1[1])*(v1[0] - v2[0])):
            inside = not inside
    return inside
####################################################################
def distance_from_edge(point, v):
    t = v[1] - v[0]
    lt = numpy.hypot(t[0], t[1])
    t = t/lt
    q = point - v[0]
    qt = numpy.dot(q, t)
    if qt < 0.:
        dist = numpy.hypot(q[0], q[1])
    elif qt > lt:
        dist = numpy.hypot(point[0] - v[1,0], point[1] - v[1,1])
    else:
        dist = numpy.sqrt(q[0]**2 + q[1]**2 - qt**2)
    return dist
####################################################################
def minimum_distance_from_boundary(point, verts, edges):
    dist = 1e9
    for e in edges:
        v = verts[e]
        t = v[1] - v[0]
        lt = numpy.hypot(t[0], t[1])
        t = t/lt
        q = point - v[0]
        qt = numpy.dot(q, t)
        if qt < 0.:
            dist = min(dist, numpy.hypot(q[0], q[1]))
        elif qt > lt:
            dist = min(dist, numpy.hypot(point[0] - v[1,0], point[1] - v[1,1]))
        else:
            dist = min(dist, numpy.sqrt(q[0]**2 + q[1]**2 - qt**2))
    return dist
####################################################################
def sample_domain(verts, edges, samples=21):
    xymin, xymax = get_bounding_box(verts, xymrg=0.5/float(samples - 1))
    xy = numpy.zeros((samples, samples, 2))
    x = numpy.linspace(xymin[0], xymax[0], samples)
    y = numpy.linspace(xymin[1], xymax[1], samples)
    for j in range(samples):
        for i in range(samples):
            xy[i,j,0] = x[i]
            xy[i,j,1] = y[j]
    return xy.reshape((samples*samples,2))
####################################################################
def get_bounding_box(verts, xymrg=0.):
    xymin = numpy.amin(verts, axis=0)
    xymax = numpy.amax(verts, axis=0)
    xyrng = xymax - xymin
    xymin = xymin - xymrg*xymrg
    xymax = xymax + xymrg*xymrg
    return xymin, xymax
####################################################################
def get_point_of_inaccessibility_MC(verts,
                                    edges,
                                    minimum_accuracy,
                                    shrink_factor=0.5/numpy.sqrt(2.),
                                    max_consecutive_misses=10):
    xymin, xymax = get_bounding_box(verts)
    accuracy = numpy.amin(xymax - xymin)
    maximin_distance = 0.
    while accuracy > minimum_accuracy:
        # begin loop through nodes
        consecutive_misses = 0
        while consecutive_misses < max_consecutive_misses:
            # select coordinates at random within bounds
            while True:
                xy = xymin + numpy.random.rand(2)*(xymax - xymin)
                # keep only if inside the domain
                if is_inside_polygon(xy, verts, edges): break

            # loop through edges and find shortest distance
            smallest_distance = minimum_distance_from_boundary(xy, verts, edges)
            
            # maximize the minimum distance through iterations
            # and keep track of the consecutive number of times
            # that a smallest distance has not been found
            if smallest_distance > maximin_distance:
                PIA = xy
                maximin_distance = smallest_distance
                consecutive_misses = 0
            else:
                consecutive_misses += 1

            # calculate current level of accuracy based on the
            # smallest distance between the upper and lower bound
            xyrng = xymax - xymin
            accuracy = numpy.amin(xyrng)

            # update the bounds of the region
            xymin = PIA - xyrng*shrink_factor
            xymax = PIA + xyrng*shrink_factor
    return PIA, maximin_distance
####################################################################

args = sys.argv
if len(args) < 2:
    iface = 1
else:
    iface = int(args[1])

if len(args) < 3:
    samples = 21
else:
    samples = int(args[2])




pth = '/d/bandrieu/GitHub/FFTsurf/test/demo_EoS_brep/brepmesh/'

strf = format(iface, '03')
buv = numpy.loadtxt(pth + 'bpts_' + strf + '.dat')
bed = numpy.loadtxt(pth + 'bedg_' + strf + '.dat', dtype=int) - 1

wsep = 0.18
wrng = 0.167*numpy.array([2.,1.])
wdat = numpy.loadtxt('../data/fig_brep_faces/contours_label_' + strf + '.dat', delimiter=',')
if wdat.ndim == 1:
    wdat = numpy.reshape(wdat, [1,len(wdat)])
for w in wdat:
    ctr = w[0:2] + wsep*numpy.array([-w[3], w[2]])
    wuv, wed = make_box(ctr, 0.5*wrng, hole=True)
    bed = numpy.vstack([bed, wed+len(buv)])
    buv = numpy.vstack([buv, wuv])




fig, ax = plt.subplots()


for e in bed:
    ax.plot(buv[e,0], buv[e,1], 'b-')

dmax = 0.
xmax = numpy.zeros(2)
for x in sample_domain(buv, bed, samples):
    if is_inside_polygon(x, buv, bed):
        d = minimum_distance_from_boundary(x, buv, bed)
        if d > dmax:
            dmax = d
            xmax = x
        ax.plot(x[0], x[1], 'r.')
        d *= 10
        ax.text(x[0], x[1], "%.3s" %d)
t = numpy.linspace(0,2.*numpy.pi,100)
ax.plot(xmax[0], xmax[1], 'r*')
ax.plot(xmax[0] + dmax*numpy.cos(t), xmax[1] + dmax*numpy.sin(t), 'g-')
    
ax.set_aspect('equal')
plt.show()


#numpy.savetxt('../data/fig_brep_faces/face_uvlabel_' + strf + '.dat', xmax, delimiter=',')
f = open('../data/fig_brep_faces/face_uvlabel_' + strf + '.dat', 'w')
f.write(str(xmax[0]) + ', ' + str(xmax[1]))
f.close()


exit()






PIA, dist = get_point_of_inaccessibility_MC(buv,
                                            bed,
                                            1.e-3)
t = numpy.linspace(0,2.*numpy.pi,100)
fig, ax = plt.subplots()
for e in bed:
    ax.plot(buv[e,0], buv[e,1], 'b-')
ax.plot(PIA[0], PIA[1], 'r*')
ax.plot(PIA[0] + dist*numpy.cos(t), PIA[1] + dist*numpy.sin(t), 'g-')
ax.set_aspect('equal')
plt.show()

