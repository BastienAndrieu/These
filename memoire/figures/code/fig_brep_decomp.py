import sys
import numpy

############################################
class Face:
    def __init__(self, outer, inner, index):
        self.outer = outer
        self.inner = inner
        self.index = index
############################################
class Curve:
    def __init__(self, xyz, uv):
        self.xyz = xyz
        self.uv = uv
############################################
class Halfedge:
    def __init__(self, face, orig, twin, prev, next, ihyp):
        self.face = int(face)
        self.orig = int(orig)
        self.twin = int(twin)
        self.prev = int(prev)
        self.next = int(next)
        self.ihyp = int(ihyp)
############################################
class Hyperedge:
    def __init__(self, edges, index):
        self.edges = edges
        self.index = index
############################################
class Wire:
    def __init__(self, edges):
        self.edges = edges
############################################
class Vertex:
    def __init__(self, xyz, edge, index):
        self.xyz = xyz
        self.edge = edge
        self.index = index
############################################
def make_wire(head, halfedges, vlist, verbose=False):
    wire = []
    ih = head
    while True:
        if verbose: print(ih, halfedges[ih].orig in vlist)
        if halfedges[ih].orig in vlist:
            start = ih
            if verbose: print(start)
            break
        ih = halfedges[ih].next
        if ih == head: break
    while True:
        wire.append(ih)
        ih = halfedges[ih].next
        if ih == start:
            if verbose: print(wire)
            return Wire(edges=wire)
############################################
def make_wire_curves(wire, edges, curves):
    xyz = numpy.empty((0,3))
    uv  = numpy.empty((0,2))
    for ie in wire.edges:
        ic = int(ie/2)
        if ie%2 == 0:
            xyz = numpy.vstack((xyz, curves[ic].xyz[:-1]))
            uv  = numpy.vstack((uv,  curves[ic].uv[:-1,2:4]))
        else:
            xyz = numpy.vstack((xyz, curves[ic].xyz[:0:-1]))
            uv  = numpy.vstack((uv,  curves[ic].uv[:0:-1,0:2]))
    return Curve(xyz=xyz, uv=uv )
############################################
def query_polyline_abscissa(p, fsq=0.5):
    n = len(p)
    s = numpy.zeros(n)
    for i in range(1,n):
        s[i] = s[i-1] + numpy.sqrt(numpy.sum(numpy.power(p[i] - p[i-1],2)))
    smid = fsq*s[-1]
    for i in range(n-1):
        if s[i] <= smid and s[i+1] >= smid:
            ds = s[i+1] - s[i]
            l = (smid - s[i])/ds
            pm = (1.0 - l)*p[i] + l*p[i+1]
            tm = (p[i+1] - p[i])/ds
            return pm, tm
############################################
pthin = '/d/bandrieu/GitHub/FFTsurf/test/demo_EoS_brep/'
pthout = '/d/bandrieu/GitHub/These/memoire/figures/data/fig_brep_faces/'

# list of faces
args = sys.argv
if len(args) < 2:
    ifaces = [0]
else:
    ifaces = []
    for a in args[1:]:
        ifaces.append(int(a)-1)
nfaces = len(ifaces)

##### Process BREP topological data
# Vertices ###
f = open(pthin + 'debug/verts.dat','r')
verts = []
vlist = []
iv = -1
for l in f:
    iv += 1
    v = [float(a) for a in l.split()]
    verts.append(Vertex(xyz=numpy.asarray(v[0:3]),
                        edge=int(2*(v[3]-1) + v[4]-1),
                        index=-1))
    if len(v) > 7:# vertex incident to at least 3 faces
        vlist.append(iv)
f.close()

# Halfedges ###
h = numpy.loadtxt(pthin + 'debug/edges.dat', dtype=int) - 1
nh = len(h)
edges = []
nf = 0
for i, e in enumerate(h):
    nf = max(nf, e[0]+1)
    if i%2 == 0:
        twin = i+1
    else:
        twin = i-1
    prev = 2*e[2] + e[3]
    next = 2*e[4] + e[5]
    edges.append(Halfedge(face=e[0],
                          orig=e[1],
                          twin=twin,
                          prev=prev,
                          next=next,
                          ihyp=-1))

# make Hyperedges ###
visited = numpy.zeros(nh, dtype=bool)
nhyp = 0
hypedg = []
for v in vlist:
    ih = verts[v].edge
    while True:
        if not visited[ih]:
            kh = ih
            el = []
            while True:
                el.append(kh)
                visited[kh] = True
                visited[edges[kh].twin] = True
                edges[kh].ihyp = nhyp
                edges[edges[kh].twin].ihyp = nhyp
                if edges[edges[kh].next].orig in vlist:
                    nhyp += 1
                    if el[0]%2 == 0:
                        hypedg.append(Hyperedge(edges=el,
                                                index=-1))
                    else:
                        for k in range(len(el)):
                            el[k] = edges[el[k]].twin
                        hypedg.append(Hyperedge(edges=el[::-1],
                                                index=-1))
                    break
                kh = edges[kh].next
        jh = edges[ih].prev # ingoing
        ih = edges[jh].twin # outgoing
        if ih == verts[v].edge: break


# Faces, Wires ###
f = open(pthin + 'debug/faces.dat','r')
faces = []
jf = 0
while jf < nf:
    he = [int(a)-1 for a in f.readline().split()]
    ih = 2*he[0] + he[1]
    wout = make_wire(ih, edges, vlist)
    winn = []
    ninner = int(f.readline())
    for i in range(ninner):
        he = [int(a)-1 for a in f.readline().split()]
        ih = 2*he[0] + he[1]
        winn.append(make_wire(ih, edges, vlist, False))
    faces.append(Face(outer=wout,
                      inner=winn,
                      index=-1))
    jf += 1
f.close()

################################
# Re-index faces, vertices and hyperedges
idv = []
idh = []
for j,i in enumerate(ifaces):
    f = faces[i]
    f.index = j
    l = []
    l.extend(f.outer.edges)
    for w in f.inner:
        l.extend(w.edges)
    for ie in l:
        e = edges[ie]
        ih = e.ihyp
        if hypedg[ih].index < 0:
            hypedg[ih].index = len(idh)
            idh.append(ih)
        iv = e.orig
        if iv in vlist:
            if verts[iv].index < 0:
                verts[iv].index = len(idv)
                idv.append(iv)

f = open(pthout + 'verts_id.dat', 'w')
for v in verts:
    f.write(str(v.index) + '\n')
f.close()

f = open(pthout + 'faces_id.dat', 'w')
for fa in faces:
    f.write(str(fa.index) + '\n')
f.close()

################################
# Process BREP geometric data
# Curves ###
fx = open(pthin + 'debug/edges_xyz.dat','r')
fu = open(pthin + 'debug/edges_uv.dat','r')
ne = int(fx.readline())
fu.readline()
curves = []
for ie in range(ne):
    np = int(fx.readline())
    fu.readline()
    xyz = numpy.zeros((np,3))
    uv = numpy.zeros((np,4))
    for i in range(np):
        xyz[i] = [float(a) for a in fx.readline().split()]
        uv[i]  = [float(a) for a in fu.readline().split()]
    curves.append(Curve(xyz=xyz,
                        uv=uv))
fx.close()

# Hypercurves ###
hypcurv = []
for i, h in enumerate(hypedg):
    xyz = numpy.empty((0,3))
    uv = numpy.empty((0,4))
    for j, ie in enumerate(h.edges):
        k = int(ie/2)
        x = curves[k].xyz
        u = curves[k].uv
        if j == len(h.edges)-1:
            xyz = numpy.vstack((xyz, x))
            uv  = numpy.vstack((uv, u))
        else:
            xyz = numpy.vstack((xyz, x[:-1]))
            uv  = numpy.vstack((uv, u[:-1]))
    hypcurv.append(Curve(xyz=xyz,
                         uv=uv))
################################
used = numpy.zeros(nhyp)
fracws = 0.07
for i in ifaces:
    strf = format(i+1, '03')
    k = 0
    cuvdat = numpy.empty((0,7))
    #cxyzdat = numpy.empty((0,7))
    f = faces[i]
    c = make_wire_curves(f.outer, edges, curves)
    numpy.savetxt(pthout+'contour_ext_'+strf+'.dat', c.uv)
    x, dx = query_polyline_abscissa(c.uv, fracws)
    fi = open(pthout+'contours_label_'+strf+'.dat', 'w')
    fi.write(str(x[0])  + ', ' +
             str(x[1])  + ', ' +
             str(dx[0]) + ', ' +
             str(dx[1]) + '\n')
    for j, w in enumerate(f.inner):
        c = make_wire_curves(w, edges, curves)
        numpy.savetxt(pthout+'contour_int_'+strf+'_'+str(j)+'.dat', c.uv)
        x, dx = query_polyline_abscissa(c.uv, (fracws+0.5)%1)
        fi.write(str(x[0])  + ', ' +
                 str(x[1])  + ', ' +
                 str(dx[0]) + ', ' +
                 str(dx[1]) + '\n')
    #
    ih = -1
    for ie in f.outer.edges:
        if edges[ie].ihyp == ih: continue
        ih = edges[ie].ihyp
        used[ih] += 1
        if ie%2 == 0:
            u = hypcurv[ih].uv[:,2:4]
            x = hypcurv[ih].xyz
        else:
            u = hypcurv[ih].uv[::-1,0:2]
            x = hypcurv[ih].xyz[::-1]
        um, dum = query_polyline_abscissa(u)
        xm, dxm = query_polyline_abscissa(x)
        k += 1
        numpy.savetxt(pthout+'curve_uv_'+strf+'_'+str(k)+'.dat', u)
        cuvdat = numpy.vstack([cuvdat,
                               numpy.hstack([um,
                                             dum,
                                             hypedg[ih].index+1,
                                             used[ih],
                                             0])])
        #cxyzdat = numpy.vstack([cuvdat,
        #                        numpy.hstack([xm,
        #                                      dxm,
        #                                      hypedg[ih].index+1])])
    for iw, w in enumerate(f.inner):
        ih = -1
        for ie in w.edges:
            if edges[ie].ihyp == ih: continue
            ih = edges[ie].ihyp
            used[ih] += 1
            if ie%2 == 0:
                u = hypcurv[ih].uv[:,2:4]
                x = hypcurv[ih].xyz
            else:
                u = hypcurv[ih].uv[::-1,0:2]
                x = hypcurv[ih].xyz[::-1]
            um, dum = query_polyline_abscissa(u)
            xm, dxm = query_polyline_abscissa(x)
            k += 1
            numpy.savetxt(pthout+'curve_uv_'+strf+'_'+str(k)+'.dat', u)
            cuvdat = numpy.vstack([cuvdat,
                               numpy.hstack([um,
                                             dum,
                                             hypedg[ih].index+1,
                                             used[ih],
                                             iw+1])])
            #cxyzdat = numpy.vstack([cuvdat,
            #                        numpy.hstack([xm,
            #                                      dxm,
            #                                      hypedg[ih].index+1])])
    #
    numpy.savetxt(pthout+'curve_uvdata_'+strf+'.dat',
                  cuvdat, delimiter=',')
    #numpy.savetxt(pthout+'curve_xyzdata_'+strf+'.dat',
    #              cxyzdat, delimiter=',')





############################################
"""
import matplotlib.pyplot as plt

scl = 0.1
fig, ax = plt.subplots(1,len(ifaces))
for j, i in enumerate(ifaces[0:2]):
    strf = format(i+1,'03')
    wlab = numpy.loadtxt(pthout+'contours_label_'+strf+'.dat', delimiter=',')
    print wlab
    if len(wlab) == 2:
        wlab = wlab[0]
    c = numpy.loadtxt(pthout+'contour_ext_'+strf+'.dat')
    ax[j].plot(c[:,0], c[:,1])
    a = wlab[0:2]
    v = scl*numpy.array((-wlab[3],wlab[2]))
    ax[j].plot(a[0], a[1], '*')
    ax[j].plot([a[0], a[0]+v[0]], [a[1], a[1]+v[1]], '-')
    ax[j].set_aspect('equal')
plt.show()
"""

"""
import bpy
import bpy_extras
import numpy
import sys
sys.path.append('/stck/bandrieu/Bureau/Python/mylibs/')
sys.path.append('/d/bandrieu/GitHub/These/memoire/figures/code/')
import lib_color as mycolors
import my_lib as myl
import visibility
import lib_fig

pth = '/d/bandrieu/GitHub/FFTsurf/test/demo_EoS_brep/'

############################################
scene = bpy.context.scene
myl.blankScene()

## Set scene   
scene.render.resolution_x = 800
scene.render.resolution_y = scene.render.resolution_x
scene.render.resolution_percentage = 100

bpy.data.worlds["World"].horizon_color = [1,1,1]
bpy.data.worlds["World"].light_settings.samples = 10

cam = scene.camera
cam.location = numpy.array([2.102,1.798,1.104])
cam.rotation_euler = numpy.array((66.7,0.778,132.2))*numpy.pi/180.0
bpy.data.cameras["Camera"].angle = 37.72*numpy.pi/180.0

## Lighting
# Environment lighting
scene.world.light_settings.use_environment_light = True
scene.world.light_settings.environment_energy = 0.3
scene.world.light_settings.environment_color = 'PLAIN'

# Create new lamp datablock
lamp_data = bpy.data.lamps.new(name="KeyLight", type='POINT')
lamp_data.energy = 1.2
lamp_data.shadow_method = 'RAY_SHADOW'
lamp_data.shadow_ray_samples = 10
lamp_data.shadow_soft_size = 2.0

# Create new object with our lamp datablock
lamp_object = bpy.data.objects.new(name="KeyLight", object_data=lamp_data)

# Link lamp object to the scene so it'll appear in this scene
scene.objects.link(lamp_object)

lamp_object.location = [3.75,1.65,3.20]

# And finally select it make active
lamp_object.select = True
scene.objects.active = lamp_object
############################################


# list of faces
args = sys.argv
if len(args) < 4:
    ifaces = [0]
else:
    ifaces = []
    for a in args[3:]:
        ifaces.append(int(a)-1)
nfaces = len(ifaces)

### import meshes
bpy.ops.import_scene.obj(filepath=pth+'mesh/mesh.obj',
                         axis_forward='Y', axis_up='Z')

obj = bpy.data.objects['mesh']
myl.setSmooth(obj)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.view3d.camera_to_view_selected() # set camera to fit surf in screen
bpy.data.cameras["Camera"].angle += numpy.pi/180.0 # increase FOV angle by 1 degree
bpy.ops.object.select_all(action='DESELECT')

nf = len(bpy.data.materials)-1
clf = numpy.loadtxt('/d/bandrieu/GitHub/These/memoire/figures/code/demo_EoS_brep_palette_modif.dat')
clf = mycolors.cc_hsv(clf, fs=1.2, fv=1.0)
clh = mycolors.cc_hsv(clf, fs=0.85, fv=0.62)#fs=0.75, fv=0.45)

### set face materials
for i in range(nf):
    mat = bpy.data.materials["mat"+str(i+1)]
    mat.diffuse_color = clf[i%len(clf)]
    mat.diffuse_intensity = 1.0
    mat.specular_intensity = 0.1
    mat.specular_hardness = 8.0


################
step = 20
niso = 6
m = step*niso + 1
u = numpy.linspace(-1.0,1.0,m)

meshes = []
for i in ifaces:
    strf = format(i+1,'03')
    c = myl.readCoeffs(pth + 'brepmesh/c_' + strf + '.cheb')
    surf = myl.addTensorProductPatch(numpy.polynomial.chebyshev.chebgrid2d(u, u, c[:,:,0]),
                                     numpy.polynomial.chebyshev.chebgrid2d(u, u, c[:,:,1]),
                                     numpy.polynomial.chebyshev.chebgrid2d(u, u, c[:,:,2]),
                                     name="surf_"+strf,
                                     location=[0,0,0],
                                     smooth=True,
                                     color=clf[i],
                                     alpha=0)
    meshes.append(bpy.data.meshes["surf_"+strf])
    mat = bpy.data.materials["mat_surf_"+strf]
    mat.use_raytrace = False
    mat.use_shadows = False
    mat.use_cast_buffer_shadows = False

lib_fig.fit_camera_to_meshes(meshes)
bpy.data.cameras["Camera"].sensor_width *= 1.01

"""

