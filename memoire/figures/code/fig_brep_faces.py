import bpy
import bpy_extras
from mathutils import Vector

import numpy
import sys
sys.path.append('/stck/bandrieu/Bureau/Python/mylibs/')
sys.path.append('/d/bandrieu/GitHub/These/memoire/figures/code/')
import lib_color as mycolors
import my_lib as myl
import lib_fig

pth = '/d/bandrieu/GitHub/FFTsurf/test/demo_EoS_brep/'
pthout = '/d/bandrieu/GitHub/These/memoire/figures/data/fig_brep_faces/'

args = sys.argv
if len(args) < 4:
    iface = 0
else:
    iface = int(args[3])-1

strf = format(iface+1,'03')

############################################
class Face:
    def __init__(self, outer, inner):
        self.outer = int(outer)
        self.inner = inner
############################################
class Edge:
    def __init__(self, xyz, uv):
        self.xyz = xyz
        self.uv = uv
############################################
class Halfedge:
    def __init__(self, face, orig, twin, prev, next):
        self.face = int(face)
        self.orig = int(orig)
        self.twin = int(twin)
        self.prev = int(prev)
        self.next = int(next)

#################################################
scene = bpy.context.scene
myl.blankScene()
## Set scene
scene.render.resolution_x = 800
scene.render.resolution_y = scene.render.resolution_x
scene.render.resolution_percentage = 100

bpy.data.worlds["World"].horizon_color = [1,1,1]
bpy.data.worlds["World"].light_settings.samples = 10

cam = scene.camera
if False:
    print('a')
    cam.location = numpy.array([1.93412, 1.37517, 1.00552])
    cam.rotation_euler = numpy.array((65.565, 0.629, 128.129))*numpy.pi/180.0
    bpy.data.cameras["Camera"].angle = 44.*numpy.pi/180.0
elif True:
    print('b')
    cam.location = numpy.array([2.102,1.798,1.104])
    cam.rotation_euler = numpy.array((66.7,0.778,132.2))*numpy.pi/180.0
    bpy.data.cameras["Camera"].angle = 37.72*numpy.pi/180.0
else:
    print('c')
    cam.location = numpy.array([1.77634,1.49005,0.90784])
    cam.rotation_euler = numpy.array((66.7,0.778,132.2))*numpy.pi/180.0
    bpy.data.cameras["Camera"].angle = 50.*numpy.pi/180.0

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

#lamp_object.location = [4.07625, 1.00545, 5.90386]
#lamp_object.location = [3.65,1.874,2.216]
lamp_object.location = [3.75,1.65,3.20]

# And finally select it make active
lamp_object.select = True
scene.objects.active = lamp_object
#################################################
clf = numpy.loadtxt('/d/bandrieu/GitHub/These/memoire/figures/code/demo_EoS_brep_palette_modif.dat')
clf = mycolors.cc_hsv(clf, fs=1.2, fv=1.0)

tri = numpy.loadtxt(pth + 'brepmesh/tri_' + strf + '.dat', dtype=int)-1
xyz = numpy.loadtxt(pth + 'brepmesh/xyz_' + strf + '.dat', dtype=float)
#uv  = numpy.loadtxt(pth + 'brepmesh/uv_'  + strf + '.dat', dtype=float)


verts = []
for x in xyz:
    verts.append(list(x))
faces = []
for t in tri:
    faces.append([int(t[0]), int(t[1]), int(t[2])])

name = "face"
#create mesh and object
msh = bpy.data.meshes.new(name)
obj = bpy.data.objects.new(name, msh)
#set mesh location
obj.location = [0,0,0]
bpy.context.scene.objects.link(obj)
#create mesh from python data
msh.from_pydata(verts,[],faces)
msh.update(calc_edges=True)
# show mesh as smooth
obj.data.use_auto_smooth = 1
mypolys = msh.polygons
for q in msh.polygons:
    q.use_smooth = True
# material
mat = bpy.data.materials.new("mat_"+name)
mat.diffuse_color = clf[iface]
mat.diffuse_intensity = 1
mat.specular_intensity = 0
mat.specular_hardness = 30
mat.use_transparency = False
obj.data.materials.append(mat)


bpy.ops.object.select_all(action='DESELECT')
scene.objects.active = obj
obj.select = True
bpy.ops.group.create(name="face_group")

################################################
# Add untrimmed surface
step = 20
niso = 6
m = step*niso + 1
u = numpy.linspace(-1.0,1.0,m)
c = myl.readCoeffs(pth + 'init/coef/c_' + strf + '.cheb')
surf = myl.addTensorProductPatch(numpy.polynomial.chebyshev.chebgrid2d(u, u, c[:,:,0]),
                                 numpy.polynomial.chebyshev.chebgrid2d(u, u, c[:,:,1]),
                                 numpy.polynomial.chebyshev.chebgrid2d(u, u, c[:,:,2]),
                                 name="surf",
                                 location=[0,0,0],
                                 smooth=True,
                                 color=clf[iface],
                                 alpha=0)
mat = bpy.data.materials["mat_surf"]
mat.use_raytrace = False
mat.use_shadows = False
mat.use_cast_buffer_shadows = False

################################################
# Mark Freestyle Edges
for ob in scene.objects:
    ob.select = False
    
scene.objects.active = surf
surf.select = True
bpy.ops.view3d.camera_to_view_selected() # set camera to fit surf in screen
bpy.data.cameras["Camera"].angle += numpy.pi/180.0 # increase FOV angle by 1 degree
bpy.ops.group.create(name="surf_group")
mesh = bpy.data.meshes["surf"]

bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='DESELECT')
bpy.ops.mesh.select_mode(type="EDGE")
bpy.ops.object.mode_set(mode='OBJECT')

for e in mesh.edges:
    v1 = e.vertices[0]
    v2 = e.vertices[1]
    p = min(v1,v2)
    q = max(v1,v2)
    if q == p + m and (p%step) == 0 and p < m:
        e.select = True
    elif q == p + 1 and (p%(step*m)) == 0:
        e.select = True

bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.loop_multi_select(ring=False)
bpy.ops.mesh.mark_freestyle_edge(clear=False)
bpy.ops.object.mode_set(mode='OBJECT')
surf.select = False

################################################
# Freestyle
#scene.render.use_freestyle = True
freestyle = scene.render.layers.active.freestyle_settings
freestyle.use_smoothness = True

bpy.ops.scene.freestyle_lineset_add() #2

lineset = freestyle.linesets["LineSet"]
lineset.visibility = 'VISIBLE'#'RANGE'#

lineset = freestyle.linesets["LineSet 2"]
lineset.visibility = 'HIDDEN'
linestyle = bpy.data.linestyles["LineStyle.001"]
linestyle.use_dashed_line = True
linestyle.dash1 = 7
linestyle.gap1 = linestyle.dash1

bpy.ops.scene.freestyle_lineset_add() #3
bpy.ops.scene.freestyle_lineset_add() #4

lineset = freestyle.linesets["LineSet 3"]
lineset.visibility = 'VISIBLE'#'RANGE'#

lineset = freestyle.linesets["LineSet 4"]
lineset.visibility = 'HIDDEN'
linestyle = bpy.data.linestyles["LineStyle.003"]
linestyle.use_dashed_line = True
linestyle.dash1 = 12
linestyle.gap1 = 10


for lineset in freestyle.linesets[0:2]:
    lineset.select_silhouette = False#True#
    lineset.select_border = False
    lineset.select_crease = False
    lineset.select_edge_mark = True
    lineset.select_by_group = True
    lineset.group = bpy.data.groups["surf_group"]

for lineset in freestyle.linesets[2:4]:
    lineset.select_silhouette = False
    lineset.select_border = True
    lineset.select_crease = False
    lineset.select_edge_mark = False
    lineset.select_by_group = True
    lineset.group = bpy.data.groups["face_group"]

for linestyle in bpy.data.linestyles[0:2]:
    linestyle.thickness = 1.85#1.5
    linestyle.thickness_position = 'INSIDE'

for linestyle in bpy.data.linestyles[2:4]:
    linestyle.thickness = 4.5
    linestyle.thickness_position = 'CENTER'

for linestyle in bpy.data.linestyles:
    linestyle.color = [0,0,0]
    linestyle.caps = "ROUND"
    linestyle.geometry_modifiers["Sampling"].sampling = 0.1



################################################
# Vertices
f = open(pth+'debug/verts.dat','r')
iv = -1
nv = -1
vertdat = []
cxyz = cam.location
render_scale = scene.render.resolution_percentage / 100
render_w = int(scene.render.resolution_x * render_scale)
render_h = int(scene.render.resolution_y * render_scale)

vert_rad = 5.e-3
vert_ns = 32
vert_nr = vert_ns

vlist = []
for l in f:
    v = [float(a) for a in l.split()]
    iv += 1
    if len(v) > 7:
        if (iface + 1) not in v[5:]: continue
        nv += 1
        vlist.append(iv)
        #obj = myl.addEmpty(name="vertex_"+str(nv), location=v[0:3])
        bpy.ops.mesh.primitive_uv_sphere_add(location=v[0:3],
                                             segments=vert_ns,
                                             ring_count=vert_nr,
                                             size=vert_rad)
        obj = bpy.context.active_object
        co_2d = bpy_extras.object_utils.world_to_camera_view(scene,
                                                             cam,
                                                             obj.location)
        result, object, matrix, location, normal = scene.ray_cast(start=cxyz,
                                                                  end=v[0:3])
        #print(iv+1, obj.location, object, int(object == obj))
        visible = int(object == obj)
        vertdat.append([iv, co_2d.x, co_2d.y, visible])
        obj.hide = True
        obj.hide_render = True
f.close()

cshx = bpy.data.cameras["Camera"].shift_x
cshy = bpy.data.cameras["Camera"].shift_y
f = open(pthout + 'verts_' + format(iface+1,'03') + '.dat', 'w')
for v in vertdat:
    vx = v[1] - 2.*cshx
    vy = v[2] - 2.*float(render_w)/float(render_h)*cshy
    f.write(str(v[0]+1) + ', ' + str(vx) + ', ' + str(vy) + ', ' + str(v[3]) + '\n')
f.close()


################################################
# Edges
fx = open(pth+'debug/edges_xyz.dat','r')
fu = open(pth+'debug/edges_uv.dat','r')
ne = int(fx.readline())
fu.readline()
edges = []
for ie in range(ne):
    np = int(fx.readline())
    fu.readline()
    xyz = numpy.zeros((np,3))
    uv = numpy.zeros((np,4))
    for i in range(np):
        xyz[i] = [float(a) for a in fx.readline().split()]
        uv[i] = [float(a) for a in fu.readline().split()]
    edges.append(Edge(xyz=xyz, uv=uv))
fx.close()

h = numpy.loadtxt(pth+'debug/edges.dat', dtype=int) - 1
nh = len(h)
hedges = []
nf = 0
for i, e in enumerate(h):
    nf = max(nf, e[0]+1)
    if i%2 == 0:
        twin = i+1
    else:
        twin = i-1
    prev = 2*e[2] + e[3]
    next = 2*e[4] + e[5]
    hedges.append(Halfedge(face=e[0], orig=e[1], twin=twin, prev=prev, next=next))

f = open(pth+'debug/faces.dat','r')
faces = []
jf = 0
while jf < nf:
    he = [int(a)-1 for a in f.readline().split()]
    outer = 2*he[0] + he[1]
    ninner = int(f.readline())
    inner = []
    for i in range(ninner):
        he = [int(a)-1 for a in f.readline().split()]
        inner.append(2*he[0] + he[1])
    jf += 1
    faces.append(Face(outer=outer, inner=inner))
f.close()

####
# concaténer edges si besoin
Wextuv = numpy.empty((0,2))
polylines = []
polylinesuv = []
ih = faces[iface].outer
while True:
    if hedges[ih].orig in vlist:
        ih0 = ih
        break
    ih = hedges[ih].next
p = numpy.empty((0,3))
q = numpy.empty((0,2))
while True:
    if ih%2 == 0:
        xyz = edges[int(ih/2)].xyz
        uv = edges[int(ih/2)].uv[:,2:4]
    else:
        xyz = edges[int((ih-1)/2)].xyz[::-1]
        uv = edges[int((ih-1)/2)].uv[::-1,0:2]
    p = numpy.vstack((p,xyz))
    q = numpy.vstack((q,uv))
    Wextuv = numpy.vstack((Wextuv, uv[:-1]))
    ih = hedges[ih].next
    if hedges[ih].orig in vlist:
        polylines.append(p)
        polylinesuv.append(q)
        p = numpy.empty((0,3))
        q = numpy.empty((0,2))
    if ih == ih0: break
numpy.savetxt(pthout+'contour_ext_'+strf+'.dat', Wextuv)
for j, q in enumerate(polylinesuv):
    numpy.savetxt(pthout+'curve_uv_'+strf+'_'+str(j)+'.dat', q)

for i,inn in enumerate(faces[iface].inner):
    ih = inn
    while True:
        if hedges[ih].orig in vlist:
            ih0 = ih
            break
        ih = hedges[ih].next
    
    p = numpy.empty((0,3))
    q = numpy.empty((0,2))
    Wintuv = numpy.empty((0,2))
    while True:
        if ih%2 == 0:
            xyz = edges[int(ih/2)].xyz
            uv = edges[int(ih/2)].uv[:,2:4]
        else:
            xyz = edges[int((ih-1)/2)].xyz[::-1]
            uv = edges[int((ih-1)/2)].uv[::-1,0:2]
        p = numpy.vstack((p,xyz))
        q = numpy.vstack((q,uv))
        Wintuv = numpy.vstack((Wintuv, uv[:-1]))
        ih = hedges[ih].next
        if hedges[ih].orig in vlist:
            polylines.append(p)
            polylinesuv.append(q)
            p = numpy.empty((0,3))
            q = numpy.empty((0,2))
        if ih == ih0: break
    numpy.savetxt(pthout+'contour_int_'+strf+'_'+str(i+1)+'.dat', Wintuv)

f = open(pthout+'nuv_'+strf+'.dat', 'w')
f.write(str(len(polylinesuv)) + ', ' + str(len(faces[iface].inner)))
f.close()
for j, q in enumerate(polylinesuv):
    numpy.savetxt(pthout+'curve_uv_'+strf+'_'+str(j+1)+'.dat', q)

"""
# séparer en segments de visibilité constante
curv_vis = []
curv_hid = []
arrow_vis = []
arrow_hid = []
for ob in scene.objects:
    ob.select = False
for j, p in enumerate(polylines):
    obj = myl.addPolyline(p, thickness=1.e-3)
    scene.objects.active = obj
    obj.select = True
    bpy.ops.object.convert(target='MESH')
    for i, x in enumerate(p):
        result, object, matrix, location, normal = scene.ray_cast(start=cxyz,
                                                                  end=x)
        vis = (object == obj)        
        if i == 0:
            if j == 0: visprev = vis
            segment = x
        elif i == len(p)-1:
            segment = numpy.vstack((segment,x))
            if visprev:
                arrow_vis.append(segment)
            else:
                arrow_hid.append(segment)
        else:
            segment = numpy.vstack((segment,x))
            if vis != visprev:
                if visprev:
                    curv_vis.append(segment)
                else:
                    curv_hid.append(segment)
                segment = x
                visprev = vis
    obj.hide = True
    obj.hide_render = True

def export_polyline_2d(p, filename):
    q = numpy.zeros((len(p),2))
    for i, x in enumerate(p):
        co = bpy_extras.object_utils.world_to_camera_view(scene,
                                                          cam,
                                                          Vector(x))
        q[i,0] = co.x - 2.*cshx
        q[i,1] = co.y - 2.*float(render_w)/float(render_h)*cshy
    numpy.savetxt(filename, q)
    return


#print('Curv VIS')
for i, p in enumerate(curv_vis):
    #if p.ndim > 1: myl.addPolyline(p, clr=[1,0,0], thickness=2.e-3)
    export_polyline_2d(p, pthout+'curve_vis_'+strf+'_'+str(i+1)+'.dat')

#print('Curv HID')
for i, p in enumerate(curv_hid):
    #if p.ndim > 1: myl.addPolyline(p, clr=[1,1,0], thickness=1.e-3)
    export_polyline_2d(p, pthout+'curve_hid_'+strf+'_'+str(i+1)+'.dat')

#print('Arrow VIS')
for i, p in enumerate(arrow_vis):
    #if p.ndim > 1: myl.addPolyline(p, clr=[0,1,0], thickness=2.e-3)
    export_polyline_2d(p, pthout+'arrow_vis_'+strf+'_'+str(i+1)+'.dat')

#print('Arrow HID')
for i, p in enumerate(arrow_hid):
    #if p.ndim > 1: myl.addPolyline(p, clr=[0,0,1], thickness=1.e-3)
    export_polyline_2d(p, pthout+'arrow_hid_'+strf+'_'+str(i+1)+'.dat')

def export_polyline_2d(p, filename):
    q = np.zeros((len(p),2))
    for i, x in enumerate(p):
        co = bpy_extras.object_utils.world_to_camera_view(scene,
                                                         cam,
                                                         [x[0],x[1],x[2]])
        q[i,0] = co.x - 2.*cshx
        q[i,1] = co.y - 2.*float(render_w)/float(render_h)*cshy
    numpy.savetxt(filename, q)
    return

f = open(pthout+'npolylines_'+strf+'.dat','w')
f.write(str(len(curv_vis))  + ', ')
f.write(str(len(curv_hid))  + ', ')
f.write(str(len(arrow_vis)) + ', ')
f.write(str(len(arrow_hid)))
f.close()
"""
################################################
# Renders

scene.render.alpha_mode = 'TRANSPARENT'
scene.render.image_settings.file_format='PNG'
pthimg = '/d/bandrieu/GitHub/These/memoire/figures/fig_brep_faces/'


### 1st render: face
scene.render.use_freestyle = False
bpy.data.objects["surf"].hide_render = True
scene.render.filepath = pthimg + 'face_' + format(iface+1,'03')
bpy.ops.render.render( write_still=True )

### 2nd render: visible borders
freestyle.linesets["LineSet 4"].show_render = False
scene.render.use_freestyle = True
mat = bpy.data.materials["mat_face"]
mat.use_transparency = True
mat.alpha = 0
mat.transparency_method = 'MASK'
mat.use_raytrace = False
mat.use_shadows = False
mat.use_cast_buffer_shadows = False

scene.render.filepath = pthimg + 'edges_vis_' + format(iface+1,'03')
bpy.ops.render.render( write_still=True )

### 3rd render: hidden borders
freestyle.linesets["LineSet 3"].show_render = False
freestyle.linesets["LineSet 4"].show_render = True
scene.render.filepath = pthimg + 'edges_hid_' + format(iface+1,'03')
bpy.ops.render.render( write_still=True )

### 4th render: surface
bpy.data.objects["surf"].hide_render = False
bpy.data.objects["face"].hide_render = True
scene.render.filepath = pthimg + 'surface_' + format(iface+1,'03')
bpy.ops.render.render( write_still=True )

exit()
################################################

uv  = numpy.loadtxt(pth + 'brepmesh/uv_'  + strf + '.dat', dtype=float)

obj = bpy.data.objects['face']
msh = bpy.data.meshes['face']
for ob in scene.objects:
    ob.select = False
scene.objects.active = obj
obj.select = True
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.uv.unwrap()
bpy.ops.object.mode_set(mode='OBJECT')

for i, f in enumerate(msh.polygons):
    for j in range(f.loop_total):
        k = f.loop_start + j
        for l in range(2):
            obj.data.uv_layers.active.data[k].uv[l] = 0.5*(1.0 + uv[tri[i,j],l])

scene.render.use_bake_clear = True
scene.render.bake_margin = 0.0
scene.render.use_bake_selected_to_active = True

imuvsize = 100
bpy.ops.image.new(name="diffuse",
                  width=imuvsize,
                  height=imuvsize,
                  alpha=True,
                  generated_type='BLANK')
img = bpy.data.images["diffuse"]

bpy.ops.object.mode_set(mode='EDIT')

area = bpy.context.screen.areas[2]
area.type = 'IMAGE_EDITOR'
area.spaces.active.image = img

scene.render.filepath = pthimg + 'faceuv_' + format(iface+1,'03')
bpy.ops.object.bake_image()
img.save_render(filepath=pthimg+'faceuv_'+format(iface+1,'03')+'.png')

npx = img.size[0]*img.size[1]
rgba = numpy.zeros((npx,4))
for i in range(npx):
    for j in range(4):
        rgba[i,j] = img.pixels[4*i+j]

hsv = mycolors.rgb2hsv(rgba[:,0:3])
ipx = numpy.argmax(numpy.sqrt(hsv[:,1]**2 + hsv[:,2]**2)*rgba[:,3])
print(rgba[ipx], 255*rgba[ipx])

clr = 255*rgba[ipx,0:3]
f = open(pthout+'facecolor_'+strf+'.dat', 'w')
for i in range(3):
    f.write(str(int(clr[i])))
    if i < 2:
        f.write(', ')
f.close()


#exit()

