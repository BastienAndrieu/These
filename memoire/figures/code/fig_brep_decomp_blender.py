import bpy
import bpy_extras
import numpy
import sys
sys.path.append('/stck/bandrieu/Bureau/Python/mylibs/')
sys.path.append('/d/bandrieu/GitHub/These/memoire/figures/code/')
import lib_color as mycolors
import my_lib as myl
import lib_fig
import visibility
import chebyshev_lib
from numpy.polynomial.chebyshev import chebval2d
############################################
class Vertex:
    def __init__(self, xyz, uv, visible, index):
        self.xyz = xyz
        self.uv = uv
        self.visible = visible
        self.index = index
#################################################
def norm2(a):
    return numpy.sqrt(numpy.sum(numpy.power(a,2)))
#################################################
def unitize(a):
    return a/norm2(a)
#################################################
pthin = '/d/bandrieu/GitHub/FFTsurf/test/demo_EoS_brep/'
pthout = '/d/bandrieu/GitHub/These/memoire/figures/data/fig_brep_faces/'

args = sys.argv
print(args)
if len(args) < 4:
    iface = 0
else:
    iface = int(args[3])-1

strf = format(iface+1,'03')

tolvis = 1.e-3
nrsvis = 32

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
cam.location = numpy.array([2.102,1.798,1.104])
cam.rotation_euler = numpy.array((66.7,0.778,132.2))*numpy.pi/180.0
cam.data.angle = 37.72*numpy.pi/180.0

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




#################################################
bpy.ops.import_scene.obj(filepath=pthin+'mesh/mesh.obj',
                         axis_forward='Y', axis_up='Z')
obj = bpy.data.objects['mesh']
bpy.ops.object.select_all(action='DESELECT')
obj.select = True
bpy.ops.view3d.camera_to_view_selected() # set camera to fit surf in screen
bpy.data.cameras["Camera"].angle += numpy.pi/180.0 # increase FOV angle by 1 degree
bpy.ops.object.delete()
#################################################




#################################################
clf = numpy.loadtxt('/d/bandrieu/GitHub/These/memoire/figures/code/demo_EoS_brep_palette_modif.dat')
clf = mycolors.cc_hsv(clf, fs=1.2, fv=1.0)

tri = numpy.loadtxt(pthin + 'brepmesh/tri_' + strf + '.dat', dtype=int)-1
xyz = numpy.loadtxt(pthin + 'brepmesh/xyz_' + strf + '.dat', dtype=float)

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
c = myl.readCoeffs(pthin + 'init/coef/c_' + strf + '.cheb')
cs, ct = chebyshev_lib.diff2(c)
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
# Face label
fid = numpy.loadtxt(pthout + 'faces_id.dat', dtype=int)

stlabel = numpy.loadtxt(pthout + 'face_uvlabel_' + strf + '.dat', delimiter=',')
xyzfacelabel = chebval2d(stlabel[0], stlabel[1], c)
myl.addEmpty('label', xyzfacelabel)

u,v = lib_fig.convert_3d_to_2d_coords(xyzfacelabel)

f = open(pthout + 'face_xyzlabel_' + strf + '.dat', 'w')
f.write(str(fid[iface]+1) + ', ' + str(u) + ', ' + str(v))
f.close()

################################################
# Mark Freestyle Edges
for ob in scene.objects:
    ob.select = False
scene.objects.active = surf
surf.select = True

if False:
    bpy.ops.view3d.camera_to_view_selected() # set camera to fit surf in screen
    cam.data.angle += numpy.pi/180.0 # increase FOV angle by 1 degree
else:
    lib_fig.fit_camera_to_meshes([bpy.data.meshes["surf"]])
    cam.data.sensor_width *= 1.01


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
# Freestyle settings
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


################################################

################################################


################################################
# Vertices
vid = numpy.loadtxt(pthout + 'verts_id.dat', dtype=int)

verts = []
vlist = []
f = open(pthin + 'debug/verts.dat','r')
for i in range(len(vid)):
    v = [float(a) for a in f.readline().split()]
    if len(v) > 7 and (iface+1) in v[5:]:
        vlist.append(i)
        vxyz = numpy.asarray(v[0:3])
        u,v = lib_fig.convert_3d_to_2d_coords(vxyz)
        vis = visibility.is_visible_point(vxyz,
                                          cam,
                                          tol=tolvis,
                                          nrs=nrsvis,
                                          clean=True)
        verts.append(Vertex(xyz=vxyz,
                            uv=numpy.array([u,v]),
                            visible=vis,
                            index=vid[i]))

f = open(pthout + 'verts_' + strf + '.dat', 'w')
for v in verts:
    f.write(str(v.index+1)      + ', ' +
            str(v.uv[0])        + ', ' +
            str(v.uv[1])        + ', ' +
            str(int(v.visible)) + '\n')
f.close()
################################################






################################################
# Edges
cuv  = numpy.loadtxt(pthout+'curve_uvdata_'+strf+'.dat', delimiter=',')
f = open(pthout + 'edges_' + strf + '.dat', 'w')
for i in range(len(cuv)):
    s = cuv[i,0]
    t = cuv[i,1]
    ds = cuv[i,2]
    dt = cuv[i,3]
    index = int(cuv[i,4])
    
    dxyzds = chebval2d(s, t, cs)
    dxyzdt = chebval2d(s, t, ct)
    nor = unitize(numpy.cross(dxyzds, dxyzdt))
    tng = unitize(ds*dxyzds + dt*dxyzdt)
    vec = numpy.cross(tng,nor)

    xyz = chebval2d(s, t, c)
    u,v = lib_fig.convert_3d_to_2d_coords(xyz)
    u2,v2 = lib_fig.convert_3d_to_2d_coords(xyz + vec)
    du = u2 - u
    dv = v2 - v
    
    vis = visibility.is_visible_point(xyz,
                                      cam,
                                      tol=tolvis,
                                      nrs=nrsvis,
                                      clean=True)    
    f.write(str(index)    + ', ' +
            str(u)        + ', ' +
            str(v)        + ', ' +
            str(du)       + ', ' +
            str(dv)       + ', ' +
            str(int(vis)) + '\n')
f.close()
################################################
