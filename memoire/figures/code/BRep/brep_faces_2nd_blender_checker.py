import bpy
import bpy_extras
import numpy
import sys
sys.path.append('/stck/bandrieu/Bureau/Python/mylibs/')
import my_lib as myl
from numpy.polynomial.chebyshev import chebval2d, chebgrid2d

sys.path.append('/d/bandrieu/GitHub/Code/Python/')
import lib_blender_edit as lbe
import lib_blender_util as lbu
import lib_blender_figures as lbf
import lib_color as lco
import lib_chebyshev as lcheb

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
pthout = '/d/bandrieu/GitHub/These/memoire/figures/data/BRep/'

imgchecker = bpy.data.images.load(filepath='/d/bandrieu/GitHub/These/memoire/figures/code/BRep/checker.png')
texchecker = bpy.data.textures.new('texture_checker', 'IMAGE')
texchecker.image = imgchecker

# Face index
args = sys.argv
if len(args) < 4:
    iface = 0
else:
    iface = int(args[3])-1

strf = format(iface+1,'03')

tolvis = 1.e-3
nrsvis = 32

#################################################
## Set Scene
scene = bpy.context.scene
lbu.clear_scene(meshes=True, lamps=True, cameras=False)

lbu.set_scene(resolution_x=800,
              resolution_y=800,
              resolution_percentage=100,
              alpha_mode='SKY',
              horizon_color=[1,1,1],
              light_samples=16,
              use_environment_light=True,
              environment_energy=0.3,
              environment_color='PLAIN')

## Set Lighting
lbu.add_point_light(name="lamp",
                    energy=1.2,
                    shadow_method='RAY_SHADOW',
                    shadow_ray_samples=16,
                    shadow_soft_size=2.0,
                    location=[3.75,1.65,3.20])

## Set Camera
cam = scene.camera
cam.location = [2.102,1.798,1.104]
cam.rotation_euler = numpy.radians(numpy.array([66.7,0.778,132.2]))
cam.data.angle = numpy.radians(37.72)
#################################################


#################################################
bpy.ops.import_scene.obj(filepath=pthin+'mesh/mesh.obj',
                         axis_forward='Y', axis_up='Z')
obj = bpy.data.objects['mesh']
bpy.ops.object.select_all(action='DESELECT')
obj.select = True
bpy.ops.view3d.camera_to_view_selected() # set camera to fit surf in screen
cam.data.angle += numpy.pi/180.0 # increase FOV angle by 1 degree
bpy.ops.object.delete()
#################################################

clf = numpy.loadtxt(pthout + 'face_color.dat')
clf = lco.cc_hsv(clf, fs=1.2, fv=1.0)

tri = numpy.loadtxt(pthin + 'brepmesh/tri_' + strf + '.dat', dtype=int)-1
xyz = numpy.loadtxt(pthin + 'brepmesh/xyz_' + strf + '.dat', dtype=float)

verts = [[x for x in p] for p in xyz]
faces = [[int(v) for v in t] for t in tri]

obj = lbu.pydata_to_mesh(verts,
                         faces,
                         name='face')

lbe.set_smooth(obj)

# material
mat = bpy.data.materials.new('mat_face')
mat.diffuse_color = clf[iface]
mat.diffuse_intensity = 1
mat.specular_intensity = 0
mat.specular_hardness = 30
mat.use_transparency = False
obj.data.materials.append(mat)

# make group for Freestyle
bpy.ops.object.select_all(action='DESELECT')
scene.objects.active = obj
obj.select = True
bpy.ops.group.create(name='face_group')


################################################
# Add untrimmed surface
step = 20
niso = 6
m = step*niso + 1
u = numpy.linspace(-1.0,1.0,m)
c = myl.readCoeffs(pthin + 'init/coef/c_' + strf + '.cheb')
cs, ct = lcheb.diff2(c)
surf = myl.addTensorProductPatch(chebgrid2d(u, u, c[:,:,0]),
                                 chebgrid2d(u, u, c[:,:,1]),
                                 chebgrid2d(u, u, c[:,:,2]),
                                 name="surf",
                                 location=[0,0,0],
                                 smooth=True,
                                 color=clf[iface],
                                 alpha=1)

# UV coordinates
u = 0.5*(u + 1.0)
bpy.ops.object.select_all(action='DESELECT')
scene.objects.active = surf
surf.select = True
bpy.ops.group.create(name='surf_group')

lbf.fit_camera_to_meshes([surf.data])
cam.data.sensor_width *= 1.01

bpy.ops.object.mode_set(mode='EDIT')
print('   unwrap UVs...')
bpy.ops.uv.unwrap(method='ANGLE_BASED',
                  fill_holes=True,
                  correct_aspect=True,
                  use_subsurf_data=False,
                  margin=0.001)
bpy.ops.object.mode_set(mode='OBJECT')
uvlayer = surf.data.uv_layers.active
print('   edit UVs...')
for j in range(m-1):
    for i in range(m-1):
        k = i + j*(m-1)
        f = surf.data.polygons[k]
        for l in [0,3]:
            uvlayer.data[f.loop_start + l].uv[0] = u[i]
        for l in [1,2]:
            uvlayer.data[f.loop_start + l].uv[0] = u[i+1]
        for l in [0,1]:
            uvlayer.data[f.loop_start + l].uv[1] = u[j]
        for l in [2,3]:
            uvlayer.data[f.loop_start + l].uv[1] = u[j+1]

mat = bpy.data.materials["mat_surf"]
mat.use_raytrace = False
mat.use_shadows = False
mat.use_shadeless = True
mat.use_cast_buffer_shadows = False

slot = mat.texture_slots.add()
slot.texture = texchecker
slot.texture_coords = 'UV'
slot.diffuse_color_factor = 1
                                            
################################################
# Face label
fid = numpy.loadtxt(pthout + 'faces/faces_id.dat', dtype=int)

stlabel = numpy.loadtxt(pthout + 'faces/face_uvlabel_' + strf + '.dat', delimiter=',')
xyzfacelabel = chebval2d(stlabel[0], stlabel[1], c)
myl.addEmpty('label', xyzfacelabel)

u,v = lbf.convert_3d_to_2d_coords(xyzfacelabel)

f = open(pthout + 'faces/face_xyzlabel_' + strf + '.dat', 'w')
f.write(str(fid[iface]+1) + ', ' + str(u) + ', ' + str(v))
f.close()


################################################
# Freestyle settings
freestyle = scene.render.layers.active.freestyle_settings
freestyle.use_smoothness = True

bpy.ops.scene.freestyle_lineset_add() #2

lineset = freestyle.linesets["LineSet"]
lineset.visibility = 'VISIBLE'

lineset = freestyle.linesets["LineSet 2"]
lineset.visibility = 'HIDDEN'
linestyle = bpy.data.linestyles["LineStyle.001"]
linestyle.use_dashed_line = True
linestyle.dash1 = 12
linestyle.gap1 = 10

bpy.ops.scene.freestyle_lineset_add() #3
lineset = freestyle.linesets["LineSet 3"]
lineset.visibility = 'VISIBLE'

for lineset in freestyle.linesets:
    lineset.select_silhouette = False
    lineset.select_border = True
    lineset.select_crease = False
    lineset.select_edge_mark = False
    lineset.select_by_group = True
    
for lineset in freestyle.linesets[0:2]:
    lineset.group = bpy.data.groups["face_group"]
freestyle.linesets[2].group = bpy.data.groups["surf_group"]

for linestyle in bpy.data.linestyles:
    linestyle.thickness_position = 'CENTER'
    linestyle.color = [0,0,0]
    linestyle.caps = "ROUND"
    linestyle.geometry_modifiers["Sampling"].sampling = 0.1

for linestyle in bpy.data.linestyles[0:2]:
    linestyle.thickness = 4.5
bpy.data.linestyles[2].thickness = 3.2
################################################



################################################
# Vertices
vid = numpy.loadtxt(pthout + 'faces/verts_id.dat', dtype=int)

verts = []
vlist = []
f = open(pthin + 'debug/verts.dat','r')
for i in range(len(vid)):
    v = [float(a) for a in f.readline().split()]
    if len(v) > 7 and (iface+1) in v[5:]:
        vlist.append(i)
        vxyz = numpy.asarray(v[0:3])
        u,v = lbf.convert_3d_to_2d_coords(vxyz)
        vis = lbu.is_visible_point(vxyz,
                                   cam,
                                   tol=tolvis,
                                   nrs=nrsvis,
                                   clean=True)
        verts.append(Vertex(xyz=vxyz,
                            uv=numpy.array([u,v]),
                            visible=vis,
                            index=vid[i]))

f = open(pthout + 'faces/verts_' + strf + '.dat', 'w')
for v in verts:
    f.write(str(v.index+1)      + ', ' +
            str(v.uv[0])        + ', ' +
            str(v.uv[1])        + ', ' +
            str(int(v.visible)) + '\n')
f.close()
################################################



################################################
# Edges
cuv  = numpy.loadtxt(pthout + 'faces/curve_uvdata_' + strf + '.dat', delimiter=',')
f = open(pthout + 'faces/edges_' + strf + '.dat', 'w')
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
    u,v = lbf.convert_3d_to_2d_coords(xyz)
    u2,v2 = lbf.convert_3d_to_2d_coords(xyz + vec)
    du = u2 - u
    dv = v2 - v
    
    vis = lbu.is_visible_point(xyz,
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



################################################
# Renders

scene.render.alpha_mode = 'TRANSPARENT'
scene.render.image_settings.file_format='PNG'
pthimg = '/d/bandrieu/GitHub/These/memoire/figures/images/BRep/faces/'


### 1st render: face
scene.render.use_freestyle = False
bpy.data.objects["surf"].hide_render = True
scene.render.filepath = pthimg + 'face_' + format(iface+1,'03')
bpy.ops.render.render( write_still=True )

### 2nd render: visible borders
scene.render.use_freestyle = True
freestyle.linesets[0].show_render = True
freestyle.linesets[1].show_render = False
freestyle.linesets[2].show_render = False
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
freestyle.linesets["LineSet"].show_render = False
freestyle.linesets["LineSet 2"].show_render = True
scene.render.filepath = pthimg + 'edges_hid_' + format(iface+1,'03')
bpy.ops.render.render( write_still=True )

### 4th render: surface
scene.render.use_freestyle = False
bpy.data.objects["surf"].hide_render = False
bpy.data.objects["face"].hide_render = True
scene.render.filepath = pthimg + 'surface_' + format(iface+1,'03')
bpy.ops.render.render( write_still=True )

### 4th render: surface outline
scene.render.use_freestyle = True
freestyle.linesets[2].show_render = True
mat = bpy.data.materials["mat_surf"]
mat.use_transparency = True
mat.alpha = 0
mat.transparency_method = 'MASK'
mat.use_raytrace = False
mat.use_shadows = False
mat.use_cast_buffer_shadows = False
scene.render.filepath = pthimg + 'surface_outline_' + format(iface+1,'03')
bpy.ops.render.render( write_still=True )

exit()

################################################


################################################
# UV domain
uv = numpy.loadtxt(pthin + 'brepmesh/uv_'  + strf + '.dat', dtype=float)
obj = bpy.data.objects['face']
msh = obj.data
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

imuvsize = 200
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

hsv = lco.rgb2hsv(rgba[:,0:3])
ipx = numpy.argmax(numpy.sqrt(hsv[:,1]**2 + hsv[:,2]**2)*rgba[:,3])

clr = 255*rgba[ipx,0:3]
f = open(pthout + 'faces/facecolor_' + strf + '.dat', 'w')
for i in range(3):
    f.write(str(int(clr[i])))
    if i < 2:
        f.write(', ')
f.close()

exit()

