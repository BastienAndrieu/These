import bpy
from mathutils import Vector, Matrix

import numpy
from numpy.polynomial.chebyshev import chebgrid2d, chebval2d

import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python/')
import lib_blender_util as lbu
import lib_blender_edit as lbe
import lib_blender_figures as lbf
import lib_chebyshev as lcheb
import lib_color as lco
from lib_compgeom import circumcircle

################################################################
def is_visible(xyz):
    result = bpy.context.scene.ray_cast(
        start=scene.camera.location,
        end=scene.camera.location + 1.001*(Vector(xyz) - scene.camera.location)
    )
    return result[1] == surf
################################################################


################################################################
pthin = '/d/bandrieu/GitHub/FFTsurf/test/demo_EoS_brep/'
pthout = '/d/bandrieu/GitHub/These/memoire/figures/data/pseudo_EdS_arete/'
pthimg = '/d/bandrieu/GitHub/These/memoire/figures/images/pseudo_EdS_arete/'

ifaceRL = [4,6]
iedge = 16#21-8
################################################################


################################################################
# SET SCENE
scene = bpy.context.scene
lbu.clear_scene(meshes=True, lamps=True, cameras=False)

resx = 1024
resy = 0.75*resx
print('res = (%d, %d)' % (rex, resy))

lbu.set_scene(
    resolution_x=resx,
    resolution_y=resy,
    resolution_percentage=100,
    alpha_mode='SKY',
    horizon_color=(1,1,1),
    light_samples=16,
    use_environment_light=True,
    environment_energy=0.3,
    environment_color='PLAIN'
)
light_settings = scene.world.light_settings
light_settings.use_ambient_occlusion = False#True
light_settings.ao_factor = 0.5
light_settings.ao_blend_type = 'MULTIPLY'

## Set Lighting
lamp = lbu.add_point_light(
    name="lamp",
    energy=1.0,
    shadow_method='NOSHADOW',
    location=(-2.68, 1.65, 3.20)
)

## Set Camera
cam = scene.camera
cam.location = (2.102, 1.798, 1.104)
cam.rotation_euler = numpy.radians((66.7, 0.778, 132.2))
cam.data.angle = numpy.radians(37.72)
################################################################


################################################################
# COLOR PALETTE
"""
color_faces = numpy.loadtxt('/d/bandrieu/GitHub/These/memoire/figures/data/BRep/face_color.dat')
color_faces = lco.cc_hsv(color_faces, fs=1.2, fv=1.0)
ncolors = len(color_faces)
"""

colorR = numpy.array([250, 117, 102])/255.#color_faces[ifaceRL[1]-1]
colorL = numpy.array([152, 212, 91])/255.#color_faces[ifaceRL[0]-1]

colors = numpy.vstack([colorR, colorL])
cc_colors = lco.cc_hsv(colors, fs=0.9, fv=1.1)
colorR = cc_colors[0]
colorL = cc_colors[1]
################################################################




################################################################
# LOAD CURRENT INTERFACE'S MESH
bpy.ops.import_scene.obj(
    filepath=pthin+'mesh/mesh.obj',
    axis_forward='Y',
    axis_up='Z'
)
surf = bpy.data.objects['mesh']
surf.name = 'surface'
# set smooth shading
lbe.set_smooth(surf)

# set materials
for slot in surf.material_slots:
    iface = int(slot.material.name[3:])
    mat = slot.material
    if iface == ifaceRL[0]:
        mat.diffuse_color = colorR
    elif iface == ifaceRL[1]:
        mat.diffuse_color = colorL
    else:
        mat.diffuse_color = (1,1,1)
        if True:
            mat.specular_intensity = 0
            mat.use_transparency = True
            mat.alpha = 0
            mat.use_raytrace = False
            mat.use_shadows = False
            mat.use_cast_buffer_shadows = False
################################################################


################################################################
# RIGHT/LEFT FACES' MATERIALS
mat_faceR = bpy.data.materials.new('mat_faceR')
mat_faceR.diffuse_color = colorR

mat_faceL = bpy.data.materials.new('mat_faceL')
mat_faceL.diffuse_color = colorL

diffuse_intensity = 1
specular_intensity = 0.2
specular_hardness = 30
################################################################


################################################################
# LOAD RIGHT/LEFT FACES EoS MESHES
ptheos = pthin + 'brepmesh_eos/'
facesRL = []
suffixe = 'RL'
for j, iface in enumerate(ifaceRL):
    strf = format(iface, '03')
    tri = numpy.loadtxt(ptheos + 'tri_' + strf + '.dat', dtype=int)-1
    xyz = numpy.loadtxt(ptheos + 'xyz_' + strf + '.dat', dtype=float)

    verts = [[x for x in p] for p in xyz]
    faces = [[int(v) for v in t] for t in tri]

    obj = lbu.pydata_to_mesh(
        verts,
        faces,
        edges=None,
        name='eos_face'+suffixe[j]
    )
    lbe.set_smooth(obj)

    # material
    if j == 0:
        obj.data.materials.append(mat_faceR)
    elif j == 1:
        obj.data.materials.append(mat_faceL)
   
    facesRL.append(obj)
################################################################


################################################################
# LOAD EDGE POLYLINE
mat_edge = bpy.data.materials.new('mat_edge')
mat_edge.diffuse_color = (0,0,0)
mat_edge.use_shadeless = True
mat_edge.use_raytrace = False
mat_edge.use_shadows = False
mat_edge.use_cast_buffer_shadows = False

f = open(pthin + 'debug/edges_xyz.dat')
f.readline()
for i in range(iedge-1):
    npts = int(f.readline())
    for j in range(npts):
        f.readline()
npts = int(f.readline())
edge_xyz = numpy.zeros((npts,3))
for j in range(npts):
    edge_xyz[j] = [float(x) for x in f.readline().split()]

edge_xyz = edge_xyz[::-1]
edge = lbu.pydata_to_polyline(
    edge_xyz,
    name='edge',
    thickness=1e-3,
)
edge.data.materials.append(mat_edge)
edge.hide_render = True
f.close()
################################################################


################################################################
# EDGE PSEUDO-EoS
m = 200
n = 50
u = numpy.linspace(-1.0,1.0,m)
v = numpy.linspace(-1.0,1.0,n)
c = lcheb.read_polynomial2(pthin + 'debug/eos_edge_c_' + format(iedge,'03') + '.cheb')
c = lcheb.flip_polynomial2(c, flip_u=True, flip_v=True)

cu, cv = lcheb.diff2(c)

xyz = chebgrid2d(u, v, c)
verts, faces = lbu.tensor_product_mesh_vf(xyz[0], xyz[1], xyz[2])

pseudoEdS = lbu.pydata_to_mesh(
    verts,
    faces,
    edges=None,
    name='pseudoEdS'
)
# set smooth shading
lbe.set_smooth(pseudoEdS)

# set material
mat_pseudoEdS = bpy.data.materials.new('mat_pseudoEdS')
mat_pseudoEdS.diffuse_color = numpy.array([137, 126, 218])/255.#(0.603, 0.488, 0.8)
pseudoEdS.data.materials.append(mat_pseudoEdS)
################################################################


################################################################
# ADJUST CAMERA
bpy.ops.object.select_all(action='DESELECT')
pseudoEdS.select = True
bpy.ops.view3d.camera_to_view_selected() # set camera to fit 'pseudoEdS' in screen
cam.data.angle += numpy.radians(1.) # increase FOV angle by 1 degree
################################################################


################################################################
# EDGE VISIBILITY
edge.hide = True
edge_vis = []
edge_hid = []

npts = len(edge_xyz)
vis = numpy.zeros(npts, dtype=bool)
for i in range(npts):
    vis[i] = is_visible(edge_xyz[i])

TOLxy = 1e-3
for i in range(npts):
    x, y = lbf.convert_3d_to_2d_coords(edge_xyz[i], normalize=True)
    if vis[i]:
        edge_vis.append([x, y])
    else:
        edge_hid.append([x, y])
    if i < npts-1:
        if vis[i] != vis[i+1]:
            tlim = [0,1]
            xylim = numpy.zeros((2,2))
            vislim = [vis[i], vis[i+1]]
            while True:
                tmid = 0.5*(tlim[0] + tlim[1])
                xyzmid = (1 - tmid)*edge_xyz[i] + tmid*edge_xyz[i+1]
                #
                for j in range(2):
                    xylim[j] = lbf.convert_3d_to_2d_coords(
                        (1 - tlim[j])*edge_xyz[i] + tlim[j]*edge_xyz[i+1],
                        normalize=True
                    )
                if numpy.sum((xylim[0] - xylim[1])**2) < TOLxy**2:
                    break
                #
                vismid = is_visible(xyzmid)
                #
                if vismid == vislim[0]:
                    tlim = [tmid, tlim[1]]
                    vislim = [vismid, vislim[1]]
                else:
                    tlim = [tlim[0], tmid]
                    vislim = [vislim[0], vismid]
            #
            x, y = lbf.convert_3d_to_2d_coords(xyzmid, normalize=True)
            edge_vis.append([x, y])
            edge_hid.append([x, y])


# EXPORT XY CURVES
numpy.savetxt(pthout + 'edge_visible.dat', edge_vis)
numpy.savetxt(pthout + 'edge_hidden.dat', edge_hid)
################################################################


################################################################
# ISO-U CURVE
u0 = -0.6
v0 = 0.05
isou_xyz = chebval2d(u0*numpy.ones(n), v, c).T

isou = lbu.pydata_to_polyline(
    isou_xyz,
    name='isou',
    thickness=1e-3,
    bevel_resolution=4,
    fill_mode='FULL'
)
isou.hide_render = True

isou_xy = numpy.zeros((n,2))
for i in range(n):
    isou_xy[i] = lbf.convert_3d_to_2d_coords(isou_xyz[i], normalize=True)
numpy.savetxt(pthout + 'iso-u_curve.dat', isou_xy)


e = chebval2d(u0, v0, c)
bpy.ops.object.empty_add(location=e)
eR = isou_xyz[0]
bpy.ops.object.empty_add(location=eR)
eL = isou_xyz[-1]
bpy.ops.object.empty_add(location=eL)

eu = chebval2d(u0, v0, cu)
ev = chebval2d(u0, v0, cv)
en = Vector(eu).cross(Vector(ev))
en.normalize()

scale_en = 0.3*(Vector(eR) - Vector(eL)).length
"""
normal_to_envelope = lbu.pydata_to_polyline(
    [e, e+scale_en*en],
    name='normal',
    thickness=1e-3,
)
"""

tng = (Vector(eR) - Vector(e)).cross(Vector(eL) - Vector(e))
tng.normalize()

occ, rcc = circumcircle(eR, eL, e)
print(rcc)
for xyz in [eR, eL, e]:
    print(Vector(xyz - occ).length, Vector(xyz - occ).dot(tng))
b2 = en
b3 = tng
b1 = b3.cross(b2)
rot = Matrix([b1, b2, b3]).transposed()
"""bpy.ops.mesh.primitive_circle_add(
    location=occ,
    radius=rcc,
    rotation=rot.to_euler(),
    vertices=100,
)"""

# RECOVER POINT ON CURVE
normals = []
for vi in [-1,1]:
    eui = chebval2d(u0, vi, cu)
    evi = chebval2d(u0, vi, cv)
    eni = Vector(eui).cross(Vector(evi))
    normals.append(eni.normalized())
delta_normals = normals[1] - normals[0]
rho = Vector(eL - eR).dot(delta_normals) / delta_normals.length_squared
print(rho)

drho_dg = numpy.sqrt(1 - (rcc/rho)**2)
g0 = occ + rho*drho_dg*tng

g0 = g0 + 4e-3*tng

"""
bpy.ops.mesh.primitive_uv_sphere_add(
    location=g0,
    size=rho,
    segments=100,
    ring_count=100
)
sphere = bpy.data.objects['Sphere']
lbe.set_smooth(sphere)
"""
#lamb = 0.5*numpy.dot(eR + eL - 2*occ, tng)
#g0 = occ + lamb*tng
print(Vector(occ - g0).length)
bpy.ops.object.empty_add(location=g0)

f = open(pthout + 'iso-u_points.dat', 'w')
for xyz in [e, eR, eL, e + scale_en*en, occ, g0, g0 + scale_en*tng, e + scale_en*(e - occ)/rcc, eR + scale_en*(eR - occ)/rcc, eL + scale_en*(eL - occ)/rcc]:
    x, y = lbf.convert_3d_to_2d_coords(xyz, normalize=True)
    f.write('%s, %s\n' % (x, y))
f.close()
################################################################


################################################################
# ANGLE ANOTATION
vecrR = Vector(eR - occ)
rR = vecrR.length
vecrR.normalize()

vecrL = Vector(eL - occ)
rL = vecrL.length
vecrL.normalize()

vecrR90d = vecrR.cross(tng)

if True:
    angle = numpy.arctan2(vecrL.dot(vecrR90d), vecrL.dot(vecrR))
    if angle < 0: angle += 2*numpy.pi
else:
    w = vecrR.cross(vecrL)
    angle = numpy.arctan2(w.length, vecrR.dot(vecrL))
    if w.dot(tng) > 0:
        angle += numpy.pi
    print(w.dot(tng))
print(angle)

v01 = 0.5*(v + 1)
xyz = numpy.zeros((n,3))
rangle = 0.33*rcc
for j in range(n):
    aj = v01[j]*angle
    xyz[j] = occ + rangle*(vecrR*numpy.cos(aj) + vecrR90d*numpy.sin(aj))

angle_anotation = lbu.pydata_to_polyline(
    xyz,
    name='angle_anotation',
    thickness=1e-3,
)
angle_anotation.hide_render = True

xy = numpy.zeros((n,2))
for i in range(n):
    xy[i] = lbf.convert_3d_to_2d_coords(xyz[i], normalize=True)
numpy.savetxt(pthout + 'angle_anotation.dat', xy)
################################################################


################################################################
# ISO-V CURVES (BOUNDARY CURVES)
for j, jv in enumerate([0,n-1]):
    isov_xyz = chebval2d(u, v[jv]*numpy.ones(m), c).T
    isov_xy = numpy.zeros((m,2))
    for i in range(m):
        isov_xy[i] = lbf.convert_3d_to_2d_coords(isov_xyz[i], normalize=True)
    numpy.savetxt(pthout + 'iso-v_curve_' + str(j) + '.dat', isov_xy)
################################################################


################################################################
# ADJUST MATERIALS
for mat in bpy.data.materials:
    if mat.name not in ['mat_edge']:
        mat.diffuse_intensity = diffuse_intensity
        mat.specular_intensity = specular_intensity
        mat.specular_hardness = specular_hardness

for mat in [mat_faceR, mat_faceL, mat_pseudoEdS]:
    mat.use_transparency = True
    mat.alpha = 1
    mat.emit = 0.22
    mat.specular_intensity = 0.3
    mat.specular_hardness = 30
    mat.raytrace_transparency.fresnel = 2.7
    mat.raytrace_transparency.fresnel_factor = 1.1
################################################################

for obj in bpy.data.objects:
    if obj.type == 'EMPTY':
        obj.empty_draw_size = 0.1

pseudoEdS.hide = True

"""
################################################################
# MAKE GROUPS
bpy.ops.object.select_all(action='DESELECT')
pseudoEdS.select = True
bpy.ops.group.create(name='group_pseudoEdS')
################################################################


################################################################
# FREESTYLE SETTINGS
scene.render.use_freestyle = False
freestyle = scene.render.layers.active.freestyle_settings
freestyle.use_smoothness = True

# pseudoEdS border
lineset = freestyle.linesets['LineSet']
lineset.name = 'lineset_pseudoEdS_vis'
lineset.select_silhouette = False
lineset.select_border = True
lineset.select_by_group = True
lineset.select_crease = False
lineset.visibility = 'RANGE'
linestyle = bpy.data.linestyles['LineStyle']
linestyle.name = 'linestyle_pseudoEds_vis'
linestyle.color = (0.603, 0.488, 0.8)
linestyle.thickness = 3
lineset.group = bpy.data.groups['group_pseudoEdS']

for linestyle in bpy.data.linestyles:
    linestyle.caps = 'ROUND'
    linestyle.geometry_modifiers["Sampling"].sampling = 0.1
################################################################
"""


################################################
# COMPOSITING
scene.use_nodes = True
tree = scene.node_tree

# clear default nodes
for node in tree.nodes:
    tree.nodes.remove(node)

# create input image node
node_render = tree.nodes.new(type='CompositorNodeRLayers')
node_render.layer = 'RenderLayer'
node_render.location = 0,0

node_cc = tree.nodes.new(type='CompositorNodeHueSat')
node_cc.color_saturation = 0.92
node_cc.color_value = 1.02
node_cc.location = 200,0

node_output = tree.nodes.new(type='CompositorNodeOutputFile')
node_output.base_path = pthimg
node_output.file_slots[0].path = 'pseudo_EdS_arete'
node_output.location = 400,0

node_viewer = tree.nodes.new(type='CompositorNodeViewer')
node_viewer.location = 400,200

# link nodes
links = tree.links
links.new(node_render.outputs[0], node_cc.inputs[1])
links.new(node_cc.outputs[0], node_output.inputs[0])
links.new(node_cc.outputs[0], node_viewer.inputs[0])
################################################


################################################################
# RENDER
if False:
    print('Rendering ...')
    #scene.render.filepath = pthimg + 'pseudo_EdS_arete'
    #bpy.ops.render.render(write_still=True)
    bpy.ops.render.render()
################################################################
