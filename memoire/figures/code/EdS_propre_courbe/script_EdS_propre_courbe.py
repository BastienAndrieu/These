import bpy
from mathutils import Vector, Matrix
import numpy

import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python/')
import lib_blender_util as lbu
import lib_blender_edit as lbe
import lib_blender_figures as lbf
import lib_bezier as lbez
from lib_EoS import canal_surface

###############################################
class BezierCurve:
    def __init__(self, curve):
        if curve.dimensions == '2D':
            dim = 2
        else:
            dim = 3
        #
        spline = curve.splines.active
        if spline.type == 'BEZIER':
            self.x = numpy.zeros((4,dim))
            self.x[0] = spline.bezier_points[0].co[0:dim]
            self.x[1] = spline.bezier_points[0].handle_right[0:dim]
            self.x[2] = spline.bezier_points[1].handle_left[0:dim]
            self.x[3] = spline.bezier_points[1].co[0:dim]
        self.update()
        return   
    #
    def update(self):
        self.xt = lbez.diff(self.x)
        self.xtt = lbez.diff(self.xt)
        return
    #
    def eval(self, t):
        return lbez.eval_bezier_curve(self.x, t)
    #
    def evald(self, t):
        return lbez.eval_bezier_curve(self.xt, t)
    #
    def evald2(self, t):
        return lbez.eval_bezier_curve(self.xtt, t)
##############################################
def set_dummy_material(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select = True
    scene.objects.active = obj
    bpy.ops.object.material_slot_remove()
    obj.data.materials.append(mat_dummy)
    return
##############################################

pth = '/d/bandrieu/GitHub/These/memoire/figures/data/EdS_propre_courbe/'

scene = bpy.context.scene
camera = scene.camera


REVERSE = True

##############################################
radius_function = bpy.data.curves['radius_function']
spine = bpy.data.curves['spine']

Br = BezierCurve(radius_function)
Bxyz = BezierCurve(spine)

"""
npts = 10
t = numpy.linspace(0,1,npts)

xyz = Bxyz.eval(t)
r = Br.eval(t)[:,1]


for i in range(npts):
    bpy.ops.mesh.primitive_uv_sphere_add(
        location=xyz[i],
        size=r[i]
    )
"""
################################################
# ADD CANAL SURFACE
m = 100
n = 100

u = numpy.linspace(0.15,1,m)
v = numpy.linspace(0,1,n)

g = Bxyz.eval(u)
dg = Bxyz.evald(u)
d2g = Bxyz.evald2(u)

r = Br.eval(u)[:,1]
dr = Br.evald(u)[:,1]


e = canal_surface(g.T, dg.T, d2g.T, r, dr, v)

verts, faces = lbu.tensor_product_mesh_vf(e[:,:,0], e[:,:,1], e[:,:,2], periodv=True)

EdS = lbu.pydata_to_mesh(
    verts,
    faces,
    edges=None,
    name='EdSpropre'
)

lbe.set_smooth(EdS)
#EdS.show_wire = True
#EdS.show_all_edges = True
EdS.show_transparent = True

mat_EdS = bpy.data.materials.new('mat_EdS')
mat_EdS.diffuse_color = (0.527, 0.800, 0.213)
mat_EdS.diffuse_intensity = 1
mat_EdS.use_transparency = True
mat_EdS.alpha = 0.68
mat_EdS.emit = 0.8#1.0
mat_EdS.specular_intensity = 0.5
mat_EdS.specular_hardness = 30
mat_EdS.raytrace_transparency.fresnel = 2.7
mat_EdS.use_shadows = False
mat_EdS.use_cast_buffer_shadows = False

EdS.data.materials.append(mat_EdS)
################################################


################################################
# EXPORT SPINE'S COORDINATES IN IMAGE
f = open(pth + 'spine_xy.dat', 'w')
if REVERSE: g = g[::-1]
for xyz in g:
    x, y = lbf.convert_3d_to_2d_coords(xyz, normalize=True)
    f.write('%s %s\n' % (x, y))
f.close()
################################################


################################################
# ADD SPHERE, PLANE AND CHARACTERISTIC CIRCLE
u0 = 0.65#0.5
if False:
    uc = numpy.linspace(0.3,0.7,100)
    emax = 0
    for ui in uc:
        gi = Bxyz.eval(ui)
        gix, giy = lbf.convert_3d_to_2d_coords(gi, normalize=True)
        gui = Bxyz.evald(ui)
        guui = Bxyz.evald2(ui)
        ri = Br.eval(ui)[1]
        rui = Br.evald(ui)[1]
        gui_cross_guui = numpy.cross(gui, guui)
        dRi = ri*rui*numpy.sum(gui**2)/numpy.sqrt(numpy.sum(gui_cross_guui**2))
        #ei = abs(numpy.sqrt(dRi**2 + 1) - 1)/ri

        occi = gi - ri*rui*gui/numpy.sum(gui**2)
        occix, occiy = lbf.convert_3d_to_2d_coords(occi, normalize=True)
        ei = (occix - gix)**2 + (occiy - giy)**2
        if ei > emax:
            print('ui = %s, ei=%s' % (ui, ei))
            u0 = ui
            emax = ei
    print('u0 = ', u0)

g0 = Bxyz.eval(u0)
dg0 = Bxyz.evald(u0)

r0 = Br.eval(u0)[1]
dr0 = Br.evald(u0)[1]

scale_dg = 0.29
if REVERSE: scale_dg *= -1

f = open(pth + 'spine_point_tangent.dat', 'w')
x, y = lbf.convert_3d_to_2d_coords(g0, normalize=True)
bpy.ops.object.empty_add(location=g0)
f.write('%s, %s\n' % (x, y))
x, y = lbf.convert_3d_to_2d_coords(g0 + scale_dg*dg0, normalize=True)
bpy.ops.object.empty_add(location=g0 + scale_dg*dg0)
f.write('%s, %s\n' % (x, y))
f.close()


## SPHERE
bpy.ops.mesh.primitive_uv_sphere_add(
    location=g0,
    size=0.995*r0,
    segments=100,
    ring_count=100
)

sphere = bpy.data.objects['Sphere']
lbe.set_smooth(sphere)

mat_sphere = bpy.data.materials.new('mat_sphere')
mat_sphere.diffuse_color = (0.800, 0.494, 0.317)
mat_sphere.diffuse_intensity = 1
mat_sphere.specular_intensity = 0.75#1
mat_sphere.specular_hardness = 30
mat_sphere.use_transparency = True
mat_sphere.raytrace_transparency.fresnel = 2.1
mat_sphere.alpha = 1
mat_sphere.emit = 1.2
mat_sphere.use_shadows = False
mat_sphere.use_cast_buffer_shadows = False

sphere.data.materials.append(mat_sphere)
sphere.draw_type = 'WIRE'


## CHARACTERISTIC CIRCLE CENTER AND RADIUS
normdg = numpy.sqrt(numpy.sum(dg0**2))
sina = dr0/normdg
occ = g0 - r0*sina*dg0/normdg
bpy.ops.object.empty_add(location=occ)

rcc = r0*numpy.sqrt(1 - sina**2)

## PLANE
b1 = Vector(dg0)
b1.normalize()
b2 = Vector((0,1,0))
b2 = b2 - b1*(b2.dot(b1))
b2.normalize()
b3 = b1.cross(b2)

rot = Matrix([
    [b2[0], b3[0], b1[0]],
    [b2[1], b3[1], b1[1]],
    [b2[2], b3[2], b1[2]]
])

bpy.ops.mesh.primitive_plane_add(
    location=occ,
    radius=1.2*r0,
    rotation=rot.to_euler()
)

plane = bpy.data.objects['Plane']

mat_plane = bpy.data.materials.new('mat_plane')
mat_plane.diffuse_color = (0.255, 0.435, 0.800)
mat_plane.diffuse_intensity = 1
mat_plane.specular_intensity = 0
mat_plane.use_transparency = True
mat_plane.alpha = 0.4
mat_plane.emit = 1.8
mat_plane.use_shadows = False
mat_plane.use_cast_buffer_shadows = False

plane.data.materials.append(mat_plane)
plane.hide = True

## CHARACTERISTIC CIRCLE
bpy.ops.mesh.primitive_circle_add(
    location=occ,
    radius=rcc,
    rotation=rot.to_euler(),
    vertices=100,
    fill_type='TRIFAN'
)

circle = bpy.data.objects['Circle']
circle.show_wire = True
circle.show_all_edges = False
circle.show_transparent = True

"""
mat_circle = bpy.data.materials.new('mat_circle')
mat_circle.diffuse_intensity = 1
mat_circle.specular_intensity = 0
mat_circle.use_transparency = True
mat_circle.alpha = 0
mat_circle.use_shadows = False
mat_circle.use_cast_buffer_shadows = False
"""

circle.data.materials.append(mat_plane)

f = open(pth + 'circle_center_point.dat', 'w')
x, y = lbf.convert_3d_to_2d_coords(occ, normalize=True)
f.write('%s, %s\n' % (x, y))
#x, y = lbf.convert_3d_to_2d_coords(g0 + 1.0*dg0, normalize=True)
#f.write('%s %s\n' % (x, y))
f.close()
################################################


################################################
# ENVELOPE LABEL
if REVERSE:
    ii = 0
else:
    ii = m-1
p = e[ii,0]
for jj in range(1,n-1):
    if e[ii,jj,2] > p[2]: p = e[ii,jj]
    
f = open(pth + 'envelope_data.dat', 'w')
x, y = lbf.convert_3d_to_2d_coords(p, normalize=True)
f.write('%s, %s\n' % (x, y))

# POINT ON ENVELOPE AND NORMAL TO ENVELOPE
vec = camera.location - Vector(occ)
vec.normalize()
vecp = (vec.dot(b2), vec.dot(b3))
delta_angle = numpy.radians(30.)
angle = numpy.arctan2(vecp[1], vecp[0]) - delta_angle
eij = occ + (b2*numpy.cos(angle) + b3*numpy.sin(angle))*rcc
bpy.ops.object.empty_add(location=eij)
x, y = lbf.convert_3d_to_2d_coords(eij, normalize=True)
f.write('%s, %s\n' % (x, y))
x, y = lbf.convert_3d_to_2d_coords(eij + 0.85*(eij - occ), normalize=True)
f.write('%s, %s\n' % (x, y))
# circle label
angle = numpy.arctan2(vecp[1], vecp[0]) + delta_angle
clab = occ + (b2*numpy.cos(angle) + b3*numpy.sin(angle))*rcc
x, y = lbf.convert_3d_to_2d_coords(clab, normalize=True)
f.write('%s, %s\n' % (x, y))
f.close()

################################################


#################################################
# MAKE GROUPS
if True:
    bpy.ops.object.select_all(action='DESELECT')
    circle.select = True
    bpy.ops.group.create(name='group_circle')
    
    bpy.ops.object.select_all(action='DESELECT')
    plane.select = True
    bpy.ops.group.create(name='group_plane')
    
    bpy.ops.object.select_all(action='DESELECT')
    sphere.select = True
    bpy.ops.group.create(name='group_sphere')
    
    bpy.ops.object.select_all(action='DESELECT')
    EdS.select = True
    bpy.ops.group.create(name='group_EdS')
else:
    grp = bpy.data.groups.get('group_circle')
    grp.objects.link(circle)
    
    grp = bpy.data.groups.get('group_plane')
    grp.objects.link(plane)
    
    grp = bpy.data.groups.get('group_sphere')
    grp.objects.link(sphere)
    
    grp = bpy.data.groups.get('group_EdS')
    grp.objects.link(EdS)
#################################################




################################################
# RENDERS
pthimg = '/d/bandrieu/GitHub/These/memoire/figures/images/EdS_propre_courbe/'

## Dummy material
mat_dummy = bpy.data.materials.new('mat_dummy')
mat_dummy.specular_intensity = 0
mat_dummy.use_transparency = True
mat_dummy.alpha = 0
mat_dummy.use_raytrace = False
mat_dummy.use_shadows = False
mat_dummy.use_cast_buffer_shadows = False

## Set layers
EdS.layers[1] = True
EdS.layers[0] = False

for obj in [sphere, plane, circle]:
    obj.layers[2] = True
    obj.layers[0] = False

scene.render.use_freestyle = False
scene.render.layers['RenderLayer'].use = False

## Render layers
# 1) ENVELOPE (NO FREESTYLE)
renderlayer_EdS = scene.render.layers.new('renderlayer_EdS')
renderlayer_EdS.layers[0] = True
renderlayer_EdS.layers[1] = True
renderlayer_EdS.layers[2] = False

## 2) SPHERE, PLANE & CIRCLE (NO FREESTYLE)
renderlayer_spc = scene.render.layers.new('renderlayer_spc')
renderlayer_spc.layers[0] = True
renderlayer_spc.layers[1] = False
renderlayer_spc.layers[2] = True


for ilayer in range(3):
    scene.layers[ilayer] = True
################################################


################################################
# FREESTYLE SETTINGS
scene.render.use_freestyle = True

color_EdS = (0.252, 0.776, 0.088)
color_plane = (0.049, 0.363, 1)
color_circle = color_plane
color_sphere = (0.799, 0.396, 0.159)

thickness_EdS_border = 1.5
thickness_plane = 2
thickness_sphere = 3


scene.render.layers.active = renderlayer_EdS
freestyle = scene.render.layers.active.freestyle_settings

# EdS silhouette
bpy.ops.scene.freestyle_lineset_add()
lineset = freestyle.linesets['LineSet']
lineset.name = 'lineset_EdS_silhouette'
lineset.select_silhouette = True
lineset.select_border = False
lineset.visibility = 'VISIBLE'
linestyle = bpy.data.linestyles['LineStyle.001']
linestyle.name = 'linestyle_EdS_silhouette'
linestyle.color = color_EdS
linestyle.thickness = 3
lineset.group = bpy.data.groups['group_EdS']

# EdS border, visible
bpy.ops.scene.freestyle_lineset_add()
lineset = freestyle.linesets['LineSet 2']
lineset.name = 'lineset_EdS_border_vis'
lineset.select_silhouette = False
lineset.select_border = True
lineset.visibility = 'VISIBLE'
linestyle = bpy.data.linestyles['LineStyle.001']
linestyle.name = 'linestyle_EdS_border_vis'
linestyle.color = color_EdS
linestyle.thickness = thickness_EdS_border
lineset.group = bpy.data.groups['group_EdS']

# EdS border, hidden
bpy.ops.scene.freestyle_lineset_add()
lineset = freestyle.linesets['LineSet 3']
lineset.name = 'lineset_EdS_border_hid'
lineset.select_silhouette = False
lineset.select_border = True
lineset.visibility = 'HIDDEN'
linestyle = bpy.data.linestyles['LineStyle.001']
linestyle.name = 'linestyle_EdS_border_hid'
linestyle.color = color_EdS
linestyle.thickness = thickness_EdS_border
linestyle.use_dashed_line = True
linestyle.dash1 = 10
linestyle.gap1 = 7
lineset.group = bpy.data.groups['group_EdS']



scene.render.layers.active = renderlayer_spc
freestyle = scene.render.layers.active.freestyle_settings

# circle, visible
bpy.ops.scene.freestyle_lineset_add()
lineset = freestyle.linesets['LineSet']
lineset.name = 'lineset_circle_vis'
lineset.select_silhouette = False
lineset.select_border = True
lineset.visibility = 'VISIBLE'
linestyle = bpy.data.linestyles['LineStyle.001']
linestyle.name = 'linestyle_circle_vis'
linestyle.color = color_circle
linestyle.thickness = 4.5
lineset.group = bpy.data.groups['group_circle']


# circle, hidden
bpy.ops.scene.freestyle_lineset_add()
lineset = freestyle.linesets['LineSet 2']
lineset.name = 'lineset_circle_hid'
lineset.select_silhouette = False
lineset.select_border = True
lineset.visibility = 'HIDDEN'
linestyle = bpy.data.linestyles['LineStyle.001']
linestyle.name = 'linestyle_circle_hid'
linestyle.color = color_circle
linestyle.thickness = 3
linestyle.use_dashed_line = True
linestyle.dash1 = 10
linestyle.gap1 = 7
lineset.group = bpy.data.groups['group_circle']


# sphere
bpy.ops.scene.freestyle_lineset_add()
lineset = freestyle.linesets['LineSet 3']
lineset.name = 'lineset_sphere'
lineset.select_silhouette = True
lineset.select_border = False
lineset.visibility = 'RANGE'
linestyle = bpy.data.linestyles['LineStyle.001']
linestyle.name = 'linestyle_sphere'
linestyle.color = color_sphere
linestyle.thickness = thickness_sphere
linestyle.use_dashed_line = True
linestyle.dash1 = 12
linestyle.gap1 = 12
lineset.group = bpy.data.groups['group_sphere']


for layer in [renderlayer_EdS, renderlayer_spc]:
    freestyle = layer.freestyle_settings
    freestyle.use_smoothness = True
    for lineset in freestyle.linesets:
        lineset.select_by_group = True
        lineset.select_crease = False
    
for linestyle in bpy.data.linestyles:
    linestyle.caps = 'ROUND'
    linestyle.geometry_modifiers["Sampling"].sampling = 0.1
################################################



################################################
# COMPOSITING
# (https://blender.stackexchange.com/questions/19500/controling-compositor-by-python)
# switch on nodes and get reference
scene.use_nodes = True
tree = scene.node_tree

# clear default nodes
for node in tree.nodes:
    tree.nodes.remove(node)

# create input image node
node_EdS = tree.nodes.new(type='CompositorNodeRLayers')
node_EdS.layer = 'renderlayer_EdS'
node_EdS.location = 0,150

node_spc = tree.nodes.new(type='CompositorNodeRLayers')
node_spc.layer = 'renderlayer_spc'
node_spc.location = 0,-150

node_mix = tree.nodes.new(type='CompositorNodeAlphaOver')
node_mix.location = 200,0

node_whitebg = tree.nodes.new(type='CompositorNodeAlphaOver')
node_whitebg.location = 500,200

node_output = tree.nodes.new(type='CompositorNodeOutputFile')
node_output.base_path = pthimg
node_output.location = 400,0

node_viewer = tree.nodes.new(type='CompositorNodeViewer')
node_viewer.location = 700,150

# link nodes
links = tree.links
links.new(node_EdS.outputs[0], node_mix.inputs[1])
links.new(node_spc.outputs[0], node_mix.inputs[2])
links.new(node_mix.outputs[0], node_output.inputs[0])
links.new(node_mix.outputs[0], node_whitebg.inputs[2])
links.new(node_whitebg.outputs[0], node_viewer.inputs[0])
################################################




################################################
# RENDER
plane.hide_render = True

bpy.ops.render.render()
################################################

