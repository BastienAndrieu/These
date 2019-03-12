import bpy
import numpy as np
from numpy.polynomial.chebyshev import chebval
import sys
sys.path.append('/stck/bandrieu/Bureau/Python/mylibs/')
sys.path.append('/d/bandrieu/GitHub/These/memoire/figures/code/')
import chebyshev_lib as cheblib
import lib_EoS as eos
import lib_fig as fig
import my_lib as myl
import bpy_extras

###############################
def fr(x):
    return 0.2*(1.0 + 1.2*x[0] + 0.2*x[1] - 0.6*x[2])
###############################

# spine's Bezier control points
b = np.array([
    [0., -0.2570,  0.0],
    [1.,  0.3086, -0.6],
    [2.,  0.3045, -0.6],
    [3., -0.1435,  0.4],
    [4.,  0.0814,  0.6]])

# convert to Chebyshev polynomial
N = len(b)-1
B2C = cheblib.B2Cmatrix(N+1)
pg = np.zeros(b.shape)
for j in range(3):
    for i in range(N+1):
        for k in range(N+1):
            pg[i,j] += B2C[i,k]*b[k,j]
# 1st and 2nd polynomial derivatives
dpg = cheblib.diff(pg)
d2pg = cheblib.diff(dpg)

# sampling
m = 200
n = m
m1 = 20
m2 = 120
dm = 12
u = np.linspace(-1,1,m)
v = np.pi*np.linspace(-1,1,n+1)

u0 = -0.5

# spine curve
g = chebval(u, pg)
dg = chebval(u, dpg)
d2g = chebval(u, d2pg)

gc = chebval(cheblib.cgl_nodes(N), pg)
pr = cheblib.fcht(fr(gc))
dpr = cheblib.diff(pr)
r = chebval(u, pr)
dr = chebval(u, dpr)

# envelope
e = eos.one_parameter_EoS(g, dg, d2g, r, dr, v)
e = e[m1:m2+1,:,:]

##############################
# clear blender scene
myl.blankScene()


############################################################
scene = bpy.context.scene

scene.camera.location = [14.6387, -19.52351, 0.52857]
scene.camera.rotation_euler = np.array([87.975, 0.62, 393.916])*np.pi/180.0
bpy.data.cameras["Camera"].lens_unit = "FOV"
bpy.data.cameras["Camera"].angle = 6.1*np.pi/180.0
bpy.data.cameras["Camera"].shift_x = 0.018
bpy.data.cameras["Camera"].shift_y = 0.03

scene.render.resolution_x = 1400
scene.render.resolution_y = 960
scene.render.resolution_percentage = 100

bpy.data.worlds["World"].horizon_color = [1,1,1]
scene.render.use_border = False
scene.render.use_freestyle = True
#scene.render.alpha_mode = 'TRANSPARENT'
############################################################

# add spine curve
if True:
    # create dummy mesh
    verts = []
    for j in range(2):
        nv = 0
        h = 1.e-3*j
        for i in range(m1-dm,m2+dm+1):
            verts.append([g[0,i], g[1,i], g[2,i]+h])
            nv += 1
            
    faces = []
    for i in range(nv-1):
        faces.append([i, i+1, nv+i+1, nv+i])
    #create mesh and object
    mesh = bpy.data.meshes.new("ObjCurve")
    obj = bpy.data.objects.new("ObjCurve",mesh)
    #set mesh location
    obj.location = [0,0,0]
    bpy.context.scene.objects.link(obj)
    #create mesh from python data
    mesh.from_pydata(verts,[],faces)
    mesh.update(calc_edges=True)
    # material
    mat = myl.surfaceMaterial("mat_dummy",np.ones(3),0,0)
    mat.specular_intensity = 0
    mat.use_raytrace = False
    mat.use_shadows = False
    mat.use_cast_buffer_shadows = False
    obj.data.materials.append(mat)

    
    # mark edges
    bpy.context.scene.objects.active = obj
    obj.select = True

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_mode(type="EDGE")
    bpy.ops.object.mode_set(mode='OBJECT')

    for ed in mesh.edges:
        v1 = ed.vertices[0]
        v2 = ed.vertices[1]
        p = min(v1,v2)
        q = max(v1,v2)
        if q == p + 1 and q < nv:
            ed.select = True

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.loop_multi_select(ring=False)
    bpy.ops.mesh.mark_freestyle_edge(clear=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    obj.select = False
    
else:
    myl.addPolyline(g[:,m1-dm:m2+dm+1].T, [0,0,0], 1.5e-3, 0)
    bpy.data.materials["Mat_line"].use_transparency = True
    bpy.data.materials["Mat_line"].alpha = 0

# add envelope ############
myl.addTensorProductPatch(e[:,:,0], e[:,:,1], e[:,:,2],
                          name="envelope",
                          periodu=False, periodv=True,
                          location=[0,0,0],
                          smooth=True,
                          color=[0.527, 0.800, 0.213], alpha=0.68, emit=1.0)
mat = bpy.data.materials["mat_envelope"]
mat.specular_intensity = 0.5
mat.specular_hardness = 30
mat.raytrace_transparency.fresnel = 2.7
mat.use_shadows = False
mat.use_cast_buffer_shadows = False
###########################


# add sphere ##############
g0 = chebval(u0, pg)
r0 = chebval(u0, pr)
bpy.ops.mesh.primitive_uv_sphere_add(
    location=g0,
    segments=100,
    ring_count=100,
    size=0.995*r0)
obj = bpy.data.objects["Sphere"]
myl.setSmooth(obj)
mat = bpy.data.materials.new("mat_sph")
mat.diffuse_color = [0.800, 0.494, 0.317]
mat.diffuse_intensity = 1
mat.specular_intensity = 1
mat.specular_hardness = 30
mat.use_transparency = True
mat.raytrace_transparency.fresnel = 2.1
mat.alpha = 1
mat.emit = 1.2
mat.use_shadows = False
mat.use_cast_buffer_shadows = False
myl.setMaterial(obj, mat)
###########################


# add point ##############
obj = myl.addEmpty(name="g0", location=g0)
fig.get_2d_coordinates(obj)
###########################


# add plane ###############
dg0 = chebval(u0, dpg)
dr0 = chebval(u0, dpr)

normdg = np.sqrt(np.sum(np.power(dg0,2)))
sina = dr0/normdg
occ = g0 - r0*sina*dg0/normdg
obj = myl.addEmpty(name="o", location=occ)
fig.get_2d_coordinates(obj)
"""
co_2d = bpy_extras.object_utils.world_to_camera_view(scene,
                                                     scene.camera,
                                                     obj.location)
print(obj.name)
print("   2D Coords:   ",
      co_2d.x - bpy.data.cameras["Camera"].shift_x, ', ',
      co_2d.y - bpy.data.cameras["Camera"].shift_y)
"""
rcc = r0*np.sqrt(1. - sina**2)
bw = dg0/normdg
bu0 = np.array([-1.,0.,0.15])
bu = bu0 - np.dot(bu0,bw)*bw
bu = bu / np.sqrt(np.sum(np.power(bu,2)))
bv = np.cross(bw,bu)

# tangent to spine (normal to plane)
#tng = np.vstack([g0, g0+dg0])
#myl.addPolyline(tng, [1,0,0], 1.5e-3, 0)
obj = myl.addEmpty(name="dg0", location=g0+0.6*dg0)
fig.get_2d_coordinates(obj)


verts = []
faces = [[0,2,3],[0,3,1]]
for i in range(2):
    for j in range(2):
        vij = occ + 1.2*r0*(bu*(-1)**i + bv*(-1)**j)
        verts.append([vij[0], vij[1], vij[2]])

#create mesh and object
mesh = bpy.data.meshes.new("plane")
object = bpy.data.objects.new("plane",mesh)

#set mesh location
object.location = bpy.context.scene.cursor_location
bpy.context.scene.objects.link(object)

#create mesh from python data
mesh.from_pydata(verts,[],faces)

# material
mat = bpy.data.materials.new("mat_pla")
mat.diffuse_color = [0.255, 0.435, 0.800]
mat.diffuse_intensity = 1
mat.specular_intensity = 0
mat.use_transparency = True
mat.alpha = 0.4
mat.emit = 1.8
mat.use_shadows = False
mat.use_cast_buffer_shadows = False
myl.setMaterial(object, mat)
###########################


# add char. circle ########
verts = []
faces = []
#rcc *= 0.995#1.003
for j in range(n):
    vj = occ + rcc*(np.cos(v[j])*bu + np.sin(v[j])*bv)
    verts.append([vj[0], vj[1], vj[2]])
    faces.append([j, (j+1)%n, n])
verts.append([occ[0], occ[1], occ[2]])
#create mesh and object
mesh = bpy.data.meshes.new("ccircle")
object = bpy.data.objects.new("ccircle",mesh)

#set mesh location
object.location = bpy.context.scene.cursor_location
bpy.context.scene.objects.link(object)

#create mesh from python data
mesh.from_pydata(verts,[],faces)

mat = bpy.data.materials.new("mat_cir")
mat.diffuse_intensity = 1
mat.specular_intensity = 0
mat.use_transparency = True
mat.alpha = 0
mat.use_shadows = False
mat.use_cast_buffer_shadows = False
myl.setMaterial(object, mat)
###########################

############################################################
# GROUPS
############################################################
for ob in scene.objects:
    ob.select = False

bpy.data.objects['ccircle'].select = True
bpy.ops.group.create(name="ccircle_group")
bpy.data.objects['ccircle'].select = False

bpy.data.objects['plane'].select = True
bpy.ops.group.create(name="plane_group")
bpy.data.objects['plane'].select = False

bpy.data.objects['envelope'].select = True
bpy.ops.group.create(name="envelope_group")
bpy.data.objects['envelope'].select = False

bpy.data.objects['Sphere'].select = True
bpy.ops.group.create(name="sphere_group")
bpy.data.objects['Sphere'].select = False

bpy.data.objects['ObjCurve'].select = True
bpy.ops.group.create(name="spine_group")
bpy.data.objects['ObjCurve'].select = False

############################################################
# FREESTYLE
############################################################
bpy.ops.scene.freestyle_lineset_add() #2
bpy.ops.scene.freestyle_lineset_add() #3
bpy.ops.scene.freestyle_lineset_add() #4
bpy.ops.scene.freestyle_lineset_add() #5
bpy.ops.scene.freestyle_lineset_add() #6
bpy.ops.scene.freestyle_lineset_add() #7
bpy.ops.scene.freestyle_lineset_add() #8
bpy.ops.scene.freestyle_lineset_add() #9
bpy.ops.scene.freestyle_lineset_add() #10
bpy.ops.scene.freestyle_lineset_add() #11
bpy.ops.scene.freestyle_lineset_add() #12

freestyle = scene.render.layers.active.freestyle_settings
freestyle.use_smoothness = True

dash_len = 10
gap_len = 7

cc_cl = [0.049,0.363,1.00]
# line set 1 (ccircle, border, visible)
lineset = freestyle.linesets["LineSet"]
lineset.select_silhouette = False
lineset.select_border = False
lineset.select_crease = False
lineset.select_by_group = True
lineset.visibility = 'RANGE'
lineset.qi_start = 0
lineset.qi_end = 1
linestyle = bpy.data.linestyles["LineStyle"]
linestyle.caps = "ROUND"
linestyle.use_chaining = False
linestyle.thickness = 4.5
linestyle.color = cc_cl
lineset.group = bpy.data.groups["ccircle_group"]

# line set 2 (ccircle, border, hidden)
lineset = freestyle.linesets["LineSet 2"]
lineset.select_silhouette = False
lineset.select_border = False
lineset.select_crease = False
lineset.select_by_group = True
lineset.visibility = 'RANGE'
lineset.qi_start = 2
lineset.qi_end = 100
linestyle = bpy.data.linestyles["LineStyle.001"]
linestyle.caps = "ROUND"
linestyle.use_chaining = True
linestyle.color = cc_cl
linestyle.use_dashed_line = True
linestyle.dash1 = dash_len
linestyle.gap1 = gap_len
lineset.group = bpy.data.groups["ccircle_group"]

pl_lw = 2.0
# line set 3 (plane, border, visible)
lineset = freestyle.linesets["LineSet 3"]
lineset.select_silhouette = False
lineset.select_border = False
lineset.select_crease = False
lineset.select_by_group = True
lineset.visibility = 'VISIBLE'
linestyle = bpy.data.linestyles["LineStyle.002"]
linestyle.caps = "ROUND"
linestyle.use_chaining = False
linestyle.color = cc_cl
lineset.group = bpy.data.groups["plane_group"]
linestyle.thickness = pl_lw

# line set 4 (plane, border, hidden)
lineset = freestyle.linesets["LineSet 4"]
lineset.select_silhouette = False
lineset.select_border = False
lineset.select_crease = False
lineset.select_by_group = True
lineset.visibility = 'HIDDEN'
linestyle = bpy.data.linestyles["LineStyle.003"]
linestyle.caps = "ROUND"
#linestyle.use_chaining = True
linestyle.color = cc_cl
linestyle.use_dashed_line = True
linestyle.dash1 = 5#dash_len
linestyle.gap1 = 7#gap_len
lineset.group = bpy.data.groups["plane_group"]
linestyle.thickness = pl_lw

en_cl = [0.252,0.776,0.088]
# line set 5 (envelope, silhouette, visible)
lineset = freestyle.linesets["LineSet 5"]
lineset.select_silhouette = True
lineset.select_border = False
lineset.select_crease = False
lineset.select_by_group = True
lineset.visibility = 'VISIBLE'
linestyle = bpy.data.linestyles["LineStyle.004"]
linestyle.caps = "ROUND"
linestyle.color = en_cl
lineset.group = bpy.data.groups["envelope_group"]

# line set 6 (envelope, silhouette, hidden)
lineset = freestyle.linesets["LineSet 6"]
lineset.select_silhouette = True
lineset.select_border = False
lineset.select_crease = False
lineset.select_by_group = True
lineset.visibility = 'HIDDEN'
linestyle = bpy.data.linestyles["LineStyle.005"]
linestyle.caps = "ROUND"
linestyle.color = en_cl
linestyle.use_dashed_line = True
linestyle.dash1 = dash_len
linestyle.gap1 = gap_len
lineset.group = bpy.data.groups["envelope_group"]

sp_cl = [0.799,0.396,0.159]
# line set 7 (sphere, silhouette, hidden)
lineset = freestyle.linesets["LineSet 7"]
lineset.select_silhouette = True
lineset.select_border = False
lineset.select_crease = False
lineset.select_by_group = True
lineset.visibility = 'RANGE'
lineset.qi_start = 0
lineset.qi_end = 1
linestyle = bpy.data.linestyles["LineStyle.006"]
linestyle.caps = "ROUND"
linestyle.color = sp_cl
lineset.group = bpy.data.groups["sphere_group"]

# line set 8 (sphere, silhouette, hidden)
lineset = freestyle.linesets["LineSet 8"]
lineset.select_silhouette = True
lineset.select_border = False
lineset.select_crease = False
lineset.select_by_group = True
lineset.visibility = 'RANGE'
lineset.qi_start = 2
lineset.qi_end = 100
linestyle = bpy.data.linestyles["LineStyle.007"]
linestyle.caps = "ROUND"
linestyle.color = sp_cl
linestyle.use_dashed_line = True
linestyle.dash1 = 12#dash_len
linestyle.gap1 = linestyle.dash1#gap_len
lineset.group = bpy.data.groups["sphere_group"]

# line set 9 (spine, edge mark/silhouette, hidden)
lineset = freestyle.linesets["LineSet 9"]
lineset.select_silhouette = True
lineset.select_border = False
lineset.select_crease = False
lineset.select_edge_mark = True
lineset.select_by_group = True
lineset.visibility = 'VISIBLE'#'RANGE'
linestyle = bpy.data.linestyles["LineStyle.008"]
linestyle.caps = "ROUND"
linestyle.color = [0,0,0]
linestyle.use_chaining = True
lineset.group = bpy.data.groups["spine_group"]
lineset.show_render = False

# line set 10 (spine, edge mark/silhouette, hidden)
lineset = freestyle.linesets["LineSet 10"]
lineset.select_silhouette = True
lineset.select_border = False
lineset.select_crease = False
lineset.select_edge_mark = True
lineset.select_by_group = True
lineset.visibility = 'HIDDEN'
linestyle = bpy.data.linestyles["LineStyle.009"]
linestyle.caps = "ROUND"
linestyle.color = [0,0,0]
linestyle.use_dashed_line = True
linestyle.dash1 = dash_len
linestyle.gap1 = gap_len
lineset.group = bpy.data.groups["spine_group"]
lineset.show_render = False



# line set 11 (envelope, silhouette, visible)
lineset = freestyle.linesets["LineSet 11"]
lineset.select_silhouette = False
lineset.select_border = True
lineset.select_crease = False
lineset.select_by_group = True
lineset.visibility = 'VISIBLE'
linestyle = bpy.data.linestyles["LineStyle.010"]
linestyle.caps = "ROUND"
linestyle.color = en_cl
lineset.group = bpy.data.groups["envelope_group"]
linestyle.thickness = 1.5


# line set 12 (envelope, silhouette, hidden)
lineset = freestyle.linesets["LineSet 12"]
lineset.select_silhouette = False
lineset.select_border = True
lineset.select_crease = False
lineset.select_by_group = True
lineset.visibility = 'HIDDEN'
linestyle = bpy.data.linestyles["LineStyle.011"]
linestyle.caps = "ROUND"
linestyle.color = en_cl
linestyle.use_dashed_line = True
linestyle.dash1 = dash_len
linestyle.gap1 = gap_len
lineset.group = bpy.data.groups["envelope_group"]
linestyle.thickness = 1.5

dl0 = 15
gl0 = 10
sdl = 0.3*dl0
tchk0 = 5.5
stchk = 0.1*tchk0
nlayers = 3
for i in range(nlayers):
    bpy.ops.scene.freestyle_lineset_add()
    lineset = freestyle.linesets["LineSet "+str(13+i)]
    lineset.select_silhouette = False
    lineset.select_border = False
    lineset.select_crease = False
    lineset.select_edge_mark = True
    lineset.select_by_group = True
    lineset.group = bpy.data.groups["spine_group"]
    if i == 0:
        lineset.visibility = 'VISIBLE'
    else:
        lineset.visibility = 'RANGE'
    lineset.qi_start = i
    if i < nlayers-1:
        lineset.qi_end = i
    else:
        lineset.qi_end = 100
    linestyle = bpy.data.linestyles["LineStyle."+format(12+i,'03')]
    linestyle.caps = "ROUND"
    linestyle.use_dashed_line = (i > 0)
    linestyle.dash1 = dl0 - (i-1)*sdl
    linestyle.gap1 = gl0
    linestyle.thickness = tchk0 - i*stchk
    
for linestyle in bpy.data.linestyles:
    linestyle.geometry_modifiers["Sampling"].sampling = 0.1

##########################################
# LIGHTING
##########################################
# Create new lamp datablock
lamp_data = bpy.data.lamps.new(name="New Lamp", type='POINT')
lamp_data.energy = 1.0
lamp_data.shadow_method = 'NOSHADOW'


# Create new object with our lamp datablock
lamp_object = bpy.data.objects.new(name="New Lamp", object_data=lamp_data)

# Link lamp object to the scene so it'll appear in this scene
bpy.context.scene.objects.link(lamp_object)

# Place lamp to a specified location
lamp_object.location = [-3.83856, -4.3118, 6.04704]

# And finally select it make active
lamp_object.select = True
bpy.context.scene.objects.active = lamp_object
