import bpy
import numpy as np
from numpy.polynomial.chebyshev import chebval
import sys
sys.path.append('/stck/bandrieu/Bureau/Python/mylibs/')
sys.path.append('/d/bandrieu/GitHub/These/memoire/figures/code/')
import chebyshev_lib as cheblib
import my_lib as myl



def fr(x):
    return 0.2*(1.0 + 1.2*x[0] + 0.2*x[1] - 0.6*x[2])
    

b = np.array([
    [0., -0.2570,  0.0],
    [1.,  0.3086, -0.6],
    [2.,  0.3045, -0.6],
    [3., -0.1435,  0.4],
    [4.,  0.0814,  0.6]])

N = len(b)-1
B2C = cheblib.B2Cmatrix(N+1)
pg = np.zeros(b.shape)
for j in range(3):
    for i in range(N+1):
        for k in range(N+1):
            pg[i,j] += B2C[i,k]*b[k,j]
dpg = cheblib.diff(pg)
d2pg = cheblib.diff(dpg)

m = 100
n = 100
m1 = 10
m2 = 60
dm = 6
u = np.linspace(-1,1,m)
#u = np.linspace(-0.8,0.2,m)
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
e = np.zeros((m2-m1+1,n,3)) #np.zeros((m,n,3))
bu0 = np.array([-1.,0.,0.15])
for i in range(m1,m2+1):
    normdg = np.sqrt(np.sum(np.power(dg[:,i],2)))
    sina = dr[i]/normdg
    occ = g[:,i] - r[i]*sina*dg[:,i]/normdg
    rcc = r[i]*np.sqrt(1. - sina**2)
    bw = dg[:,i]/normdg
    bu = bu0 - np.dot(bu0,bw)*bw
    #bu = d2g[:,i] - np.dot(d2g[:,i],bw)*bw
    bu = bu / np.sqrt(np.sum(np.power(bu,2)))
    bv = np.cross(bw,bu)
    for j in range(n):
        e[i-m1,j] = occ + rcc*(np.cos(v[j])*bu + np.sin(v[j])*bv)

##############################
# clear blender scene
myl.blankScene()

# add spine curve
myl.addPolyline(g[:,m1-dm:m2+dm+1].T, [0,0,0], 1.5e-3, 0)
bpy.data.materials["Mat_line"].use_transparency = True
bpy.data.materials["Mat_line"].alpha = 0

# add envelope ############
verts = []
faces = []
for i in range(e.shape[0]):
    for j in range(n):
        verts.append([e[i,j,0], e[i,j,1], e[i,j,2]])
        if i < e.shape[0]-1:
            faces.append([i*n + j, i*n + (j+1)%n, (i+1)*n + (j+1)%n, (i+1)*n + j])
#create mesh and object
mesh = bpy.data.meshes.new("envelope")
object = bpy.data.objects.new("envelope",mesh)

#set mesh location
object.location = bpy.context.scene.cursor_location
bpy.context.scene.objects.link(object)

#create mesh from python data
mesh.from_pydata(verts,[],faces)
mesh.update(calc_edges=True)

# show mesh as smooth
object.data.use_auto_smooth = 1
#bpy.ops.object.shade_smooth()
mypolys = mesh.polygons
for q in mypolys:
    q.use_smooth = True

mat = bpy.data.materials.new("mat_enve")
mat.diffuse_color = [0.527, 0.800, 0.213]
mat.diffuse_intensity = 1
mat.specular_intensity = 0
mat.use_transparency = True
mat.raytrace_transparency.fresnel = 2.7
mat.alpha = 0.68
mat.emit = 1.0

myl.setMaterial(object, mat)
###########################


# add sphere ##############
g0 = chebval(u0, pg)
r0 = chebval(u0, pr)
bpy.ops.mesh.primitive_uv_sphere_add(location=g0, segments=100, ring_count=100, size=0.995*r0)
#bpy.context.object.scale = (r0,r0,r0)
myl.setSmooth(bpy.context.object)
mat = bpy.data.materials.new("mat_sph")
mat.diffuse_color = [0.800, 0.494, 0.317]
mat.diffuse_intensity = 1
mat.specular_intensity = 0
mat.use_transparency = True
mat.raytrace_transparency.fresnel = 2.1
mat.alpha = 1
mat.emit = 1.2
myl.setMaterial(bpy.context.object, mat)
###########################


# add point ##############
g0 = chebval(u0, pg)
r0 = chebval(u0, pr)
bpy.ops.mesh.primitive_uv_sphere_add(location=g0, segments=50, ring_count=50, size=1.e-2)
#bpy.context.object.scale = (r0,r0,r0)
myl.setSmooth(bpy.context.object)
mat = myl.lineMaterial("mat_point",[0,0,0])
myl.setMaterial(bpy.context.object, mat)
###########################


# add plane ###############
dg0 = chebval(u0, dpg)
dr0 = chebval(u0, dpr)

normdg = np.sqrt(np.sum(np.power(dg0,2)))
sina = dr0/normdg
occ = g0 - r0*sina*dg0/normdg
rcc = r0*np.sqrt(1. - sina**2)
bw = dg0/normdg
#bu = np.array([0,1,0]) - bw[1]*bw
bu = bu0 - np.dot(bu0,bw)*bw
bu = bu / np.sqrt(np.sum(np.power(bu,2)))
bv = np.cross(bw,bu)

# tangent to spine (normal to plane)
tng = np.vstack([g0, g0+dg0])
myl.addPolyline(tng, [1,0,0], 1.5e-3, 0)

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
myl.setMaterial(object, mat)
###########################


###########################
"""
t = np.linspace(-1,1,20)
v = chebval(t, pg)
#r = chebval(t, pr)
r = fr(v)
print(r.shape)
for i in range(len(t)):
    bpy.ops.mesh.primitive_uv_sphere_add(location=v[:,i], segments=32, size=r[i])
"""
###########################

scene = bpy.context.scene

az = (20.0 - 90.0)*np.pi/180.0
el = 0.0#-5.0*np.pi/180.0
xcam = np.array([1.0000, 0.1067, -0.3586])#g0
rcam = 18.959#34.6*r0
#scene.camera.location = xcam + rcam*np.array([np.cos(az)*np.cos(el), np.sin(az)*np.cos(el), np.sin(el)])
scene.camera.location = [14.6387, -19.52351, 0.52857]
#scene.camera.rotation_euler = np.array([89.98, 0.62, 379.9])*np.pi/180.0
scene.camera.rotation_euler = np.array([87.975, 0.62, 393.916])*np.pi/180.0
bpy.data.cameras["Camera"].lens_unit = "FOV"
bpy.data.cameras["Camera"].angle = 6.609*np.pi/180.0


scene.render.resolution_x = 1000
scene.render.resolution_y = 0.75*scene.render.resolution_x
scene.render.resolution_percentage = 100

bpy.data.worlds["World"].horizon_color = [1,1,1]
scene.render.use_border = False
scene.render.use_freestyle = True
scene.render.alpha_mode = 'TRANSPARENT'


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
linestyle.dash1 = 5
linestyle.gap1 = linestyle.dash1
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

# line set 4 (plane, border, visible)
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
linestyle.dash1 = 5
linestyle.gap1 = linestyle.dash1
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
linestyle.dash1 = 5
linestyle.gap1 = linestyle.dash1
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
linestyle.dash1 = 5
linestyle.gap1 = linestyle.dash1
lineset.group = bpy.data.groups["sphere_group"]

# line set 9 (spine, silhouette, hidden)
lineset = freestyle.linesets["LineSet 9"]
lineset.select_silhouette = True
lineset.select_border = False
lineset.select_crease = False
lineset.select_by_group = True
lineset.visibility = 'RANGE'
linestyle = bpy.data.linestyles["LineStyle.008"]
linestyle.caps = "ROUND"
linestyle.color = [0,0,0]
linestyle.use_chaining = True
lineset.group = bpy.data.groups["spine_group"]

# line set 10 (spine, silhouette, hidden)
lineset = freestyle.linesets["LineSet 10"]
lineset.select_silhouette = True
lineset.select_border = False
lineset.select_crease = False
lineset.select_by_group = True
lineset.visibility = 'HIDDEN'
linestyle = bpy.data.linestyles["LineStyle.009"]
linestyle.caps = "ROUND"
linestyle.color = [0,0,0]
linestyle.use_dashed_line = True
linestyle.dash1 = 5
linestyle.gap1 = linestyle.dash1
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
linestyle.dash1 = 5
linestyle.gap1 = linestyle.dash1
lineset.group = bpy.data.groups["envelope_group"]
linestyle.thickness = 1.5

