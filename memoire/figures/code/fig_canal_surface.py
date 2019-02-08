import bpy
import numpy as np
from numpy.polynomial.chebyshev import chebval
import sys
sys.path.append('/stck/bandrieu/Bureau/Python/mylibs/')
sys.path.append('/d/bandrieu/GitHub/These/memoire/figures/code/')
import chebyshev_lib as cheblib
import my_lib as myl



def fr(x):
    return 0.2*(1.0 + 1.5*x[0] + 0.2*x[1] - 0.6*x[2])
    

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
e = np.zeros((m,n,3))
bu0 = np.array([-1.,0.,0.15])
for i in range(m):
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
        e[i,j] = occ + rcc*(np.cos(v[j])*bu + np.sin(v[j])*bv)

##############################
# clear blender scene
myl.blankScene()

# add spine curve
myl.addPolyline(g.T, [0,0,0], 3.e-3, 0)

# add envelope ############
verts = []
faces = []
for i in range(m):
    for j in range(n):
        verts.append([e[i,j,0], e[i,j,1], e[i,j,2]])
        if i < m-1:
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
mat.raytrace_transparency.fresnel = 3.4
mat.alpha = 0.5
mat.emit = 1.0

myl.setMaterial(object, mat)
###########################


# add sphere ##############
g0 = chebval(u0, pg)
r0 = chebval(u0, pr)
bpy.ops.mesh.primitive_uv_sphere_add(location=g0, segments=100, size=r0)
#bpy.context.object.scale = (r0,r0,r0)
myl.setSmooth(bpy.context.object)
mat = bpy.data.materials.new("mat_sph")
mat.diffuse_color = [0.800, 0.494, 0.317]
mat.diffuse_intensity = 1
mat.specular_intensity = 0
mat.use_transparency = True
mat.raytrace_transparency.fresnel = 2.1
mat.alpha = 1
mat.emit = 0.8

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
###########################


# add char. circle ########
verts = []
faces = []
rcc *= 1.003
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
scene.camera.location = xcam + rcam*np.array([np.cos(az)*np.cos(el), np.sin(az)*np.cos(el), np.sin(el)])
scene.camera.rotation_euler = np.array([89.98, 0.62, 379.9])*np.pi/180.0
bpy.data.cameras["Camera"].lens_unit = "FOV"
bpy.data.cameras["Camera"].angle = 6.609*np.pi/180.0


scene.render.resolution_x = 800
scene.render.resolution_y = 600
scene.render.resolution_percentage = 100

bpy.data.worlds["World"].horizon_color = [1,1,1]
scene.render.use_border = False
scene.render.use_freestyle = True
#scene.render.alpha_mode = 'TRANSPARENT'



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



bpy.ops.scene.freestyle_lineset_add()
bpy.ops.scene.freestyle_lineset_add()
bpy.ops.scene.freestyle_lineset_add()
bpy.ops.scene.freestyle_lineset_add()

# line set 1 (ccircle, border, visible)
select_silhouette = False
select_crease = False
select_by_group = True

# line set 2 (ccircle, border, hidden)

# line set 3 (plane, border, visible)

# line set 4 (plane, border, visible)

# line set 5 (envelope, silhouette, visible)



