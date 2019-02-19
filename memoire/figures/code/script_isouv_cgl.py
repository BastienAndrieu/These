import bpy
import sys
sys.path.append('/stck/bandrieu/Bureau/Python/mylibs/')
sys.path.append('/d/bandrieu/GitHub/These/memoire/figures/code/')
import my_lib as myl
import my_lib1 as mylb
import chebyshev as cheb
import chebyshev_lib as cheblib
import numpy as np

bezier_poly = bpy.data.objects["SurfPatch"]
bpy.context.scene.objects.active = bezier_poly
bezier_poly.select = True

bpy.ops.object.transform_apply(location=True, rotation=True)

bezier_poly = bpy.data.curves["SurfPatch"]
bezier_poly.splines[0].use_endpoint_u = True
bezier_poly.splines[0].use_endpoint_v = True
mb = bezier_poly.splines[0].point_count_u #order_u
nb = bezier_poly.splines[0].point_count_v #order_v
B = np.zeros((mb,nb,3))
for l, p in enumerate(bezier_poly.splines[0].points):
        i = l%mb
        j = int((l-i)/mb)
        for k in range(3):
            B[i,j,k] = p.co[k]
M = cheb.B2Cmatrix(mb)
N = cheb.B2Cmatrix(nb)
C = np.zeros((mb,nb,3))
for k in range(3):
    C[:,:,k] = mylb.mymatmul(mylb.mymatmul(M, B[:,:,k]), N.T)
duC, dvC = cheblib.diff2(C)

M = 6
N = 5
uc = -np.cos(np.arange(M+1)*np.pi/float(M))
vc = -np.cos(np.arange(N+1)*np.pi/float(N))

m = 201
n = m
u = np.linspace(-1.,1.,m)
v = np.linspace(-1.,1.,n)

eps = 1.e-13

k = 0
listu = []
for i in range(m-1):
        if u[i] < uc[k]+eps and u[i+1] > uc[k]:
                if u[i+1] - uc[k] > uc[k] - u[i]:
                        u[i] = uc[k]
                        listu.append(i)
                else:
                        u[i+1] = uc[k]
                        listu.append(i+1)
                k += 1
listu.append(m-1)

k = 0
listv = []
for j in range(n-1):
        if v[j] < vc[k]+eps and v[j+1] > vc[k]:
                if v[j+1] - vc[k] > vc[k] - v[j]:
                        v[j] = vc[k]
                        listv.append(j*m)
                else:
                        v[j+1] = vc[k]
                        listv.append((j+1)*m)
                k += 1
listv.append((n-1)*m)


x = np.polynomial.chebyshev.chebgrid2d(u, v, C[:,:,0])
y = np.polynomial.chebyshev.chebgrid2d(u, v, C[:,:,1])
z = np.polynomial.chebyshev.chebgrid2d(u, v, C[:,:,2])
verts = []
for j in range(n):
        for i in range(m):
                verts.append([x[i,j], y[i,j], z[i,j]])
                
faces = []
for j in range(n-1):
        for i in range(m-1):
                faces.append([j*m+i, j*m+i+1, (j+1)*m+i+1, (j+1)*m+i])

#create mesh and object
mesh = bpy.data.meshes.new("mypatch")
obj = bpy.data.objects.new("mypatch",mesh)

bpy.context.scene.objects.link(obj)

#create mesh from python data
mesh.from_pydata(verts,[],faces)
mesh.update(calc_edges=True)

# show mesh as smooth
obj.data.use_auto_smooth = 1
mypolys = mesh.polygons
for q in mypolys:
        q.use_smooth = True

# material
mat = myl.surfaceMaterial("myMat_Patch", [1,1,1], 1)
myl.setMaterial(obj,mat)
    
##################################################
patch = bpy.data.objects["mypatch"]

scene = bpy.context.scene
scene.objects.active = patch
patch.select = True
bpy.ops.object.material_slot_remove()
patch.data.materials.append(bpy.data.materials["Material.001"])

################################################
# add nodes
mat = myl.lineMaterial("mat_nodes", [0,0,0])

xc = np.polynomial.chebyshev.chebgrid2d(uc, vc, C[:,:,0])
yc = np.polynomial.chebyshev.chebgrid2d(uc, vc, C[:,:,1])
zc = np.polynomial.chebyshev.chebgrid2d(uc, vc, C[:,:,2])

nod_rad = 3.e-2
nsegm = 32
nring = 32
for j in range(N+1):
        for i in range(M+1):
               bpy.ops.mesh.primitive_uv_sphere_add(
                       location=[xc[i,j], yc[i,j], zc[i,j]],
                       segments=nsegm,
                       ring_count=nring,
                       size=nod_rad)
               myl.setSmooth(bpy.context.object)
               myl.setMaterial(bpy.context.object, mat)
################################################

################################################
# add tangents and normal
i = 2
j = 3
su = np.polynomial.chebyshev.chebval2d(uc[i], vc[j], duC)
sv = -np.polynomial.chebyshev.chebval2d(uc[i], vc[j], dvC)
sw = np.cross(su,sv)
vec = np.vstack((su,sv,sw))
s = np.array([xc[i,j], yc[i,j], zc[i,j]])

sclvec = 0.4
thck = 1.e-2

I3 = np.eye(3)
for i in range(3):
        myl.addPolyline(np.vstack([s, s+sclvec*vec[i]]), I3[i], thck, 0)


xij = bpy.data.objects.new("xij", None)
xij.location = s
bpy.context.scene.objects.link(xij)

xu = bpy.data.objects.new("xu", None)
xu.location = s + sclvec*su
bpy.context.scene.objects.link(xu)

xv = bpy.data.objects.new("xv", None)
xv.location = s + sclvec*sv
bpy.context.scene.objects.link(xv)

xw = bpy.data.objects.new("n", None)
xw.location = s + sclvec*sw
bpy.context.scene.objects.link(xw)
################################################

# Groups
for ob in scene.objects:
    ob.select = False

patch.select = True
bpy.ops.group.create(name="patch_group")
patch.select = False


################################################
# Mark Freestyle Edges

# iso-u/v ############################
scene.objects.active = patch
patch.select = True
mesh = bpy.data.meshes["mypatch"]

bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='DESELECT')
bpy.ops.mesh.select_mode(type="EDGE")
bpy.ops.object.mode_set(mode='OBJECT')

for e in mesh.edges:
    v1 = e.vertices[0]
    v2 = e.vertices[1]
    p = min(v1,v2)
    q = max(v1,v2)
    if q == p + m and p in listu:
        e.select = True
    elif q == p + 1 and p in listv:
        e.select = True
        
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.loop_multi_select(ring=False)
bpy.ops.mesh.mark_freestyle_edge(clear=False)
bpy.ops.object.mode_set(mode='OBJECT')
patch.select = False


################################################
# Freestyle
scene.render.use_freestyle = True
freestyle = scene.render.layers.active.freestyle_settings
freestyle.use_smoothness = True

bpy.ops.scene.freestyle_lineset_add() #2


lw = 1.5
clu = [0,0,1]
clv = [1,0,0]

# line set 2 (iso-uv)
lineset = freestyle.linesets["LineSet 2"]
lineset.select_silhouette = False
lineset.select_border = False
lineset.select_crease = False
lineset.select_edge_mark = True
lineset.select_by_group = True
lineset.group = bpy.data.groups["patch_group"]
lineset.visibility = 'RANGE'
linestyle = bpy.data.linestyles["LineStyle.001"]
linestyle.caps = "ROUND"
linestyle.color = [0,0,0]
linestyle.thickness = lw
linestyle.use_dashed_line = True
linestyle.dash1 = 3
linestyle.gap1 = 7

for linestyle in bpy.data.linestyles:
    linestyle.geometry_modifiers["Sampling"].sampling = 0.02
