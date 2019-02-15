import bpy
import sys
sys.path.append('/stck/bandrieu/Bureau/Python/mylibs/')
import my_lib as myl
import my_lib1 as mylb
import chebyshev as cheb
import numpy as np


step = 10
niso = 9

m = step*niso + 1
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
#C = myl.NURBS2Chebyshev(bpy.data.curves["SurfPatch"])

mylb.writeCoeffs2(C, '/d/bandrieu/GitHub/These/memoire/figures/code/parametric_patch.cheb')
myl.addPatch(C, m, [1,1,1], 1)

bezier_poly = bpy.data.objects["SurfPatch"]
patch = bpy.data.objects["mypatch"]

patch.location = bezier_poly.location
patch.rotation_euler = bezier_poly.rotation_euler
    
scene = bpy.context.scene
scene.objects.active = patch
patch.select = True
bpy.ops.object.material_slot_remove()
patch.data.materials.append(bpy.data.materials["Material.001"])
bpy.ops.object.duplicate_move()

patch_dummy = bpy.data.objects["mypatch.001"]
mat = myl.surfaceMaterial("mat_dummy",np.ones(3),0,0)
mat.specular_intensity = 0
mat.use_raytrace = False
mat.use_shadows = False
mat.use_cast_buffer_shadows = False
bpy.ops.object.material_slot_remove()
patch_dummy.data.materials.append(mat)

################################################
verts = []
faces = []
for j in range(nb):
    for i in range(mb):
        verts.append([B[i,j,0], B[i,j,1], B[i,j,2]])
        if j < nb-1 and i < mb-1:
            faces.append([j*mb+i, j*mb+i+1, (j+1)*mb+i+1, (j+1)*mb+i])
#create mesh and object
mesh = bpy.data.meshes.new("bezier_polygon")
obj = bpy.data.objects.new("bezier_polygon",mesh)
#set mesh location
obj.location = bezier_poly.location
obj.rotation_euler = bezier_poly.rotation_euler
bpy.context.scene.objects.link(obj)
#create mesh from python data
mesh.from_pydata(verts,[],faces)
mesh.update(calc_edges=True)
# material
myl.setMaterial(obj,bpy.data.materials["mat_dummy"])
################################################

# Groups
for ob in scene.objects:
    ob.select = False

patch.select = True
bpy.ops.group.create(name="patch_group")
patch.select = False

patch_dummy.select = True
bpy.ops.group.create(name="patch_dummy_group")
patch_dummy.select = False

bez_ply = bpy.data.objects["bezier_polygon"]
bez_ply.select = True
bpy.ops.group.create(name="bezier_polygon_group")
bez_ply.select = False

################################################
# Mark Freestyle Edges

# iso-u ############################
scene.objects.active = patch
patch.select = True
mesh = bpy.data.meshes["mypatch"]

bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='DESELECT')
bpy.ops.mesh.select_mode(type="EDGE")
bpy.ops.object.mode_set(mode='OBJECT')

for i in range(len(mesh.edges)):
    v1 = mesh.edges[i].vertices[0]
    v2 = mesh.edges[i].vertices[1]
    p = min(v1,v2)
    q = max(v1,v2)
    if q == p + m and (p%step) == 0 and p < m:
        mesh.edges[i].select = True
        
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.loop_multi_select(ring=False)
bpy.ops.mesh.mark_freestyle_edge(clear=False)
bpy.ops.object.mode_set(mode='OBJECT')
patch.select = False

# iso-v ############################
scene.objects.active = patch_dummy
patch_dummy.select = True
mesh = bpy.data.meshes["mypatch.001"]

bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='DESELECT')
bpy.ops.mesh.select_mode(type="EDGE")
bpy.ops.object.mode_set(mode='OBJECT')

for i in range(len(mesh.edges)):
    v1 = mesh.edges[i].vertices[0]
    v2 = mesh.edges[i].vertices[1]
    p = min(v1,v2)
    q = max(v1,v2)
    if q == p + 1 and (p%(step*m)) == 0:
        mesh.edges[i].select = True
        
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.loop_multi_select(ring=False)
bpy.ops.mesh.mark_freestyle_edge(clear=False)
bpy.ops.object.mode_set(mode='OBJECT')
patch_dummy.select = False

# bezier control polygon
scene.objects.active = bez_ply
bez_ply.select = True
mesh = bpy.data.meshes["bezier_polygon"]
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.select_mode(type="EDGE")
bpy.ops.mesh.mark_freestyle_edge(clear=False)
bpy.ops.object.mode_set(mode='OBJECT')
bez_ply.select = False
################################################

################################################
# Freestyle
scene.render.use_freestyle = True
freestyle = scene.render.layers.active.freestyle_settings
freestyle.use_smoothness = True

bpy.ops.scene.freestyle_lineset_add() #2
bpy.ops.scene.freestyle_lineset_add() #3
bpy.ops.scene.freestyle_lineset_add() #4
bpy.ops.scene.freestyle_lineset_add() #5

lw = 1.5
clu = [1,0,0]
clv = [0,0,1]

# line set 2 (iso-u)
lineset = freestyle.linesets["LineSet 2"]
lineset.select_silhouette = False
lineset.select_border = False
lineset.select_crease = False
lineset.select_edge_mark = True
lineset.select_by_group = True
lineset.visibility = 'RANGE'
linestyle = bpy.data.linestyles["LineStyle.001"]
linestyle.caps = "ROUND"
linestyle.color = clu
linestyle.thickness = lw
lineset.group = bpy.data.groups["patch_group"]

# line set 3 (iso-v)
lineset = freestyle.linesets["LineSet 3"]
lineset.select_silhouette = False
lineset.select_border = False
lineset.select_crease = False
lineset.select_edge_mark = True
lineset.select_by_group = True
lineset.visibility = 'RANGE'
linestyle = bpy.data.linestyles["LineStyle.002"]
linestyle.caps = "ROUND"
linestyle.color = clv
linestyle.thickness = lw
lineset.group = bpy.data.groups["patch_dummy_group"]

cl_bez = [0,0,0]
lw_bez = 3.0
dl_bez = 3
# line set 4 (bezier control polygon)
lineset = freestyle.linesets["LineSet 4"]
lineset.select_silhouette = False
lineset.select_border = False
lineset.select_crease = False
lineset.select_edge_mark = True
lineset.select_by_group = True
lineset.visibility = 'VISIBLE'
linestyle = bpy.data.linestyles["LineStyle.003"]
linestyle.caps = "ROUND"
linestyle.color = cl_bez
linestyle.thickness = lw_bez
lineset.group = bpy.data.groups["bezier_polygon_group"]

# line set 5 (bezier control polygon)
lineset = freestyle.linesets["LineSet 5"]
lineset.select_silhouette = False
lineset.select_border = False
lineset.select_crease = False
lineset.select_edge_mark = True
lineset.select_by_group = True
lineset.visibility = 'HIDDEN'
linestyle = bpy.data.linestyles["LineStyle.004"]
linestyle.caps = "ROUND"
linestyle.color = cl_bez
linestyle.thickness = lw_bez
linestyle.use_dashed_line = True
linestyle.dash1 = dl_bez
linestyle.gap1 = linestyle.dash1
lineset.group = bpy.data.groups["bezier_polygon_group"]
