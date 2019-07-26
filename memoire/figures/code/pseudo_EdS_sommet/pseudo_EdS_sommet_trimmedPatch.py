import bpy
from mathutils import Vector, Matrix

import numpy
import numpy.linalg
from numpy.polynomial.chebyshev import chebgrid2d, chebval2d

import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python/')
import lib_blender_util as lbu
import lib_blender_edit as lbe
import lib_blender_figures as lbf
import lib_chebyshev as lcheb
import lib_color as lco
import lib_brep as lbrep
import lib_linalg
from lib_compgeom import circumcircle, complete_orthonormal_matrix

################################################################
class Vertex:
    def __init__(self, xyz, edge, faces):
        self.xyz = xyz
        self.edge = edge
        self.faces = faces
        return
################################################################

################################################################
def bilinear_patch(a, b, c, d, u, v):
    xyz = numpy.zeros((3,len(u),len(v)))
    for k in range(3):
        xyz[k] = 0.25*(
            a[k]*numpy.outer(1.0-u, 1.0-v) + 
            b[k]*numpy.outer(1.0+u, 1.0-v) + 
            c[k]*numpy.outer(1.0+u, 1.0+v) + 
            d[k]*numpy.outer(1.0-u, 1.0+v) )
    return xyz
################################################################


################################################################
pthin = '/d/bandrieu/GitHub/FFTsurf/test/demo_EoS_brep/'
pthimg = '/d/bandrieu/GitHub/These/memoire/figures/images/pseudo_EdS_sommet/'

ivert = 7
rho = 0.11
################################################################


#################################################
# CHECKER TEXTURE
imgchecker = bpy.data.images.load(filepath='/d/bandrieu/GitHub/These/memoire/figures/code/BRep/checker.png')
texchecker = bpy.data.textures.new('texture_checker', 'IMAGE')
texchecker.image = imgchecker
#################################################


################################################################
# SET SCENE
scene = bpy.context.scene
lbu.clear_scene(meshes=True, lamps=True, cameras=False)

resx = 1024
resy = 0.75*resx

lbu.set_scene(
    resolution_x=resx,
    resolution_y=resy,
    resolution_percentage=100,
    alpha_mode='SKY',
    horizon_color=(1,1,1),
    light_samples=4,#16,
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
#cam.rotation_euler = numpy.radians((66.7, 0.778, 132.2))
cam.rotation_euler = numpy.radians((66.7, 0.778, 127.3))
cam.data.angle = numpy.radians(37.72)
################################################################


################################################################
# LOAD VERTICES
verts = []
f = open(pthin + 'debug/verts.dat', 'r')
while True:
    line = f.readline()
    if line == '': break
    num = [float(a) for a in line.split()]
    verts.append(
        Vertex(
            xyz=num[0:3],
            edge=[int(a)-1 for a in num[3:5]],
            faces=[int(a) for a in num[5:]]
        )
    )
f.close()
################################################################


V = verts[ivert - 1]
#print('faces = ', V.faces)

#bpy.ops.mesh.primitive_uv_sphere_add(location=v.xyz, size=1e-2)


################################################################
# GET UV COORDS IN INCIDENT FACES
Vuv = []

halfedges = lbrep.read_halfedges(pthin + 'debug/edges.dat')
curves = lbrep.read_curves(pthin + 'debug/edges_uv.dat', pthin + 'debug/edges_xyz.dat')

ih = 2*V.edge[0] + V.edge[1]
while True:
    #
    if ih%2 == 0:
        uv = curves[int(ih/2)].uv[:,0:2]
    else:
        uv = curves[int((ih-1)/2)].uv[::-1,2:4]
    #
    Vuv.append(uv[0])
    #
    ih = halfedges[ih].prev
    ih = halfedges[ih].twin
    if ih == 2*V.edge[0] + V.edge[1]: break
################################################################





################################################################
# LOAD INCIDENT FACES AND MAKE OFFSET FACES
color_face = numpy.loadtxt('/d/bandrieu/GitHub/These/memoire/figures/data/BRep/face_color.dat')
color_face = lco.cc_hsv(color_face, fs=1.2, fv=1.0)

m = 100
u = numpy.linspace(-1,1,m)

nf = len(V.faces)

corners = []

for iloc, iface in enumerate(V.faces):
    strf = format(iface,'03')

    mat = bpy.data.materials.new('mat_face_'+strf)
    mat.diffuse_color = color_face[iface-1]
    
    c = lcheb.read_polynomial2(pthin + 'brepmesh/c_' + strf + '.cheb')

    xyz = chebgrid2d(u, u, c)
    mverts, mfaces = lbu.tensor_product_mesh_vf(xyz[0], xyz[1], xyz[2])
    obj = lbu.pydata_to_mesh(
        mverts,
        mfaces,
        name='face_'+strf
    )
    lbe.set_smooth(obj)
    obj.data.materials.append(mat)

    # OFFSET
    c = lcheb.read_polynomial2(pthin + 'brepmesh_eos/c_' + strf + '.cheb')
    xyz = chebgrid2d(u, u, c)
    mverts, mfaces = lbu.tensor_product_mesh_vf(xyz[0], xyz[1], xyz[2])
    obj = lbu.pydata_to_mesh(
        mverts,
        mfaces,
        name='EdS_face_'+strf
    )
    lbe.set_smooth(obj)
    obj.data.materials.append(mat)

    #
    xyz = chebval2d(Vuv[iloc][0], Vuv[iloc][1], c)
    corners.append(xyz)

# SPHERE RADIUS
rho = 0
for xyz in corners:
    rho += Vector(xyz - V.xyz).length
rho /= float(len(corners))

bpy.ops.mesh.primitive_uv_sphere_add(
    location=V.xyz,
    size=rho
)

for obj in bpy.data.objects:
    if obj.name[0:6] == 'Sphere':
        obj_sphere = obj
        break
obj_sphere.draw_type = 'WIRE'
obj_sphere.show_all_edges = True
obj_sphere.hide_render = True


corner_avg = numpy.zeros(3)
for xyz in corners:
    corner_avg = corner_avg + xyz
corner_avg = corner_avg/float(len(corners))


obj = lbu.pydata_to_polyline(
    [Vector((0,0,0)), corner_avg - V.xyz],
    thickness=2e-3,
    name='corner_avg')
obj.location = V.xyz
obj.hide_render = True
################################################################





################################################################
# OFFSET EDGES
end_planes = []
arc_mid = []
jm = int(m/2)

ih = 2*V.edge[0] + V.edge[1]
while True:
    # do stuff...
    if ih%2 == 0:
        ie = int(ih/2)
    else:
        ie = int((ih-1)/2)
    ie += 1
    c = lcheb.read_polynomial2(pthin + 'debug/eos_edge_c_' + format(ie, '03') + '.cheb')
    if ih%2 != 0:
        c = lcheb.flip_polynomial2(c, flip_u=True, flip_v=True)
    #
    mat = bpy.data.materials.new('mat_edge_'+str(ie))
    mat.diffuse_color = color_face[8+ie]

    xyz = chebgrid2d(u, u, c)
    mverts, mfaces = lbu.tensor_product_mesh_vf(xyz[0], xyz[1], xyz[2])
    obj = lbu.pydata_to_mesh(
        mverts,
        mfaces,
        name='EdS_edge_'+strf
    )
    lbe.set_smooth(obj)
    obj.data.materials.append(mat)
    #
    xyz = chebval2d(-numpy.ones(m), u, c)
    eR = Vector(xyz[:,0])
    e = Vector(xyz[:,jm])
    eL = Vector(xyz[:,-1])
    tng = (eR - e).cross(eL - e)
    tng.normalize()
    occ, rcc = circumcircle(eR, eL, e)
    end_planes.append((tng, occ))
    #
    rot = Matrix(complete_orthonormal_matrix(tng, i=2)).transposed()
    bpy.ops.mesh.primitive_plane_add(
        location=V.xyz,
        rotation=rot.to_euler(),
        radius=rcc
    )
    #
    # arc midpoint and tangent
    xyz = chebval2d(-1, 0, c)
    cu, cv = lcheb.diff2(c)
    xyz_v = chebval2d(-1, 0, cv)
    arc_mid.append((Vector(xyz - V.xyz).normalized(), Vector(xyz_v).normalized()))
    #
    # move on to next incident (outgoing) halfedge
    ih = halfedges[ih].prev
    ih = halfedges[ih].twin
    if ih == 2*V.edge[0] + V.edge[1]: break

mat_plane = bpy.data.materials.new('mat_plane')
mat_plane.use_transparency = True
mat_plane.alpha = 0.5

obj_planes = []
for obj in bpy.data.objects:
    if obj.name[0:5] == 'Plane':
        obj_planes.append(obj)
        #obj.show_axis = True
        obj.show_transparent = True
        obj.hide_render = True
        obj.data.materials.append(mat_plane)

nor_avg = Vector((0,0,0))
for nor, pt in end_planes:
    nor_avg += nor
nor_avg.normalize()


obj = lbu.pydata_to_polyline(
    [Vector((0,0,0)), -rho*nor_avg],
    thickness=2e-3,
    name='nor_avg')
obj.location = V.xyz
obj.hide_render = True


"""
# BOUNDING GREAT CIRCLES
for vec1, vec2 in arc_mid:
    vec3 = vec1.cross(vec2)
    rot = Matrix((vec1, vec2, vec3)).transposed()
    bpy.ops.mesh.primitive_circle_add(
        location=V.xyz,
        rotation=rot.to_euler(),
        radius=1.1*rho,
        vertices=100,
        fill_type='TRIFAN'
    )
"""

# BOUNDING CONE
generators = []
ng = len(arc_mid)
for i in range(ng):
    j = (i+1)%ng
    nori = arc_mid[i][0].cross(arc_mid[i][1])
    norj = arc_mid[j][0].cross(arc_mid[j][1])
    vec = nori.cross(norj).normalized()
    for sens in [-1,1]:
        xyz = sens*vec
        keep = True
        for k in range(ng):
            if k == i or k == j: continue
            nork = arc_mid[k][0].cross(arc_mid[k][1])
            if xyz.dot(nork) < -1e-6:
                keep = False
                break
        if keep:
            generators.append(xyz)
            break

mverts = [Vector((0,0,0))] +  [2*rho*xyz for xyz in generators]
mfaces = [[0, 1+i, 1+(i+1)%len(generators)] for i in range(len(generators))]
cone = lbu.pydata_to_mesh(
    mverts,
    mfaces,
    name='bounding_cone'
)
cone.location = V.xyz
#cone.layers[1] = True
#cone.layers[0] = False

"""
# BISECTOR PLANES
npl = len(end_planes)
mat = numpy.array([
    end_planes[0][0],
    end_planes[1][0],
    end_planes[2][0]
])
rhs = numpy.array([
    end_planes[0][0].dot(end_planes[0][1]),
    end_planes[1][0].dot(end_planes[1][1]),
    end_planes[2][0].dot(end_planes[2][1]),
])

pt = numpy.linalg.solve(mat, rhs)

for i in range(npl):
    nor = end_planes[i][0] + end_planes[(i+1)%npl][0]
    nor = nor.cross(end_planes[i][0].cross(end_planes[(i+1)%npl][0]))
    rot = Matrix(complete_orthonormal_matrix(nor, i=2)).transposed()
    bpy.ops.mesh.primitive_plane_add(
        location=pt,
        rotation=rot.to_euler(),
        radius=rho
    )
"""
################################################################



################################################################
# VERTEX EOS TRIMMED PATCH
iv = ivert + 9 + 16
strv = format(iv, '03')
tri = numpy.loadtxt(pthin + 'brepmesh_eos/tri_' + strv + '.dat', dtype=int)-1
xyz = numpy.loadtxt(pthin + 'brepmesh_eos/xyz_' + strv + '.dat', dtype=float)
uv = numpy.loadtxt(pthin + 'brepmesh_eos/uv_' + strv + '.dat', dtype=float)

mverts = [[x for x in p] for p in xyz]
mfaces = [[int(v) for v in t] for t in tri]
pseudoEdS = lbu.pydata_to_mesh(
    mverts,
    mfaces,
    name='EdS_vert'
)
lbe.set_smooth(pseudoEdS)

# TEXTURE COORDS
scene.objects.active = pseudoEdS
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.uv.unwrap(
    method='ANGLE_BASED',
    fill_holes=True,
    correct_aspect=True,
    use_subsurf_data=False,
    margin=0.001
)
bpy.ops.object.mode_set(mode='OBJECT')
uvlayer = pseudoEdS.data.uv_layers.active
for i, f in enumerate(pseudoEdS.data.polygons):
    for j in range(f.loop_total):
        k = f.loop_start + j
        for l in range(2):
            uvlayer.data[k].uv[l] = 0.5*(uv[f.vertices[j],l] + 1.0)

# MATERIAL
mat = bpy.data.materials.new('mat_pseudoEdS')
mat.diffuse_color = color_face[(iv-1)%len(color_face)]
#
slot = mat.texture_slots.add()
slot.texture = texchecker
slot.texture_coords = 'UV'
slot.blend_type = 'MULTIPLY'
slot.diffuse_color_factor = 0.1
pseudoEdS.data.materials.append(mat)
################################################################


################################################################
# ADJUST CAMERA
bpy.ops.object.select_all(action='DESELECT')
#obj_sphere.select = True
pseudoEdS.select = True

bpy.ops.view3d.camera_to_view_selected()
cam.data.angle += numpy.radians(5.)
bpy.ops.object.select_all(action='DESELECT')
################################################################



################################################################
# ADJUST ALL SURFACE MATERIALS
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        if obj.name in [obj_sphere] + obj_planes: continue
        if len(obj.data.materials) < 1: continue
        mat = obj.data.materials[0]
        mat.diffuse_intensity = 1
        mat.specular_intensity = 0.3
        mat.specular_hardness = 20
################################################################



"""
################################################################
# SET TEXTURE COORDS AND ADD CHECKER SLOT
n = m
uu = 0.5*(u + 1)
vv = uu

for obj in bpy.data.objects:
    #if obj.type == 'MESH':
    if obj.name[0:8] == 'bilinear' or obj.name[0:9] == 'spherical':
        scene.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.uv.unwrap(
            method='ANGLE_BASED',
            fill_holes=True,
            correct_aspect=True,
            use_subsurf_data=False,
            margin=0.001
        )
        bpy.ops.object.mode_set(mode='OBJECT')
        uvlayer = obj.data.uv_layers.active
        for j in range(n-1):
            for i in range(m-1):
                k = i + j*(m-1)
                f = obj.data.polygons[k]
                for l in [0,3]:
                    uvlayer.data[f.loop_start + l].uv[0] = uu[i]
                for l in [1,2]:
                    uvlayer.data[f.loop_start + l].uv[0] = uu[i+1]
                for l in [0,1]:
                    uvlayer.data[f.loop_start + l].uv[1] = vv[j]
                for l in [2,3]:
                    uvlayer.data[f.loop_start + l].uv[1] = vv[j+1]

        mat = obj.data.materials[0]
        mat.diffuse_intensity = 1
        mat.specular_intensity = 0.3
        mat.specular_hardness = 20
        
        slot = mat.texture_slots.add()
        slot.texture = texchecker
        slot.texture_coords = 'UV'
        slot.blend_type = 'MULTIPLY'
        slot.diffuse_color_factor = 0.09
################################################################



################################################################
# USE SEPARATE LAYERS
for obj in bpy.data.objects:
    if obj.name[0:8] == 'EdS_face':
        obj.layers[1] = True
        obj.layers[2] = True
        obj.layers[0] = False
    if obj.name[0:8] == 'bilinear':
        obj.layers[1] = True
        obj.layers[0] = False
    if obj.name[0:9] == 'spherical' or obj.name[0:8] == 'EdS_edge':
        obj.layers[2] = True
        obj.layers[0] = False


for ilayer in range(3):
    cam.layers[ilayer] = True
    lamp.layers[ilayer] = True
################################################################
"""
