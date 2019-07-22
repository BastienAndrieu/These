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
import lib_brep as lbrep

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
normals = []

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
    cu, cv = lcheb.diff2(c)
    xyz = chebgrid2d(u, u, c)
    xyz_u = chebgrid2d(u, u, cu)
    xyz_v = chebgrid2d(u, u, cv)

    invsqrtdet = 1/numpy.sqrt(
        numpy.sum(xyz_u**2, axis=0)*numpy.sum(xyz_v**2, axis=0) - numpy.sum(xyz_u*xyz_v, axis=0)**2
        )
    
    nor = numpy.zeros((3,m,m))
    for i in range(3):
        j = (i+1)%3
        k = (j+1)%3
        nor[i] = (xyz_u[j]*xyz_v[k] - xyz_u[k]*xyz_v[j])*invsqrtdet

    xyz = xyz + rho*nor
    mverts, mfaces = lbu.tensor_product_mesh_vf(xyz[0], xyz[1], xyz[2])
    obj = lbu.pydata_to_mesh(
        mverts,
        mfaces,
        name='EdS_face_'+strf
    )
    lbe.set_smooth(obj)
    obj.data.materials.append(mat)

    # NORMAL AT VERTEX
    xyz = chebval2d(Vuv[iloc][0], Vuv[iloc][1], c)
    xyz_u = chebgrid2d(Vuv[iloc][0], Vuv[iloc][1], cu)
    xyz_v = chebgrid2d(Vuv[iloc][0], Vuv[iloc][1], cv)
    nor = Vector(xyz_u).cross(Vector(xyz_v))
    nor.normalize()

    normals.append(nor)
################################################################


################################################################
# OFFSET EDGES
ih = 2*V.edge[0] + V.edge[1]
while True:
    #
    jh = halfedges[ih].twin
    if jh < ih:
        ic = int(jh/2)
        faces = [halfedges[jh].face, halfedges[ih].face]
    else:
        ic = int(ih/2)
        faces = [halfedges[ih].face, halfedges[jh].face]
    #
    uv = curves[ic].uv
    xyz = curves[ic].xyz
    n = len(uv)
    xyzLR = []
    norLR = []
    #
    for iloc, iface in enumerate(faces):
        c = lcheb.read_polynomial2(pthin + 'brepmesh/c_' + format(iface+1,'03') + '.cheb')
        cu, cv = lcheb.diff2(c)
        jloc = (iloc+1)%2
        ui = uv[:,2*jloc]
        vi = uv[:,2*jloc+1]
        xyzLR.append(chebval2d(ui, vi, c))
        xyz_u = chebval2d(ui, vi, cu)
        xyz_v = chebval2d(ui, vi, cv)
        norLR.append([(Vector(xyz_u[:,i]).cross(Vector(xyz_v[:,i]))).normalized() for i in range(n)])
    #
    eos = numpy.zeros((3,n,m))
    v = numpy.linspace(0,1,m)
    for i in range(n):
        norR = norLR[1][i]
        norL = norLR[0][i]
        tng = (norL.cross(norR)).normalized()
        norR90d = norR.cross(tng)
        angle = numpy.arctan2(norL.dot(norR90d), norL.dot(norR))
        if angle < 0: angle += 2*numpy.pi
        for j in range(m):
            aj = v[j]*angle
            eos[:,i,j] = xyz[i] + rho*(norR*numpy.cos(aj) + norR90d*numpy.sin(aj))
            #eos[:,i,j] = xyz[i] + rho*(norR*(1-v[j]) + norL*v[j])
    #
    mverts, mfaces = lbu.tensor_product_mesh_vf(eos[0], eos[1], eos[2])
    obj = lbu.pydata_to_mesh(
        mverts,
        mfaces,
        name='EdS_edge_'+str(ic)
    )
    lbe.set_smooth(obj)

    mat = bpy.data.materials.new('mat_edge_'+str(ic))
    mat.diffuse_color = color_face[8+ic]
    obj.data.materials.append(mat)
    #
    ih = halfedges[ih].prev
    ih = halfedges[ih].twin
    if ih == 2*V.edge[0] + V.edge[1]: break
################################################################


################################################################
# QUADPOLY
c = Vector((0,0,0))
for nor in normals:
    c = c + nor
c.normalize()

q = []
for i in range(nf):
    q.append((normals[i] + normals[(i+1)%nf]).normalized())

color_quad = numpy.tile(color_face[ivert-1], (3,1))
hsv = lco.rgb2hsv(color_quad)
hsv[:,0] = (hsv[:,0] + 0.08*(2*numpy.random.random(3)-1))%1
hsv[:,1] = numpy.maximum(0, numpy.minimum(1, hsv[:,1]*(1 + 0.15*(2*numpy.random.random(3)-1))))
hsv[:,2] = numpy.maximum(0, numpy.minimum(1, hsv[:,2]*(1 + 0.15*(2*numpy.random.random(3)-1))))
color_quad = lco.hsv2rgb(hsv)

for i in range(nf):
    mat = bpy.data.materials.new('mat_quad_'+str(i+1))
    mat.diffuse_color = color_quad[i]
    
    xyz = bilinear_patch(c, q[(i-1)%nf], normals[i], q[i], u, u)
    mverts, mfaces = lbu.tensor_product_mesh_vf(xyz[0], xyz[1], xyz[2])
    obj = lbu.pydata_to_mesh(
        mverts,
        mfaces,
        name='bilinear_'+str(i+1)
    )
    obj.location = V.xyz
    obj.scale = rho*numpy.ones(3)
    lbe.set_smooth(obj)
    obj.data.materials.append(mat)
    
    invr = 1/numpy.sqrt(numpy.sum(xyz**2, axis=0))
    for j in range(3):
        xyz[j] = xyz[j]*invr
    mverts, mfaces = lbu.tensor_product_mesh_vf(xyz[0], xyz[1], xyz[2])
    obj = lbu.pydata_to_mesh(
        mverts,
        mfaces,
        name='spherical_'+str(i+1)
    )
    obj.location = V.xyz
    obj.scale = rho*numpy.ones(3)
    lbe.set_smooth(obj)
    obj.data.materials.append(mat)
################################################################



################################################################
# ADJUST CAMERA
bpy.ops.object.select_all(action='DESELECT')
for obj in bpy.data.objects:
    if obj.name[0:9] == 'spherical':
        obj.select = True

bpy.ops.view3d.camera_to_view_selected()
cam.data.angle += numpy.radians(5.) 
################################################################



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
