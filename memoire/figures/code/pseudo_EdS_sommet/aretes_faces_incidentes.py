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
pthin = '/d/bandrieu/GitHub/FFTsurf/test/demo_EoS_brep/'
pthout = '/d/bandrieu/GitHub/These/memoire/figures/data/pseudo_EdS_sommet/'
pthimg = '/d/bandrieu/GitHub/These/memoire/figures/images/pseudo_EdS_sommet/'

ivert = 7
################################################################


################################################################
# SET SCENE
scene = bpy.context.scene
lbu.clear_scene(meshes=True, lamps=True, cameras=False)

resx = 1000
resy = resx#0.75*resx

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
    energy=1.2,
    shadow_method='RAY_SHADOW',
    shadow_ray_samples=16,
    shadow_soft_size=2.0,
    location=(2.66,0.99,2.49)
)

## Set Camera
cam = scene.camera
cam.location = (2.102, 1.798, 1.104)
cam.rotation_euler = numpy.radians((66.7, 0.778, 132.2))
#cam.rotation_euler = numpy.radians((66.7, 0.778, 127.3))
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
# LOAD FACES
color_face = numpy.loadtxt('/d/bandrieu/GitHub/These/memoire/figures/data/BRep/face_color.dat')
color_face = lco.cc_hsv(color_face, fs=1.2, fv=1.0)

bpy.ops.import_scene.obj(
    filepath=pthin+'mesh/mesh.obj',
    axis_forward='Y',
    axis_up='Z'
)
obj = bpy.data.objects['mesh']
lbe.set_smooth(obj)
bpy.ops.object.select_all(action='DESELECT')
obj.select = True
bpy.ops.view3d.camera_to_view_selected()
cam.data.angle += numpy.radians(1.) 


for mat in obj.data.materials:
    mat.diffuse_color = color_face[int(mat.name[3:])-1]
    mat.diffuse_intensity = 1
    mat.specular_intensity = 0
    mat.specular_hardness = 30

#xyz_facelabel = []
normals = []
xyz_facecenter = []
for iloc, iface in enumerate(V.faces):
    strf = format(iface,'03')

    mat = bpy.data.materials.new('mat_face_'+strf)
    mat.diffuse_color = color_face[iface-1]
    
    c = lcheb.read_polynomial2(pthin + 'brepmesh/c_' + strf + '.cheb')
    cu, cv = lcheb.diff2(c)

    xyz_u = chebgrid2d(Vuv[iloc][0], Vuv[iloc][1], cu)
    xyz_v = chebgrid2d(Vuv[iloc][0], Vuv[iloc][1], cv)
    normals.append(Vector(xyz_u).cross(Vector(xyz_v)).normalized())
    xyz_facecenter.append(chebval2d(0, 0, c))


normal_avg = Vector((0,0,0))
for nor in normals:
    normal_avg += nor
normal_avg.normalize()
################################################################


################################################################
# ADJUST CAMERA
# align with average normal at vertex
cam.rotation_mode = 'QUATERNION'
cam.rotation_quaternion = Vector(normal_avg).to_track_quat('Z','Y')

# fit view
bpy.ops.object.select_all(action='DESELECT')
obj.select = True
bpy.ops.view3d.camera_to_view_selected()
cam.data.angle += numpy.radians(1.)
################################################################


################################################################
# EXPORT VERTEX IMAGE COORDS
vx, vy = lbf.convert_3d_to_2d_coords(V.xyz, normalize=True)
#print('(vx, vy) = (%s, %s)' % (vx, vy))

f = open(pthout + 'xy_vertex.dat', 'w')
f.write('%s, %s' % (vx, vy))
f.close()
################################################################


################################################################
# EXPORT FACE LABEL IMAGE COORDS
Rfacelabel = 0.13
f = open(pthout + 'xy_facelabel.dat', 'w')
for c in xyz_facecenter:
    cx, cy = lbf.convert_3d_to_2d_coords(c, normalize=True)
    #print('(cx, cy) = (%s, %s)' % (cx, cy))
    dx = cx - vx
    dy = cy - vy
    #print('(dx, dy) = (%s, %s)' % (dx, dy))
    l = Rfacelabel/numpy.hypot(dx, dy)
    f.write('%s, %s\n' % (vx + l*dx, vy + l*dy))
f.close()
################################################################


################################################################
# EXPORT EDGE LABEL IMAGE COORDS
Redgelabel = Rfacelabel
sep_edgelabel = 0.025
numedge = 0
ih = 2*V.edge[0] + V.edge[1]
f = open(pthout + 'xy_edgelabel.dat', 'w')
while True:
    numedge += 1
    jh = halfedges[ih].twin
    if jh < ih:
        ic = int(jh/2)
        xyz = curves[ic].xyz[-1::-1]
    else:
        ic = int(ih/2)
        xyz = curves[ic].xyz
    #
    xy = numpy.zeros((len(xyz),2))
    for i, p in enumerate(xyz):
        xy[i] = lbf.convert_3d_to_2d_coords(p, normalize=True)
    #
    numpy.savetxt(pthout + 'xy_edge_' + str(numedge) + '.dat', xy)
    #
    """s = numpy.zeros(len(xy))
    for i in range(1,len(xy)):
        s[i] = s[i-1] + numpy.hypot(
            xy[i][0] - xy[i-1][0],
            xy[i][1] - xy[i-1][1]
        )
    sm = 0.4*s[-1]"""
    di = 0
    for i in range(len(xy)-1):
        dip1 = numpy.hypot(xy[i+1][0] - vx, xy[i+1][1] - vy)
        #if s[i] <= sm and s[i+1] >= sm:
        if di <= Redgelabel and dip1 >= Redgelabel:
            #tm = (sm - s[i])/(s[i+1] - s[i])
            tm = (Redgelabel - di)/(dip1 - di)
            xym = (1 - tm)*xy[i] + tm*xy[i+1]
            tng = xy[i+1] - xym
            tng = tng/numpy.hypot(tng[0], tng[1])
            f.write('%s, %s\n' % (
                xym[0] - sep_edgelabel*tng[1],
                xym[1] + sep_edgelabel*tng[0]
            ))
        di = dip1
    #
    ih = halfedges[ih].prev
    ih = halfedges[ih].twin
    if ih == 2*V.edge[0] + V.edge[1]: break

f.close()
################################################################
