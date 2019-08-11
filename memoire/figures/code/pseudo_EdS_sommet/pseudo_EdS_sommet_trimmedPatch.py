ROOT = '/home/bastien/'#'/d/bandrieu/'#

import bpy
from mathutils import Vector, Matrix

import numpy
import numpy.linalg
from numpy.polynomial.chebyshev import chebgrid2d, chebval2d

import sys
sys.path.append(ROOT + 'GitHub/Code/Python/')
import lib_blender_util as lbu
import lib_blender_edit as lbe
import lib_blender_figures as lbf
import lib_chebyshev as lcheb
import lib_color as lco
import lib_brep as lbrep
import lib_linalg
from lib_compgeom import circumcircle, complete_orthonormal_matrix, minimal_OBB

################################################################
class Vertex:
    def __init__(self, xyz, edge, faces):
        self.xyz = xyz
        self.edge = edge
        self.faces = faces
        return
################################################################



################################################################
def extrema_latitude(o_minus_v, t, r1, r1perp, a, b3):
    beta = numpy.arctan2(r1.dot(numpy.cross(t, b3)), r1.dot(b3)) + numpy.array([0, numpy.pi])
    beta = beta%(2*numpy.pi)
    u = [0, a]
    for bi in beta:
        if bi <= a:
            u.append(bi)
    #
    print('u = ', [ui/a for ui in u])
    x = numpy.tile(o_minus_v, (len(u), 1)).T + numpy.outer(r1, numpy.cos(u)) + numpy.outer(r1perp, numpy.sin(u))
    lat = numpy.arcsin((x[0]*b3[0] + x[1]*b3[1] + x[2]*b3[2])/numpy.sqrt(numpy.sum(x**2, axis=0)))
    print('lat/PI = ', lat/numpy.pi)
    return numpy.amin(lat), numpy.amax(lat)
################################################################

################################################################
def extrema_longitude(o_minus_v, t, r1, r1perp, a, B, nsample=10):
    u = a*numpy.linspace(0,1,nsample)
    x = numpy.tile(o_minus_v, (nsample, 1)).T + numpy.outer(r1, numpy.cos(u)) + numpy.outer(r1perp, numpy.sin(u))
    lon = numpy.arctan2(
        B[0,1]*x[0] + B[1,1]*x[1] + B[2,1]*x[2],
        B[0,0]*x[0] + B[1,0]*x[1] + B[2,0]*x[2]
    )
    imin = numpy.argmin(lon)
    imax = numpy.argmax(lon)

    uext = [0,0]
    for j, i in enumerate([imin, imax]):
        if i == 0:
            uext[j] = 0
        elif i == nsample-1:
            uext[j] = a
        else:
            uext[j] = u[i]
            ui = Newton_extremum_longitude(u[i]/a, o_minus_v, B, a, r1, r1perp)
            if ui is not None: uext[j] = ui*a
            
    print('u = ', [ui/a for ui in uext])
    x = numpy.tile(o_minus_v, (2, 1)).T + numpy.outer(r1, numpy.cos(uext)) + numpy.outer(r1perp, numpy.sin(uext))
    lon = numpy.arctan2(
        B[0,1]*x[0] + B[1,1]*x[1] + B[2,1]*x[2],
        B[0,0]*x[0] + B[1,0]*x[1] + B[2,0]*x[2]
    )
    print('lon/PI = ', lon/numpy.pi)
    return min(lon), max(lon)
################################################################



################################################################
def Newton_extremum_longitude(u, o_minus_v, B, a, r1, r1perp, TOL=1e-6, itmax=100):
    #
    for it in range(itmax):
        rvec = r1*numpy.cos(a*u) + r1perp*numpy.sin(a*u)
        x = o_minus_v + rvec
        x_u = a*(r1perp*numpy.cos(a*u) - r1*numpy.sin(a*u))
        x_uu = -a**2*rvec
        #
        cosu = x.dot(B[:,0])
        sinu = x.dot(B[:,1])
        lon = numpy.arctan2(sinu, cosu)
        e = B.dot([-sinu, cosu, 0])
        #
        cosu_u = x_u.dot(B[:,0])
        sinu_u = x_u.dot(B[:,1])
        e_u = B.dot([-sinu_u, cosu_u, 0])
        #
        h = e.dot(x_u)
        h_u = e_u.dot(x_u) + e.dot(x_uu)
        print('it. #%d, u = %s, lon/PI = %s, |h| = %s' % (it, u, lon/numpy.pi, abs(h)))
        #
        du = -h/h_u
        print('         du = %s' % du)
        if abs(du) < TOL and abs(h) < TOL:
            return u
        #
        u += du
        u = min(1, max(0, u))
    return None
################################################################



################################################################
pth_test = '/home/bastien/TravauxThese/SphericalGeometry/extrema_longitude_latitude/'
def LL_patch_from_arcs(planes, corners, center, nsample=10):
    m = len(planes)

    # normalize corners onto unit sphere
    if False:
        xyz_sample = corners
    else:
        xyz_sample = []
        for i, (tng, occ) in enumerate(planes):
            ci = corners[i]
            cj = corners[(i-1)%m]
            #
            r1 = cj - occ
            r2 = ci - occ
            r190d = numpy.cross(r1, tng)
            a = numpy.arctan2(r2.dot(r190d), r2.dot(r1))%(2*numpy.pi)
            mij = occ + r1*numpy.cos(0.5*a) + r190d*numpy.sin(0.5*a)
            xyz_sample.append(cj)
            xyz_sample.append(mij)
        
    s = numpy.array([xyz - center for xyz in xyz_sample])
    r = numpy.sqrt(numpy.sum(s**2, axis=1))
    ravg = numpy.sum(r)/float(len(s))
    s = s/numpy.tile(r, (3,1)).T
    print('s = ')
    print(s)

    # orthonormal basis
    B = complete_orthonormal_matrix(numpy.sum(s, axis=0), i=0).T
    print('Btmp =')
    print(B)

    # central projection onto plane tangent to unit sphere at point r1 = B[:,0]
    s_dot_r1 = s[:,0]*B[0,0] + s[:,1]*B[1,0] + s[:,2]*B[2,0]
    s_dot_r1 = numpy.sign(s_dot_r1)*numpy.maximum(1e-6, numpy.absolute(s_dot_r1))
    inv_s_dot_r1 = 1./s_dot_r1
    p = s*numpy.tile(inv_s_dot_r1, (3,1)).T
    print('p =')
    print(p)

    # coordinates in local frame (r2, r3)
    ab = lib_linalg.matmul(p, B[:,1:3])
    print('p_2d =')
    print(ab)

    # mimimum-area OBB
    ctr_ab, rng_ab, axes_ab = minimal_OBB(ab)
    print('ctr_ab =')
    print(ctr_ab)
    print('rng_ab =')
    print(rng_ab)
    print('area =')
    print(rng_ab[0]*rng_ab[1])
    print('axes_ab =')
    print(axes_ab)

    # final rotation matrix
    B[:,1:3] = lib_linalg.matmul(B[:,1:3], axes_ab)
    numpy.savetxt(pth_test + 'rotation_matrix.dat', B)

    # longitude-latitude extrema
    min_lon = numpy.pi
    max_lon = -numpy.pi
    min_lat = 0.5*numpy.pi
    max_lat = -0.5*numpy.pi
    for i, (tng, occ) in enumerate(planes):
        ci = corners[i]
        cj = corners[(i-1)%m]
        #
        fid = open(pth_test +'arc_'+str(i)+'.dat', 'w')
        for vec in [center, occ, tng, cj, ci]:
            fid.write('%s %s %s\n' % (vec[0], vec[1], vec[2]))
        fid.close()
        #
        r1 = cj - occ
        r2 = ci - occ
        r190d = numpy.cross(r1, tng)
        a = numpy.arctan2(numpy.dot(r2, r190d), numpy.dot(r2, r1))
        if a < 0: a += 2*numpy.pi
        # latitude
        min_lati, max_lati = extrema_latitude(occ - center, tng, r1, r190d, a, B[:,2])
        min_lat = min(min_lat, min_lati)
        max_lat = max(max_lat, max_lati)
        # longitude
        min_loni, max_loni = extrema_longitude(occ - center, tng, r1, r190d, a, B)
        min_lon = min(min_lon, min_loni)
        max_lon = max(max_lon, max_loni)

    print('%s < lon/PI < %s' % (min_lon/numpy.pi, max_lon/numpy.pi))
    print('%s < lat/PI < %s' % (min_lat/numpy.pi, max_lat/numpy.pi))
    ctr_tl = 0.5*numpy.array([max_lon + min_lon, max_lat + min_lat])
    rng_tl = 0.5*numpy.array([max_lon - min_lon, max_lat - min_lat])

    return (ravg, B, ctr_tl, rng_tl)
################################################################










################################################################
def LL_patch_from_corners(arcs, center, mrg=0, construction_steps=False, critere='area'):
    m = len(arcs)

    # normalize corners onto unit sphere
    s = numpy.array([arc[0] - center for arc in arcs])
    r = numpy.sqrt(numpy.sum(s**2, axis=1))
    ravg = numpy.sum(r)/float(m)
    s = s/numpy.tile(r, (3,1)).T

    # orthonormal basis
    R = complete_orthonormal_matrix(numpy.sum(s, axis=0), i=0).T

    # central projection onto plane tangent to unit sphere at point r1 = R[:,0]
    s_dot_r1 = s[:,0]*R[0,0] + s[:,1]*R[1,0] + s[:,2]*R[2,0]
    s_dot_r1 = numpy.sign(s_dot_r1)*numpy.maximum(1e-6, numpy.absolute(s_dot_r1))
    inv_s_dot_r1 = 1./s_dot_r1
    p = s*numpy.tile(inv_s_dot_r1, (3,1)).T

    # coordinates in local frame (r2, r3)
    ab = lib_linalg.matmul(p, R[:,1:3])
    if construction_steps:
        abverts = [(a, b, 0) for a, b in ab]
        obj = lbu.pydata_to_mesh(
            verts=abverts,
            faces=[],
            edges=[(i, (i+1)%m) for i in range(m)],
            name='ab_points'
        )
        obj.layers[3] = True
        obj.layers[0] = False

    # mimimum-area OBB
    ctr_ab, rng_ab, axes_ab = minimal_OBB(ab)
    if construction_steps:
        OBBverts = [((-1)**(i+1), (-1)**(j+1), 0) for j in range(2) for i in range(2)]
        OBBedges = [(0, 1), (1, 3), (3, 2), (2, 0)]
        obj = lbu.pydata_to_mesh(
            verts=OBBverts,
            faces=[],
            edges=OBBedges,
            name='ab_OBB'
        )
        obj.layers[3] = True
        obj.layers[0] = False
        obj.scale = (rng_ab[0], rng_ab[1], 1)
        obj.location = (ctr_ab[0], ctr_ab[1], 0)
        obj.rotation_euler[2] = numpy.arctan2(axes_ab[1,0], axes_ab[0,0])

    R[:,1:3] = lib_linalg.matmul(R[:,1:3], axes_ab)

    # xyz-coords in rotated frame
    s = numpy.empty((0,3), dtype=float)
    for arc in arcs:
        s = numpy.vstack([s, lib_linalg.matmul((arc - numpy.tile(center, (len(arc),1)))/ravg, R)])
    n = len(s)
    
    if construction_steps:
        obj = lbu.pydata_to_mesh(
            verts=s,
            faces=[],
            edges=[(i, (i+1)%n) for i in range(n)],
            name='s_points'
        )
        obj.layers[2] = True
        obj.layers[0] = False
        obj.location = center
        obj.scale = ravg*numpy.ones(3)
        obj.rotation_euler = Matrix(R).to_euler()

    
    # spherical coords: longitude t(heta), latitude l(ambda)
    tl = numpy.zeros((n,2))
    tl[:,0] = numpy.arctan2(s[:,1], s[:,0])
    tl[:,1] = numpy.arcsin(s[:,2])
    
    if construction_steps:
        tlverts = [(t/numpy.pi, l/numpy.pi, 0) for t, l in tl]
        obj = lbu.pydata_to_mesh(
            verts=tlverts,
            faces=[],
            edges=[(i, (i+1)%n) for i in range(n)],
            name='tl_points'
        )
        obj.layers[4] = True
        obj.layers[0] = False

    min_tl = numpy.amin(tl, axis=0)
    max_tl = numpy.amax(tl, axis=0)    

    ctr_tl = 0.5*(min_tl + max_tl)
    print('lambda_0/pi = ', ctr_tl[1]/numpy.pi)
    rng_tl = (1 + mrg)*0.5*(max_tl - min_tl)
    print('max |lambda|/pi = ', max(abs(ctr_tl[1] - rng_tl[1]), ctr_tl[1] + rng_tl[1])/numpy.pi)

    if construction_steps:
        obj = lbu.pydata_to_mesh(
            verts=OBBverts,
            faces=[],
            edges=OBBedges,
            name='tl_OBB'
        )
        obj.layers[4] = True
        obj.layers[0] = False
        obj.scale = (rng_tl[0]/numpy.pi, rng_tl[1]/numpy.pi, 1)
        obj.location = (ctr_tl[0]/numpy.pi, ctr_tl[1]/numpy.pi, 0)

    # uv-coords
    uv = (tl - numpy.tile(ctr_tl, (n,1)))/numpy.tile(rng_tl, (n,1))

    return (ravg, R, ctr_tl, rng_tl, uv)
    
################################################################














################################################################
pthin = ROOT + 'GitHub/FFTsurf/test/demo_EoS_brep/'
pthout = ROOT + '/GitHub/These/memoire/figures/data/pseudo_EdS_sommet/trimmed_patch/'
pthimg = ROOT + 'GitHub/These/memoire/figures/images/pseudo_EdS_sommet/'

ivert = 7
rho = 0.11
################################################################


#################################################
# CHECKER TEXTURE
imgchecker = bpy.data.images.load(filepath=ROOT + 'GitHub/These/memoire/figures/images/checker6.png')
texchecker = bpy.data.textures.new('texture_checker', 'IMAGE')
texchecker.image = imgchecker
#################################################


################################################################
# SET SCENE
scene = bpy.context.scene
lbu.clear_scene(meshes=True, lamps=True, cameras=False)

resx = 1024
resy = 0.85*resx

lbu.set_scene(
    resolution_x=resx,
    resolution_y=resy,
    resolution_percentage=100,
    alpha_mode='TRANSPARENT',
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
    shadow_method='RAY_SHADOW',#'NOSHADOW',
    shadow_ray_samples=16,
    shadow_soft_size=2.0,
    location=(2.66,0.99,3.34)#location=(-2.68, 1.65, 3.20)
)
lamp.data.shadow_color = 0.3*numpy.ones(3)

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
print('V.xyz = ', V.xyz)


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
color_face = numpy.loadtxt(ROOT + 'GitHub/These/memoire/figures/data/BRep/face_color.dat')
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

corner_avg = numpy.zeros(3)
for xyz in corners:
    corner_avg = corner_avg + xyz
corner_avg = corner_avg/len(corners)

# SPHERE RADIUS
rho = 0
for xyz in corners:
    rho += Vector(xyz - V.xyz).length
rho /= float(len(corners))

bpy.ops.mesh.primitive_uv_sphere_add(
    location=V.xyz,
    size=rho,
    segments=100,
    ring_count=100
)

sphere = bpy.data.objects['Sphere']
lbe.set_smooth(sphere)
sphere.draw_type = 'WIRE'
sphere.show_all_edges = True

mat = bpy.data.materials.new('mat_sphere')
mat.diffuse_color = (0.8,0.8,0.8)
mat.diffuse_intensity = 1
mat.specular_intensity = 0.346
mat.specular_hardness = 20
mat.emit = 0.4
mat.use_transparency = True
mat.raytrace_transparency.fresnel = 2.

sphere.data.materials.append(mat)

sphere.layers[1] = True
sphere.layers[0] = False
################################################################



################################################################
# OFFSET EDGES
iclr = [15,29,16]
end_planes = []
jm = int(m/2)

ih = 2*V.edge[0] + V.edge[1]
iedg = -1
while True:
    iedg += 1
    #
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
    mat.diffuse_color = color_face[iclr[iedg]-1]#color_face[8+ie]

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
    print('tng = ', tng)
    occ, rcc = circumcircle(eR, eL, e)
    #print(Vector(occ - V.xyz).length/rho)
    end_planes.append((tng, occ))
    #
    rot = Matrix(complete_orthonormal_matrix(tng, i=2)).transposed()
    bpy.ops.mesh.primitive_plane_add(
        location=occ,
        rotation=rot.to_euler(),
        radius=1.1*rcc
    )
    #
    """bpy.ops.mesh.primitive_circle_add(
        location=occ,
        rotation=rot.to_euler(),
        radius=rcc,
        vertices=100,
        fill_type='TRIFAN'
    )"""
    #
    # move on to next incident (outgoing) halfedge
    ih = halfedges[ih].prev
    ih = halfedges[ih].twin
    if ih == 2*V.edge[0] + V.edge[1]: break

"""
color_planes = lco.sample_colormap('IR', len(end_planes))
color_planes = lco.cc_hsv(color_planes, fs=1.4, fv=0.9)
"""
color_planes = numpy.array([
    (250, 117, 102),
    (152, 212, 91),
    (104, 171, 217)
])/255.
f = open(pthout + 'color_planes.dat', 'w')
for r, g, b in color_planes:
    f.write('%s, %s, %s\n' % (r, g, b))
f.close()

obj_planes = [bpy.data.objects['Plane'+suf] for suf in ['', '.001', '.002']]
for ipl, obj in enumerate(obj_planes):
    #obj.show_axis = True
    obj.show_transparent = True
    #obj.hide_render = True
    mat_plane = bpy.data.materials.new('mat_plane_'+str(ipl))
    mat_plane.diffuse_color = color_planes[ipl]
    mat_plane.diffuse_intensity = 1
    mat_plane.specular_intensity = 0
    mat_plane.use_transparency = True
    mat_plane.alpha = 0.5
    mat_plane.emit = 0.3
    obj.data.materials.append(mat_plane)
    obj.layers[1] = True
    obj.layers[0] = False

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
mat.diffuse_color = color_face[19]#color_face[(iv-1)%len(color_face)]
#
slot = mat.texture_slots.add()
slot.texture = texchecker
slot.texture_coords = 'UV'
slot.blend_type = 'MULTIPLY'
slot.diffuse_color_factor = 0.1
pseudoEdS.data.materials.append(mat)
################################################################


################################################################
# VERTEX EOS UNTRIMMED PATCH
c = lcheb.read_polynomial2(pthin + 'debug/eos_vert_c_' + format(ivert, '03') + '.cheb')
xyz = chebgrid2d(u, u, c)
mverts, mfaces = lbu.tensor_product_mesh_vf(xyz[0], xyz[1], xyz[2])
LLpatch = lbu.pydata_to_mesh(
    mverts,
    mfaces,
    name='LLpatch'
)
lbe.set_smooth(LLpatch)

# TEXTURE COORDS
scene.objects.active = LLpatch
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.uv.unwrap(
    method='ANGLE_BASED',
    fill_holes=True,
    correct_aspect=True,
    use_subsurf_data=False,
    margin=0.001
)
bpy.ops.object.mode_set(mode='OBJECT')
uvlayer = LLpatch.data.uv_layers.active
n = m
uu = 0.5*(u + 1)
vv = uu
for j in range(n-1):
    for i in range(m-1):
        k = i + j*(m-1)
        f = LLpatch.data.polygons[k]
        for l in [0,3]:
            uvlayer.data[f.loop_start + l].uv[0] = uu[i]
        for l in [1,2]:
            uvlayer.data[f.loop_start + l].uv[0] = uu[i+1]
        for l in [0,1]:
            uvlayer.data[f.loop_start + l].uv[1] = vv[j]
        for l in [2,3]:
            uvlayer.data[f.loop_start + l].uv[1] = vv[j+1]

# MATERIAL
mat = bpy.data.materials.new('mat_LLpatch')
mat.diffuse_color = bpy.data.materials['mat_pseudoEdS'].diffuse_color
#
slot = mat.texture_slots.add()
slot.texture = texchecker
slot.texture_coords = 'UV'
slot.blend_type = 'MULTIPLY'
slot.diffuse_color_factor = 0.1
LLpatch.data.materials.append(mat)

LLpatch.layers[2] = True
LLpatch.layers[0] = False
################################################################



################################################################
# ADJUST CAMERA
if True:
    cam.rotation_mode = 'QUATERNION'
    cam.location = (0.90851, 0.71343, 0.76544)
    cam.rotation_quaternion = (0.417, 0.241, 0.439, 0.758)
    cam.data.angle = numpy.radians(34.)
else:
    bpy.ops.object.select_all(action='DESELECT')
    #obj_sphere.select = True
    pseudoEdS.select = True
    
    bpy.ops.view3d.camera_to_view_selected()
    cam.data.angle += numpy.radians(10.)
    bpy.ops.object.select_all(action='DESELECT')
################################################################


################################################################
# EXPORT VERTEX IMAGE COORDS
bpy.ops.object.empty_add(location=V.xyz)
vx, vy = lbf.convert_3d_to_2d_coords(V.xyz, normalize=True)
#print('(vx, vy) = (%s, %s)' % (vx, vy))
f = open(pthout + 'xy_vertex.dat', 'w')
f.write('%s, %s' % (vx, vy))
f.close()
################################################################


################################################################
# EXPORT PLANES IMAGE COORDS
f = open(pthout + 'xy_planes.dat', 'w')
scltng = 0.135
for tng, occ in end_planes:
    x, y = lbf.convert_3d_to_2d_coords(occ, normalize=True)
    X, Y = lbf.convert_3d_to_2d_coords(occ + scltng*tng, normalize=True)
    f.write('%s, %s, %s, %s\n' % (x, y, X, Y))
f.close()
################################################################


################################################################
# EXPORT CORNERS IMAGE COORDS
f = open(pthout + 'xy_corners.dat', 'w')
corner_label = ['\\unv_{'+str(i)+'}' for i in range(len(corners))] + ['\\vrm{q}']
for i, xyz in enumerate(corners):# + [corner_avg]):
    x, y = lbf.convert_3d_to_2d_coords(xyz, normalize=True)
    f.write('%s, %s, %s\n' % (x, y, corner_label[i]))
f.close()
################################################################


################################################################
# EXPORT ARCS IMAGE COORDS
xyz_arcs = []
ncc = 200
tc = numpy.linspace(0, 1, ncc)
npl = len(end_planes)
for ipl, (tng, occ) in enumerate(end_planes):
    ci = corners[ipl]
    cj = corners[(ipl-1)%npl]
    r1 = cj - occ
    r2 = ci - occ
    r190d = numpy.cross(r1, tng)
    a = numpy.arctan2(numpy.dot(r2, r190d), numpy.dot(r2, r1))
    ts = numpy.linspace(0, a, ncc)
    td = numpy.linspace(a, 2*numpy.pi, ncc)
    if a < 0: a += 2*numpy.pi
    arc_solid = numpy.array(
        [occ + numpy.cos(ts[i])*r1 + numpy.sin(ts[i])*r190d for i in range(ncc)]
    )
    xyz_arcs.append(arc_solid)
    arc_dashed = numpy.array(
        [occ + numpy.cos(td[i])*r1 + numpy.sin(td[i])*r190d for i in range(ncc)]
    )
    xy_arc_solid = numpy.array(
        [lbf.convert_3d_to_2d_coords(xyz, normalize=True) for xyz in arc_solid]
    )
    xy_arc_dashed = numpy.array(
        [lbf.convert_3d_to_2d_coords(xyz, normalize=True) for xyz in arc_dashed]
    )
    numpy.savetxt(pthout + 'xy_arc_solid_' + str(ipl) + '.dat', xy_arc_solid)
    numpy.savetxt(pthout + 'xy_arc_dashed_' + str(ipl) + '.dat', xy_arc_dashed)
################################################################




################################################################
# LL-PATCH
ravg, R, ctr_tl, rng_tl, uv = LL_patch_from_corners(
    xyz_arcs,
    V.xyz,
    mrg=2e-2,
    construction_steps=True,
    critere='width'
)
print('ctr_tl/PI =', ctr_tl/numpy.pi, ', rng_tl/PI =', rng_tl/numpy.pi)
print('area = ', rng_tl[0]*numpy.cos(ctr_tl[1])*numpy.sin(rng_tl[1]))

us = ctr_tl[0] + u*rng_tl[0]
vs = ctr_tl[1] + u*rng_tl[1]

xs = numpy.outer(numpy.cos(us), numpy.cos(vs))
ys = numpy.outer(numpy.sin(us), numpy.cos(vs))
zs = numpy.outer(numpy.ones(len(us)), numpy.sin(vs))

mverts, mfaces = lbu.tensor_product_mesh_vf(xs, ys, zs)
LLpatch2 = lbu.pydata_to_mesh(
    verts=mverts,
    faces=mfaces,
    edges=[],
    name='LLpatch2'
)
lbe.set_smooth(LLpatch)

LLpatch2.location = V.xyz
LLpatch2.rotation_euler = Matrix(R).to_euler()
LLpatch2.scale = ravg*numpy.ones(3)

LLpatch2.draw_type = 'WIRE'
LLpatch2.show_all_edges = True
LLpatch2.hide_render = True
LLpatch2.layers[2] = True
LLpatch2.layers[0] = False



#min_max_dz = LL_patch_from_arcs(end_planes, corners, V.xyz)
ravg, R, ctr_tl, rng_tl = LL_patch_from_arcs(end_planes, corners, V.xyz, nsample=10)
print('ctr_tl/PI =', ctr_tl/numpy.pi, ', rng_tl/PI =', rng_tl/numpy.pi)
print('area = ', rng_tl[0]*numpy.cos(ctr_tl[1])*numpy.sin(rng_tl[1]))

OBBtl = None
for obj in bpy.data.objects:
    if obj.name == 'tl_OBB':
        OBBtl = obj
        break
if OBBtl is not None:
    OBBtl.scale = (rng_tl[0]/numpy.pi, rng_tl[1]/numpy.pi, 1)
    OBBtl.location = (ctr_tl[0]/numpy.pi, ctr_tl[1]/numpy.pi, 0)

us = ctr_tl[0] + u*rng_tl[0]
vs = ctr_tl[1] + u*rng_tl[1]

xs = numpy.outer(numpy.cos(us), numpy.cos(vs))
ys = numpy.outer(numpy.sin(us), numpy.cos(vs))
zs = numpy.outer(numpy.ones(len(us)), numpy.sin(vs))

mverts, mfaces = lbu.tensor_product_mesh_vf(xs, ys, zs)
LLpatch3 = lbu.pydata_to_mesh(
    verts=mverts,
    faces=mfaces,
    edges=[],
    name='LLpatch3'
)
lbe.set_smooth(LLpatch)

LLpatch3.location = V.xyz
LLpatch3.rotation_euler = Matrix(R).to_euler()
LLpatch3.scale = ravg*numpy.ones(3)

LLpatch3.draw_type = 'WIRE'
LLpatch3.show_all_edges = True
LLpatch3.hide_render = True
LLpatch3.layers[2] = True
LLpatch3.layers[0] = False




c = lcheb.read_polynomial2(ROOT + 'GitHub/FFTsurf/test/LL_patch/surf.cheb')
xyz = chebgrid2d(u, u, c)
mverts, mfaces = lbu.tensor_product_mesh_vf(xyz[0], xyz[1], xyz[2])
LLpatch4 = lbu.pydata_to_mesh(
    mverts,
    mfaces,
    name='LLpatch4'
)
lbe.set_smooth(LLpatch4)
LLpatch4.layers[2] = True
LLpatch4.layers[0] = False
################################################################


################################################################
# EXPORT DATA FOR FFTSURF TEST
f = open(ROOT + 'GitHub/FFTsurf/test/LL_patch/input.dat', 'w')
f.write('%d\n' % len(end_planes))
cc_center = [pla[1] for pla in end_planes]
normal = [pla[0] for pla in end_planes]
for x, y, z in [V.xyz] + cc_center + normal + corners:
    f.write('%s %s %s\n' % (x, y, z))
f.close()
################################################################


################################################################
# ADJUST ALL SURFACE MATERIALS
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        if obj in obj_planes or obj == sphere: continue
        if len(obj.data.materials) < 1: continue
        mat = obj.data.materials[0]
        mat.diffuse_intensity = 1
        mat.specular_intensity = 0#0.3
        mat.specular_hardness = 20
################################################################




################################################################
# USE SEPARATE LAYERS
"""
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
"""
for ilayer in range(3):
    cam.layers[ilayer] = True
    lamp.layers[ilayer] = True
################################################################

