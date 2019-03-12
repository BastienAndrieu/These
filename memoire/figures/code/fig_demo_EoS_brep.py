import bpy
import bpy_extras
from mathutils import Vector

import numpy
import sys
sys.path.append('/stck/bandrieu/Bureau/Python/mylibs/')
sys.path.append('/d/bandrieu/GitHub/These/memoire/figures/code/')
import lib_color as mycolors
import my_lib as myl
import lib_fig

pth = '/d/bandrieu/GitHub/FFTsurf/test/demo_EoS_brep/'


#########################################
EoS = False#True#
if EoS:
    suff = '_new'
else:
    suff = ''
#########################################


##
use_freestyle = True
edge_flw = 1.8
edge_tck = 2.e-3#1.25e-3
vert_rad = 3.*edge_tck
vert_ns = 32
vert_nr = vert_ns


##
scene = bpy.context.scene
myl.blankScene()

## Set scene   
scene.render.resolution_x = 1200
scene.render.resolution_y = scene.render.resolution_x
scene.render.resolution_percentage = 100

bpy.data.worlds["World"].horizon_color = [1,1,1]
bpy.data.worlds["World"].light_settings.samples = 10

cam = scene.camera
if False:
    print('a')
    cam.location = numpy.array([1.93412, 1.37517, 1.00552])
    cam.rotation_euler = numpy.array((65.565, 0.629, 128.129))*numpy.pi/180.0
    bpy.data.cameras["Camera"].angle = 44.*numpy.pi/180.0
elif True:
    print('b')
    cam.location = numpy.array([2.102,1.798,1.104])
    cam.rotation_euler = numpy.array((66.7,0.778,132.2))*numpy.pi/180.0
    bpy.data.cameras["Camera"].angle = 37.72*numpy.pi/180.0
else:
    print('c')
    cam.location = numpy.array([1.77634,1.49005,0.90784])
    cam.rotation_euler = numpy.array((66.7,0.778,132.2))*numpy.pi/180.0
    bpy.data.cameras["Camera"].angle = 50.*numpy.pi/180.0

## Lighting
# Environment lighting
scene.world.light_settings.use_environment_light = True
scene.world.light_settings.environment_energy = 0.3
scene.world.light_settings.environment_color = 'PLAIN'

# Create new lamp datablock
lamp_data = bpy.data.lamps.new(name="KeyLight", type='POINT')
lamp_data.energy = 1.2
lamp_data.shadow_method = 'RAY_SHADOW'
lamp_data.shadow_ray_samples = 10
lamp_data.shadow_soft_size = 2.0

# Create new object with our lamp datablock
lamp_object = bpy.data.objects.new(name="KeyLight", object_data=lamp_data)

# Link lamp object to the scene so it'll appear in this scene
scene.objects.link(lamp_object)

#lamp_object.location = [4.07625, 1.00545, 5.90386]
#lamp_object.location = [3.65,1.874,2.216]
lamp_object.location = [3.75,1.65,3.20]

# And finally select it make active
lamp_object.select = True
scene.objects.active = lamp_object
######################


name = 'mesh'
if EoS:
    name += '_eos'
bpy.ops.import_scene.obj(filepath=pth+name+'/'+name+'.obj',
                         axis_forward='Y', axis_up='Z')
obj = bpy.data.objects[name]
myl.setSmooth(obj)


nf = len(bpy.data.materials)-1
#clf = mycolors.sample_colormap('I2', nf)
clf = numpy.loadtxt('/d/bandrieu/GitHub/These/memoire/figures/code/demo_EoS_brep_palette_modif.dat')
clf = mycolors.cc_hsv(clf, fs=1.2, fv=1.0)
clh = mycolors.cc_hsv(clf, fs=0.85, fv=0.62)#fs=0.75, fv=0.45)

## set face materials
i = 0
for i in range(nf):
    mat = bpy.data.materials["mat"+str(i+1)]
    mat.diffuse_color = clf[i%len(clf)]
    mat.diffuse_intensity = 1.0
    mat.specular_intensity = 0.1
    mat.specular_hardness = 8.0

## trace edges
f = open(pth+'debug/edges_xyz'+suff+'.dat','r')
ne = int(f.readline())
for ie in range(ne):
    np = int(f.readline())
    p = numpy.zeros((np,3))
    for ip in range(np):
        x,y,z = [float(a) for a in f.readline().split()]
        p[ip,0] = x
        p[ip,1] = y
        p[ip,2] = z
    if use_freestyle:
        myl.addFreestylePolyline(p,
                                 name="edge_"+str(ie),
                                 location=[0,0,0],
                                 color=[0,0,0],
                                 thickness=2.0,
                                 newlineset=False)
        #                        dashed=False,
        #                        dash=0,
        #                        gap=0)
    else:
        myl.addPolyline(p, [0,0,0], edge_tck, 0)
f.close()

if use_freestyle:
    scene.render.use_freestyle = True
    freestyle = scene.render.layers.active.freestyle_settings

    lineset = freestyle.linesets["LineSet"]
    lineset.select_silhouette = False
    lineset.select_border = False
    lineset.select_crease = False
    lineset.select_edge_mark = True
    lineset.visibility = 'VISIBLE'
    linestyle = bpy.data.linestyles["LineStyle"]
    linestyle.caps = "ROUND"
    linestyle.thickness = edge_flw

    bpy.ops.scene.freestyle_lineset_add()
    lineset = freestyle.linesets["LineSet 2"]
    lineset.select_silhouette = False
    lineset.select_border = False
    lineset.select_crease = False
    lineset.select_edge_mark = True
    lineset.visibility = 'HIDDEN'
    linestyle = bpy.data.linestyles["LineStyle.001"]
    linestyle.caps = "ROUND"
    #linestyle.alpha = 0.4
    linestyle.thickness = edge_flw
    linestyle.use_dashed_line = True
    linestyle.dash1 = 7
    linestyle.gap1 = linestyle.dash1

    for linestyle in bpy.data.linestyles:
        linestyle.geometry_modifiers["Sampling"].sampling = 0.05
        
    if not EoS:
        for lineset in freestyle.linesets:
            lineset.select_crease = True
        for i in range(16):
            bpy.data.objects['edge_'+str(i)].hide_render = True

## vertices
mat_v = myl.lineMaterial('mat_verts',[0,0,0])
#verts = numpy.loadtxt(pth+'debug/verts_new.dat')
#for iv, v in enumerate(verts):
cxyz = cam.location

f = open(pth+'debug/verts'+suff+'.dat','r')
nv = -1
vertdat = []
for l in f:
    v = [float(a) for a in l.split()]
    if len(v) > 7:
        nv += 1
        bpy.ops.mesh.primitive_uv_sphere_add(location=v[0:3],
                                             segments=vert_ns,
                                             ring_count=vert_nr,
                                             size=vert_rad)
        if nv == 0:
            name = 'Sphere'
        else:
            name = 'Sphere.'+format(nv,'03')
        obj = bpy.data.objects[name]
        myl.setMaterial(obj, mat_v)
        co_2d = bpy_extras.object_utils.world_to_camera_view(scene,
                                                             cam,
                                                             obj.location)
        result, object, matrix, location, normal = scene.ray_cast(start=cxyz,
                                                                  end=v[0:3])
        visible = (object == obj)
        obj.hide_render = True
        if not visible:
            obj.hide = True
        vertdat.append([co_2d.x, co_2d.y, visible])
f.close()

f = open('/d/bandrieu/GitHub/These/memoire/figures/data/demo_EoS_brep_verts'+suff+'.dat','w')
for v in vertdat:
    f.write(str(v[0]) + ' ' + str(v[1]) + ' ')
    if v[2]:
        f.write('1\n')
    else:
        f.write('0\n')
f.close()

############################
sx = 0.17
sy = 0.05
hx = 0.07#sx
hy = 0.6*hx#sy

def halfedge_coords(xyz_orig, xyz_dest, normal_to_face):
    u = numpy.asarray(xyz_dest) - numpy.asarray(xyz_orig)
    lu = numpy.sqrt(numpy.sum(numpy.power(u,2)))
    u = u/lu
    w = numpy.asarray(normal_to_face)
    w = w/numpy.sqrt(numpy.sum(numpy.power(w,2)))
    v = numpy.cross(w, u)
    a = numpy.asarray(xyz_orig) + sx*u + sy*v
    b = numpy.asarray(xyz_dest) - sx*u + sy*v
    c = b - hx*u + hy*v
    d = b - hx*u 
    p = numpy.vstack((a,b,c,d))
    u = numpy.zeros((4,2))
    for i in range(4):
        co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, Vector(p[i]))
        u[i,0] = co_2d.x
        u[i,1] = co_2d.y
    """
    txt = '\draw[thick] ('
    for i in range(3):
        txt += str(u[i,0]) + ', ' + str(u[i,1]) + ')'
        if i < 2:
            txt += ' -- ('
        else:
            txt += ';'
    """
    txt = '\draw[thick] ('+str(u[0,0])+', '+str(u[0,1])+ ') -- '
    txt += '('+str(u[3,0])+', '+str(u[3,1])+ ');\n'
    txt += '\draw[thick, fill=black] ('+str(u[3,0])+', '+str(u[3,1])+ ') -- '
    txt += '('+str(u[1,0])+', '+str(u[1,1])+ ') -- '
    txt += '('+str(u[2,0])+', '+str(u[2,1])+ ') -- cycle;\n%'
    #print(txt)
    return u

"""
x = numpy.array([[0.5-sy, 0.5, 0.5-sx],
                 [0.5-sy, 0.5, -0.5+sx],
                 [0.5-sx, 0.5, -0.5+sy],
                 [-0.5+sx, 0.5, -0.5+sy],
                 [-0.5+sy, 0.5, -0.5+sx],
                 [-0.5+sy, 0.5, 0.5-sx],
                 [-0.5+sx, 0.5, 0.5-sy],
                 [0.5-sx, 0.5, 0.5-sy],###
                 [0.5-sy, -0.5+sx, 0.5],
                 [0.5-sy, 0.5-sx, 0.5],
                 [0.5-sx, 0.5-sy, 0.5],
                 [-0.5+sx, 0.5-sy, 0.5],
                 [-0.5+sy, 0.5-sx, 0.5],
                 [-0.5+sy, 0.5-sx, 0.5],
                 [-0.5+sy, -.05+sx, 0.5],
                 [-0.5+sx, -0.5+sy, 0.5],
                 [0.5-sx, -0.5+sy, 0.5]])
u = numpy.zeros((len(x),2))
for i in range(len(x)):
    obj = myl.addEmpty(name="endpoint"+str(i), location=x[i])
    xi, yi = lib_fig.get_2d_coordinates(obj)
    u[i,0] = xi
    u[i,1] = yi

for i in range(8):
    j = 2*i+1
    print( '(', str(u[2*i,0]), ',', str(u[2*i,1]), ') -- (', str(u[j,0]), ',', str(u[j,1]), ');')
"""

w = 0.5
R = 0.25
H = 0.4
"""
hlist = [
    [[ w,-w,w], [ w, w,w], [0,0,1]],
    [[ w, w,w], [-w, w,w], [0,0,1]],
    [[-w, w,w], [-w,-w,w], [0,0,1]],
    [[-w,-w,w], [ w,-w,w], [0,0,1]],
    [[ w,w,-w], [-w,w,-w], [0,1,0]],
    [[-w,w,-w], [-w,w, w], [0,1,0]],
    [[-w,w, w], [ w,w, w], [0,1,0]],
    [[ w,w, w], [ w,w,-w], [0,1,0]]]
f = open('/d/bandrieu/GitHub/These/memoire/figures/data/demo_EoS_brep_hedg'+suff+'.dat','w')
for i, h in enumerate(hlist):
    u = halfedge_coords(h[0], h[1], h[2])
    if i < 4:
        j = 5
    else:
        j = 3
    cj = clh[j]
    for k in range(2):
        f.write(str(u[k,0]) + ' ' + str(u[k,1]) + ' ')
    for cjk in cj:
        f.write(str(cjk) + ' ')
    f.write('\n')
f.close()
"""

hlist = [
    [],#1
    [
        [[w, w,-w], [w, w, w], [1,0,0]],
        [[w, w, w], [w,-w, w], [1,0,0]]],#2
    [],#3
    [
        [[ w,w,-w], [-w,w,-w], [0,1,0]],
        [[-w,w,-w], [-w,w, w], [0,1,0]],
        [[-w,w, w], [ w,w, w], [0,1,0]],
        [[ w,w, w], [ w,w,-w], [0,1,0]]],#4
    [],#5
    [
        [[ w,-w,w], [ w, w,w], [0,0,1]],
        [[ w, w,w], [-w, w,w], [0,0,1]],
        [[-w, w,w], [-w,-w,w], [0,0,1]],
        [[-w,-w,w], [ w,-w,w], [0,0,1]]],#6
    [],#7
    [],#8 [[w,0,R], [w+H,0,R], [0,0.2,1]]]
    []]#9

f = open('/d/bandrieu/GitHub/These/memoire/figures/data/demo_EoS_brep_hedg'+suff+'.dat','w')
for i, fh in enumerate(hlist):
    if len(fh) < 1: continue
    c = clh[i]
    if i == 3:
        c = 0.8*c
    for h in fh:
        u = halfedge_coords(h[0], h[1], h[2])
        for k in range(2):
            f.write(str(u[k,0]) + ' ' + str(u[k,1]) + ' ')
        for cj in c:
            f.write(str(cj) + ' ')
        f.write('\n')
f.close()








"""
halfedge_coords([ w,-w,w], [ w, w,w], [0,0,1])
halfedge_coords([ w, w,w], [-w, w,w], [0,0,1])
halfedge_coords([-w, w,w], [-w,-w,w], [0,0,1])
halfedge_coords([-w,-w,w], [ w,-w,w], [0,0,1])
print('')
halfedge_coords([ w,w,-w], [-w,w,-w], [0,1,0])
halfedge_coords([-w,w,-w], [-w,w, w], [0,1,0])
halfedge_coords([-w,w, w], [ w,w, w], [0,1,0])
halfedge_coords([ w,w, w], [ w,w,-w], [0,1,0])
"""

############################
sphere_ns = 32
sphere_nr = 32
mat = bpy.data.materials.new("mat_sph")
mat.diffuse_color = [0.800, 0.494, 0.317]
mat.diffuse_intensity = 1
mat.specular_intensity = 0.3
mat.specular_hardness = 30.
mat.use_transparency = True
mat.raytrace_transparency.fresnel = 3.0#2.2
mat.alpha = 0.6
mat.emit = 0.8
mat.use_raytrace = False
mat.use_shadows = False
mat.use_cast_buffer_shadows = False


if False:#not EoS:
    """
    spheres = numpy.loadtxt(pth+'spheres.dat')
    for i, s in enumerate(spheres):
        bpy.ops.mesh.primitive_uv_sphere_add(location=s[0:3],
                                             segments=sphere_ns,
                                             ring_count=sphere_nr,
                                             size=s[3])
    """
    bpy.ops.import_mesh.stl(filepath='/d/bandrieu/GitHub/These/memoire/figures/data/sphere_packing_mesh.stl')
    name = 'Sphere Packing Mesh'
    obj = bpy.data.objects[name]
    obj.hide = True
    obj.hide_render = True
    msh = bpy.data.meshes[name]
    
    for i, v in enumerate(msh.vertices):
        r = 0.15*(1.0 + 0.15*numpy.cos(5.0*(v.co.x+v.co.y+v.co.z)))
        bpy.ops.mesh.primitive_uv_sphere_add(location=v.co,
                                             segments=sphere_ns,
                                             ring_count=sphere_nr,
                                             size=r)
        obj = bpy.data.objects["Sphere."+format(nv+1+i,'03')]
        obj.layers[1] = True
        obj.layers[0] = False
        myl.setSmooth(obj)
        myl.setMaterial(obj, mat)
    
    mat = bpy.data.materials.new("mat_eos")
    mat.diffuse_color = [0.527, 0.800, 0.213]
    mat.diffuse_intensity = 1
    mat.specular_intensity = 0#0.3
    mat.specular_hardness = 30.
    mat.use_transparency = True
    mat.raytrace_transparency.fresnel = 2.0#2.2
    mat.raytrace_transparency.fresnel_factor = 1.7
    mat.alpha = 0.72#0.6
    mat.emit = 0.6#0.8
    mat.use_raytrace = False
    mat.use_shadows = False
    mat.use_cast_buffer_shadows = False
    bpy.ops.import_scene.obj(filepath=pth+'mesh_eos/mesh_eos.obj',
                             axis_forward='Y', axis_up='Z')
    obj = bpy.data.objects["mesh_eos"]
    obj.layers[2] = True
    obj.layers[0] = False
    myl.setSmooth(obj)
    scene.objects.active = obj
    for i in range(31):
        bpy.ops.object.material_slot_remove()
    myl.setMaterial(obj, mat)
