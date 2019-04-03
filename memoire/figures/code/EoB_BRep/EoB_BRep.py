import bpy
import bpy_extras
from mathutils import Vector
import numpy
import sys
sys.path.append('/stck/bandrieu/Bureau/Python/mylibs/')
import my_lib as myl


sys.path.append('/d/bandrieu/GitHub/Code/Python/')
import lib_blender_edit as lbe
import lib_blender_util as lbu
import lib_blender_figures as lbf
import lib_color as lco

pthin = '/d/bandrieu/GitHub/FFTsurf/test/demo_EoS_brep/'
pthout = '/d/bandrieu/GitHub/These/memoire/figures/'


args = sys.argv
if len(args) < 4:
    EoB = False
else:
    EoB = bool(int(args[3]))
    
if len(args) < 5:
    rendu = True
else:
    rendu = bool(int(args[4]))

if EoB:
    suffin = '_new'
    suffout = '2'
else:
    suffin = ''
    suffout = '1'

#######################################################
##
edge_flw = 1.8
edge_tck = 2.e-3#1.25e-3
vert_rad = 3.*edge_tck
vert_ns = 32
vert_nr = vert_ns


#################################################
## Set Scene
scene = bpy.context.scene
lbu.clear_scene(meshes=True, lamps=True, cameras=False)

lbu.set_scene(resolution_x=800,
              resolution_y=800,
              resolution_percentage=100,
              alpha_mode='SKY',
              horizon_color=[1,1,1],
              light_samples=16,
              use_environment_light=True,
              environment_energy=0.3,
              environment_color='PLAIN')

## Set Lighting
lbu.add_point_light(name="lamp",
                    energy=1.2,
                    shadow_method='RAY_SHADOW',
                    shadow_ray_samples=16,
                    shadow_soft_size=2.0,
                    location=[3.75,1.65,3.20])

## Set Camera
cam = scene.camera
cam.location = [2.102,1.798,1.104]
cam.rotation_euler = numpy.radians(numpy.array([66.7,0.778,132.2]))
cam.data.angle = numpy.radians(37.72)
#################################################

name = 'mesh'
if EoB:
    name += '_eos'
bpy.ops.import_scene.obj(filepath=pthin+name+'/'+name+'.obj',
                         axis_forward='Y', axis_up='Z')
obj = bpy.data.objects[name]
lbe.set_smooth(obj)


nf = len(bpy.data.materials)-1
clf = numpy.loadtxt(pthout + 'data/BRep/face_color.dat')
clf = lco.cc_hsv(clf, fs=1.2, fv=1.0)

## set face materials
for i in range(nf):
    mat = bpy.data.materials["mat"+str(i+1)]
    mat.diffuse_color = clf[i%len(clf)]
    mat.diffuse_intensity = 1.0
    mat.specular_intensity = 0.1
    mat.specular_hardness = 8.0

## vertices
fin = open(pthin + 'debug/verts' + suffin + '.dat','r')
fout = open(pthout + 'data/EoB_BRep/verts' + suffout + '.dat','w')
for l in fin:
    v = [float(a) for a in l.split()]
    if len(v) > 7:# keep only vertices incident to at least 3 faces
        visible = lbu.is_visible_point(v[0:3], cam, tol=1.e-3, nrs=32, clean=True)
        x, y = lbf.convert_3d_to_2d_coords(v[0:3])
        fout.write(str(x) + ', ' + str(y) + ', ' + str(int(visible)) + '\n')
fin.close()
fout.close()
    
## trace edges
f = open(pthin + 'debug/edges_xyz' + suffin + '.dat', 'r')
ne = int(f.readline())
edgesobj = []
for ie in range(ne):
    np = int(f.readline())
    p = numpy.zeros((np,3))
    for ip in range(np):
        x,y,z = [float(a) for a in f.readline().split()]
        p[ip,0] = x
        p[ip,1] = y
        p[ip,2] = z
    myl.addFreestylePolyline(p,
                             name="edge_"+str(ie),
                             location=[0,0,0],
                             color=[0,0,0],
                             thickness=2.0,
                             newlineset=False)
    obj = bpy.data.objects["edge_"+str(ie)]
    obj.hide = True
    obj.hide_render = True
    edgesobj.append(obj)
f.close()


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
linestyle.thickness = edge_flw
linestyle.use_dashed_line = True
linestyle.dash1 = 7
linestyle.gap1 = linestyle.dash1

for linestyle in bpy.data.linestyles:
    linestyle.geometry_modifiers["Sampling"].sampling = 0.05

if not EoB:
    for lineset in freestyle.linesets:
        lineset.select_crease = True

## Renders
if rendu:
    scene.render.image_settings.file_format='PNG'
    pthimg = pthout + 'images/EoB_BRep/'

    # 1) faces
    scene.render.alpha_mode = 'TRANSPARENT'#'SKY'
    scene.render.use_freestyle = False
    scene.render.filepath = pthimg + 'shell' + suffout
    bpy.ops.render.render(write_still=True)

    # 2) visible edges
    for obj in edgesobj:
        obj.hide = False
        obj.hide_render = False
    if not EoB:
        for i in range(18):
            bpy.data.objects['edge_'+str(i)].hide_render = True
    scene.render.use_freestyle = True
    scene.render.alpha_mode = 'TRANSPARENT'

    for i in range(nf):
        mat = bpy.data.materials["mat"+str(i+1)]
        mat.use_transparency = True
        mat.alpha = 0
        mat.use_raytrace = False
        mat.use_shadows = False
        mat.use_cast_buffer_shadows = False
        mat.transparency_method = 'MASK'

    freestyle.linesets["LineSet"].show_render = True
    freestyle.linesets["LineSet 2"].show_render = False
    scene.render.filepath = pthimg + 'edges_visible' + suffout
    bpy.ops.render.render(write_still=True)

    # 3) hidden edges
    freestyle.linesets["LineSet"].show_render = False
    freestyle.linesets["LineSet 2"].show_render = True
    scene.render.filepath = pthimg + 'edges_hidden' + suffout
    bpy.ops.render.render(write_still=True)

    exit()
