import bpy
import bpy_extras
import numpy
import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python/')
import lib_blender_edit as lbe
import lib_blender_util as lbu
import lib_blender_figures as lbf

#################################################
pthin = '/d/bandrieu/GitHub/FFTsurf/test/demo_EoS_brep/'
pthout = '/d/bandrieu/GitHub/These/memoire/figures/data/piecewise_smooth_surface/'
#################################################

tolvis = 1.e-3
nrsvis = 32

#################################################
# SET SCENE
scene = bpy.context.scene
lbu.clear_scene(meshes=True, lamps=True, cameras=False)

lbu.set_scene(resolution_x=800,
              resolution_y=800,
              resolution_percentage=100,
              alpha_mode='TRANSPARENT',
              horizon_color=[1,1,1],
              light_samples=16,
              use_environment_light=True,
              environment_energy=0.3,
              environment_color='PLAIN')

# Set Lighting
lbu.add_point_light(name="lamp",
                    energy=1.2,
                    shadow_method='RAY_SHADOW',
                    shadow_ray_samples=16,
                    shadow_soft_size=2.0,
                    location=[3.75,1.65,3.20])

# Set Camera
cam = scene.camera
cam.location = [2.102,1.798,1.104]
cam.rotation_euler = numpy.radians(numpy.array([66.7,0.778,132.2]))
cam.data.angle = numpy.radians(37.72)
#################################################



#################################################
# READ BREP VERTICES XYZ
verts_xyz = []
f = open(pthin + 'debug/verts.dat', 'r')
for line in f:
    data = [float(x) for x in line.split()]
    verts_xyz.append(data[0:3])
f.close()
#################################################



#################################################
# READ BREP EDGES XYZ
edges_xyz = []
f = open(pthin + 'debug/edges_xyz.dat', 'r')
ne = int(f.readline())
for i in range(ne):
    n = int(f.readline())
    xyz = numpy.zeros((n,3))
    for j in range(n):
        xyz[j] = [float(x) for x in f.readline().split()]
    edges_xyz.append(xyz)
f.close()
#################################################



#################################################
# READ HYPERGRAPH DATA
iverts = []
iedges = []
f = open(pthin + 'debug/hyperedges.dat', 'r')
nhe = int(f.readline())
for i in range(nhe):
    nedg = int(f.readline())
    verts = [int(v)-1 for v in f.readline().split()]
    if verts[0] != verts[1]:
        for v in verts:
            if v not in iverts: iverts.append(v)
    f.readline() # hyperfaces
    for j in range(nedg):
        e, ih = [int(h)-1 for h in f.readline().split()]
        if e not in iedges: iedges.append(e)
f.close()
#################################################


#################################################
# ADD SINGULAR CURVES
"""
mat_edge = bpy.data.materials.new('mat_edge')
mat_edge.diffuse_color = (0,0,0)
mat_edge.use_shadeless = True

for e in iedges:
    obj = lbu.pydata_to_polyline(
        points=edges_xyz[e],
        name='edge'+str(e+1),
        thickness=2e-3
    )
    obj.data.materials.append(mat_edge)
"""
# Freestyle settings
freestyle = scene.render.layers.active.freestyle_settings
freestyle.use_smoothness = True

bpy.ops.scene.freestyle_lineset_add() #2

lineset = freestyle.linesets["LineSet"]
lineset.visibility = 'VISIBLE'

lineset = freestyle.linesets["LineSet 2"]
lineset.visibility = 'HIDDEN'
linestyle = bpy.data.linestyles["LineStyle.001"]
linestyle.use_dashed_line = True
linestyle.dash1 = 7
linestyle.gap1 = linestyle.dash1

for lineset in freestyle.linesets:
    lineset.select_silhouette = False
    lineset.select_border = False
    lineset.select_crease = True
    lineset.select_edge_mark = True

for linestyle in bpy.data.linestyles:
    linestyle.thickness = 2
    linestyle.color = (0,0,0)
    linestyle.caps = "ROUND"
    linestyle.geometry_modifiers["Sampling"].sampling = 0.1


# EXPORT AS SVG (separate visible and hidden in two SVG output files)
# (...)
#################################################


#################################################
# ADD PIECEWISE SMOOTH SURFACE
print('Adding surface...')
bpy.ops.import_scene.obj(
    filepath=pthin+'mesh/mesh.obj',
    axis_forward='Y',
    axis_up='Z'
)
print('done.')        
obj = bpy.data.objects['mesh']
lbe.set_smooth(obj)

scene.objects.active = obj
for i in range(len(bpy.data.materials)):
    bpy.ops.object.material_slot_remove()

mat = bpy.data.materials.new('mat_surface')
mat.diffuse_color = (1,1,1)
mat.diffuse_intensity = 1
mat.specular_intensity = 0.2
mat.specular_hardness = 20
obj.data.materials.append(mat)

bpy.ops.view3d.camera_to_view_selected() # set camera to fit surface in screen
cam.data.angle += numpy.pi/180.0 # increase FOV angle by 1 degree
#################################################


#################################################
# ADD SINGULAR POINTS
"""
for v in iverts:
    bpy.ops.mesh.primitive_uv_sphere_add(
        location=verts_xyz[v],
        size=1e-2
    )
"""
print('Writing corners_xyza.dat...')
f = open(pthout + 'corners_xya.dat', 'w')
for v in iverts:
    xyz = verts_xyz[v]
    u, v = lbf.convert_3d_to_2d_coords(xyz, normalize=True)
    visible = lbu.is_visible_point(
        xyz=xyz,
        cam=cam,
        tol=tolvis,
        nrs=nrsvis,
        clean=True
    )
    f.write('%s, %s, %d\n' % (u, v, int(visible)))
f.close()
print('done.')
#################################################



#################################################
# RENDERING
scene.render.image_settings.file_format='PNG'
pthimg = '/d/bandrieu/GitHub/These/memoire/figures/images/piecewise_smooth_surface/'

print('Rendering surface...')
### 1st render: surface
scene.render.use_freestyle = False
scene.render.filepath = pthimg + 'surface'
bpy.ops.render.render(write_still=True)
print('done.')

print('Rendering visible edges...')
### 2nd render: visible edges
mat = bpy.data.materials['mat_surface']
mat.use_transparency = True
mat.alpha = 0
mat.transparency_method = 'MASK'
mat.use_raytrace = False
mat.use_shadows = False
mat.use_cast_buffer_shadows = False

scene.render.use_freestyle = True
freestyle.linesets[0].show_render = True
freestyle.linesets[1].show_render = False
scene.render.filepath = pthimg + 'edges_visible'
bpy.ops.render.render(write_still=True)
print('done.')

print('Rendering hidden edges...')
### 3rd render: hidden edges
freestyle.linesets[0].show_render = False
freestyle.linesets[1].show_render = True
scene.render.filepath = pthimg + 'edges_hidden'
bpy.ops.render.render(write_still=True)
print('done.')

#################################################

