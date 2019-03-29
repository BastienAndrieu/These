import bpy
import bpy_extras

import numpy
from numpy.polynomial.chebyshev import chebgrid2d

import sys
sys.path.append('/stck/bandrieu/Bureau/Python/mylibs/')
sys.path.append('/d/bandrieu/GitHub/These/memoire/figures/code/')
import my_lib as myl
import lib_color as mycolors
import lib_fig

#################################################

pthin = '/d/bandrieu/GitHub/FFTsurf/test/demo_intersection/simple/'
pthout = '/d/bandrieu/GitHub/These/memoire/figures/'

#################################################
scene = bpy.context.scene

myl.blankScene()

## Set scene
scene.render.resolution_x = 800
scene.render.resolution_y = 600
scene.render.resolution_percentage = 100

scene.world.horizon_color = [1,1,1]

## Lighting
# Environment lighting
scene.world.light_settings.use_environment_light = True
scene.world.light_settings.environment_energy = 0.2
scene.world.light_settings.environment_color = 'PLAIN'
scene.world.light_settings.use_ambient_occlusion = True
scene.world.light_settings.ao_factor = 1
scene.world.light_settings.ao_blend_type = 'MULTIPLY'
scene.world.light_settings.samples = 10

# Create new lamp datablock
lamp_data = bpy.data.lamps.new(name="KeyLight", type='POINT')
lamp_data.energy = 1.5#2.0
lamp_data.distance = 30
lamp_data.shadow_method = 'RAY_SHADOW'
lamp_data.shadow_ray_samples = 10
lamp_data.shadow_soft_size = 3.0
lamp_data.shadow_color = 0.6*numpy.ones(3)

# Create new object with our lamp datablock
lamp_object = bpy.data.objects.new(name="KeyLight", object_data=lamp_data)

# Link lamp object to the scene so it'll appear in this scene
scene.objects.link(lamp_object)

lamp_object.location = [3.73603, -0.89244, 5.9]#[4.41, -2.22, 5.90]

# And finally select it make active
lamp_object.select = True
scene.objects.active = lamp_object


## Camera
cam = scene.camera
cam.location = [4.40783, -5.13018, 4.29261]
cam.rotation_euler = numpy.radians(numpy.array([55.137, 0.574, 44.611]))
cam.data.lens = 35.0


#########################################

m = 101
u = numpy.linspace(-1.0,1.0,m)

#clf = mycolors.sample_colormap('I2', 2)
clf = numpy.loadtxt('/d/bandrieu/GitHub/These/memoire/figures/code/demo_EoS_brep_palette_modif.dat')
clf = clf[[1,7]]#clf[[6,8]]#clf[[6,1]]
clf = mycolors.cc_hsv(clf, fs=1.2, fv=1.0)

# add surfaces
for isurf in range(2):
    c = myl.readCoeffs(pthin + 'c_' + str(isurf+1) + '.cheb')
    surf = myl.addTensorProductPatch(chebgrid2d(u, u, c[:,:,0]),
                                     chebgrid2d(u, u, c[:,:,1]),
                                     chebgrid2d(u, u, c[:,:,2]),
                                     name="surf_"+str(isurf),
                                     location=[0,0,0],
                                     smooth=True,
                                     color=clf[isurf],
                                     alpha=1)
    mat = bpy.data.materials["mat_surf_"+str(isurf)]
    mat.diffuse_intensity = 1.0
    mat.specular_shader = 'COOKTORR'
    mat.specular_intensity = 0.265
    mat.specular_hardness = 21

    
## Freestyle
freestyle = scene.render.layers.active.freestyle_settings
freestyle.use_smoothness = True

bpy.ops.scene.freestyle_lineset_add() #2

lineset = freestyle.linesets["LineSet"]
lineset.visibility = 'VISIBLE'#'RANGE'#

lineset = freestyle.linesets["LineSet 2"]
lineset.visibility = 'HIDDEN'
linestyle = bpy.data.linestyles["LineStyle.001"]
linestyle.use_dashed_line = True
linestyle.dash1 = 7
linestyle.gap1 = 9

for lineset in freestyle.linesets:
    lineset.select_silhouette = False
    lineset.select_border = True
    lineset.select_crease = False
    lineset.select_edge_mark = False

for linestyle in bpy.data.linestyles:
    linestyle.color = [0,0,0]
    linestyle.caps = "ROUND"
    linestyle.thickness = 1.7
    linestyle.geometry_modifiers["Sampling"].sampling = 0.1

##




"""
bpy.ops.mesh.primitive_plane_add(location=[0, 0, -0.4693])
plan = bpy.data.objects['Plane']
plan.scale = 20*numpy.ones(3)

mat = bpy.data.materials.new("mat_plan")
mat.use_only_shadow = True
plan.data.materials.append(mat)
"""

"""
lib_fig.fit_camera_to_meshes([bpy.data.objects['surf_0'].data,
                              bpy.data.objects['surf_1'].data])

cam.data.sensor_width *= 1.01

## Intersection curve
curve = numpy.loadtxt(pthin + 'curve_xyz.dat')

f = open(pthout + 'data/fig_simple_intersection_curve.dat', 'w')
for p in curve:
    x, y = lib_fig.convert_3d_to_2d_coords(p)
    f.write(str(x) + ' ' + str(y) + '\n')
f.close()

ipoint = int(numpy.around(0.33*len(curve)))
x, y = lib_fig.convert_3d_to_2d_coords(curve[ipoint])
f = open(pthout + 'data/fig_simple_intersection_point.dat', 'w')
f.write(str(x) + ', ' + str(y) + '\n')
for isurf in range(2):
    uv = numpy.loadtxt(pthin + 'curve_uv'+str(isurf+1)+'.dat')
    
    a = uv[ipoint]
    v = uv[ipoint+1] - a
    v = numpy.array([-v[1], v[0]])/numpy.hypot(v[0], v[1])
    f.write(str(a[0]) + ', ' + str(a[1]) + '\n')
    f.write(str(v[0]) + ', ' + str(v[1]) + '\n')
f.close()
"""


