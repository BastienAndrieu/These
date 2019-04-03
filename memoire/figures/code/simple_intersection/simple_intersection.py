import bpy
import bpy_extras

import numpy
from numpy.polynomial.chebyshev import chebgrid2d

import sys
sys.path.append('/stck/bandrieu/Bureau/Python/mylibs/')
import my_lib as myl
sys.path.append('/d/bandrieu/GitHub/Code/Python/')
import lib_color as lco
import lib_blender_util as lbu
import lib_blender_edit as lbe
import lib_blender_figures as lbf

#################################################

pthin = '/d/bandrieu/GitHub/FFTsurf/test/demo_intersection/simple/'
pthout = '/d/bandrieu/GitHub/These/memoire/figures/'

#################################################
## Set Scene
scene = bpy.context.scene
lbu.clear_scene(meshes=True, lamps=True, cameras=False)

lbu.set_scene(resolution_x=800,
              resolution_y=600,
              resolution_percentage=100,
              alpha_mode='SKY',
              horizon_color=[1,1,1],
              light_samples=16,
              use_environment_light=True,
              environment_energy=0.2,
              environment_color='PLAIN')

## Set Lighting
lamp = lbu.add_point_light(name="lamp",
                           energy=1.5,
                           shadow_method='RAY_SHADOW',
                           shadow_ray_samples=16,
                           shadow_soft_size=3.0,
                           location=[3.73603, -0.89244, 5.9])
lamp.data.distance = 30
lamp.data.shadow_color = 0.6*numpy.ones(3)


## Set Camera
cam = scene.camera
cam.location = [4.40783, -5.13018, 4.29261]
cam.rotation_euler = numpy.radians(numpy.array([55.137, 0.574, 44.611]))
cam.data.lens = 35.0
#################################################

#########################################

m = 101
u = numpy.linspace(-1.0,1.0,m)

#clf = mycolors.sample_colormap('I2', 2)
clf = numpy.loadtxt(pthout + 'data/BRep/face_color.dat')
clf = clf[[1,7]]
clf = lco.cc_hsv(clf, fs=1.2, fv=1.0)

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


lbf.fit_camera_to_meshes([bpy.data.objects['surf_0'].data,
                              bpy.data.objects['surf_1'].data])

cam.data.sensor_width *= 1.01

## Intersection curve
curve = numpy.loadtxt(pthin + 'curve_xyz.dat')

f = open(pthout + 'data/simple_intersection/curve_xy.dat', 'w')
for p in curve:
    x, y = lbf.convert_3d_to_2d_coords(p)
    f.write(str(x) + ' ' + str(y) + '\n')
f.close()

ipoint = int(numpy.around(0.33*len(curve)))
x, y = lbf.convert_3d_to_2d_coords(curve[ipoint])
f = open(pthout + 'data/simple_intersection/point.dat', 'w')
f.write(str(x) + ', ' + str(y) + '\n')
for isurf in range(2):
    uv = numpy.loadtxt(pthin + 'curve_uv'+str(isurf+1)+'.dat')
    
    a = uv[ipoint]
    v = uv[ipoint+1] - a
    v = numpy.array([-v[1], v[0]])/numpy.hypot(v[0], v[1])
    f.write(str(a[0]) + ', ' + str(a[1]) + '\n')
    f.write(str(v[0]) + ', ' + str(v[1]) + '\n')
f.close()


################################################
# Renders

scene.render.alpha_mode = 'TRANSPARENT'
scene.render.image_settings.file_format='PNG'
pthimg = '/d/bandrieu/GitHub/These/memoire/figures/images/simple_intersection/'


### 1st render: face
scene.render.use_freestyle = False
scene.render.filepath = pthimg + 'surfaces'
bpy.ops.render.render( write_still=True )

### 2nd render: visible borders
freestyle.linesets["LineSet 2"].show_render = False
scene.render.use_freestyle = True
for i in range(2):
    mat = bpy.data.materials["mat_surf_"+str(i)]
    mat.use_transparency = True
    mat.alpha = 0
    mat.transparency_method = 'MASK'
    mat.use_raytrace = False
    mat.use_shadows = False
    mat.use_cast_buffer_shadows = False

scene.render.filepath = pthimg + 'borders_visible'
bpy.ops.render.render( write_still=True )

### 3rd render: hidden borders
freestyle.linesets["LineSet"].show_render = False
freestyle.linesets["LineSet 2"].show_render = True

scene.render.filepath = pthimg + 'borders_hidden'
bpy.ops.render.render( write_still=True )
