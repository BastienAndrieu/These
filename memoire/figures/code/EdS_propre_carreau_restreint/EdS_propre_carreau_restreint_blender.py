import bpy
import bpy_extras
import numpy
from numpy.polynomial.chebyshev import chebgrid2d

import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python/')
import lib_blender_edit as lbe
import lib_blender_util as lbu
import lib_blender_figures as lbf
import lib_chebyshev as lcheb

pthin = '/d/bandrieu/GitHub/These/memoire/figures/data/EdS_propre_carreau_restreint/'


#################################################
## Set Scene
scene = bpy.context.scene
cam = scene.camera
lbu.clear_scene(meshes=True, lamps=True, cameras=False)

lbu.set_scene(resolution_x=800,
              resolution_y=600,
              resolution_percentage=100,
              alpha_mode='SKY',
              horizon_color=[1,1,1],
              light_samples=16,
              use_environment_light=True,
              environment_energy=0.3,
              environment_color='PLAIN')

## Set Lighting
lbu.add_point_light(name="lamp",
                    energy=1,
                    shadow_method='RAY_SHADOW',
                    shadow_ray_samples=1,
                    shadow_soft_size=0.01,
                    location=[1.37, -1.89, 0.9])
#################################################


#################################################
bpy.ops.import_scene.obj(
    filepath='/d/bandrieu/GitHub/FFTsurf/test/demo_EoS_MAT/mesh_eos_optim.obj',
    axis_forward='Y', 
    axis_up='Z'
)
eds = bpy.data.objects['mesh_eos_optim']
lbe.set_smooth(eds)

mat_eds = bpy.data.materials['mat1']
mat_eds.diffuse_color = (0.5,1,0.5)
mat_eds.diffuse_intensity = 1
mat_eds.specular_intensity = 0.2
mat_eds.specular_hardness = 30
mat_eds.use_transparency = True
mat_eds.alpha = 0.25
eds.show_transparent = True
eds.hide_render = True

bpy.ops.object.select_all(action='DESELECT')
eds.select = True
bpy.ops.view3d.camera_to_view_selected() # set camera to fit surf in screen
cam.data.angle += numpy.pi/180.0 # increase FOV angle by 1 degree
#################################################

#################################################
# CHECKER TEXTURE
imgchecker = bpy.data.images.load(filepath='/d/bandrieu/GitHub/These/memoire/figures/code/BRep/checker.png')
texchecker = bpy.data.textures.new('texture_checker', 'IMAGE')
texchecker.image = imgchecker

mat_surf = bpy.data.materials.new('mat_surf')
mat_surf.use_raytrace = False
mat_surf.use_shadows = False
mat_surf.use_shadeless = True
mat_surf.use_cast_buffer_shadows = False
slot = mat_surf.texture_slots.add()
slot.texture = texchecker
slot.texture_coords = 'UV'
slot.diffuse_color_factor = 1
#################################################

#################################################
m = 100
u = numpy.linspace(-1,1,m)
v = 0.5*(u + 1)
clr = [(1,1,1), (1,0.5,0.5)]
suff = ['skeleton', 'eos']

for isurf in range(2):
    tri = numpy.loadtxt(pthin + 'tri' + str(isurf) + '.dat', dtype=int)-1
    xyz = numpy.loadtxt(pthin + 'xyz' + str(isurf) + '.dat', dtype=float)
    
    verts = [[x for x in p] for p in xyz]
    faces = [[int(v) for v in t] for t in tri]

    obj = lbu.pydata_to_mesh(verts,
                             faces,
                             name='face_'+str(isurf))
    mat_face = bpy.data.materials.new('mat_face_'+str(isurf))
    mat_face.diffuse_color = clr[isurf]
    mat_face.diffuse_intensity = 1
    mat_face.specular_intensity = 0.2
    mat_face.specular_hardness = 30
    mat_face.use_transparency = False
    obj.data.materials.append(mat_face)
    lbe.set_smooth(obj)




    
    c = lcheb.read_polynomial2(pthin + 'c_' + suff[isurf] + '.cheb')
    xyz = chebgrid2d(u, u, c)
    verts, faces = lbu.tensor_product_mesh_vf(xyz[0], xyz[1], xyz[2])
    
    surf = lbu.pydata_to_mesh(verts,
                              faces,
                              name='surf_'+str(isurf))

    bpy.ops.object.select_all(action='DESELECT')
    scene.objects.active = surf
    surf.select = True
    bpy.ops.object.mode_set(mode='EDIT')
    print('   unwrap UVs...')
    bpy.ops.uv.unwrap(method='ANGLE_BASED',
                      fill_holes=True,
                      correct_aspect=True,
                      use_subsurf_data=False,
                      margin=0.001)
    bpy.ops.object.mode_set(mode='OBJECT')
    uvlayer = surf.data.uv_layers.active
    print('   edit UVs...')
    for j in range(m-1):
        for i in range(m-1):
            k = i + j*(m-1)
            f = surf.data.polygons[k]
            for l in [0,3]:
                uvlayer.data[f.loop_start + l].uv[0] = v[i]
            for l in [1,2]:
                uvlayer.data[f.loop_start + l].uv[0] = v[i+1]
            for l in [0,1]:
                uvlayer.data[f.loop_start + l].uv[1] = v[j]
            for l in [2,3]:
                uvlayer.data[f.loop_start + l].uv[1] = v[j+1]
    """
    mat_surf.diffuse_color = clr[i]
    mat_surf.diffuse_intensity = 1
    mat_surf.specular_intensity = 0.2
    mat_surf.specular_hardness = 30
    mat_surf.use_transparency = True
    mat_surf.alpha = 0.2
    """
    surf.data.materials.append(mat_surf)
    surf.show_transparent = True
    lbe.set_smooth(surf)

    surf.hide_render = True
#################################################


    

################################################
# FREESTYLE SETTINGS
scene.render.use_freestyle = True
freestyle = scene.render.layers.active.freestyle_settings
freestyle.use_smoothness = True

bpy.ops.scene.freestyle_lineset_add() #2

freestyle.linesets[0].visibility = 'VISIBLE'
freestyle.linesets[1].visibility = 'HIDDEN'

for lineset in freestyle.linesets:
    lineset.select_silhouette = False
    lineset.select_border = True
    lineset.select_crease = False
    lineset.select_edge_mark = False

bpy.data.linestyles[1].use_dashed_line = True
bpy.data.linestyles[1].dash1 = 7
bpy.data.linestyles[1].gap1 = 7

for linestyle in bpy.data.linestyles:
    linestyle.thickness = 1.5
    linestyle.geometry_modifiers["Sampling"].sampling = 0.1
################################################
