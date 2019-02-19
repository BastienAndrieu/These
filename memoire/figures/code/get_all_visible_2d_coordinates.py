#https://blender.stackexchange.com/questions/882/how-to-find-image-coordinates-of-the-rendered-vertex
import bpy
import bpy_extras

scene = bpy.context.scene
cam = scene.camera

render_scale = scene.render.resolution_percentage / 100
render_w = int(scene.render.resolution_x * render_scale)
render_h = int(scene.render.resolution_y * render_scale)
               

for obj in bpy.data.objects:
    if obj.hide or obj.hide_render or obj.type == 'CAMERA' or obj.type == 'LAMP':
        continue
    co = obj.location
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, co)
    print(obj.name)
    print("   2D Coords:   ", co_2d.x, ', ', co_2d.y)
    print("   Pixel Coords:", co_2d.x*render_w, ', ', co_2d.y*render_h)
    print("")
