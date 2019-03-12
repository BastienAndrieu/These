import bpy
import bpy_extras
from mathutils import Vector

##############################################
def convert_3d_to_2d_coords(xyz):
    scene = bpy.context.scene
    cam = scene.camera
    
    render_scale = scene.render.resolution_percentage / 100
    render_w = int(scene.render.resolution_x * render_scale)
    render_h = int(scene.render.resolution_y * render_scale)

    co = Vector(xyz)
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, co)
    u = co_2d.x - 2.*bpy.data.cameras["Camera"].shift_x
    v = co_2d.y - 2.*float(render_w)/float(render_h)*bpy.data.cameras["Camera"].shift_y
    return u, v
    
##############################################
def get_2d_coordinates(obj):
    scene = bpy.context.scene
    cam = scene.camera
    
    render_scale = scene.render.resolution_percentage / 100
    render_w = int(scene.render.resolution_x * render_scale)
    render_h = int(scene.render.resolution_y * render_scale)

    co = obj.location
    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, co)
    x = co_2d.x - 2.*bpy.data.cameras["Camera"].shift_x
    y = co_2d.y - 2.*float(render_w)/float(render_h)*bpy.data.cameras["Camera"].shift_y
    print(obj.name)
    print("   2D Coords:   ", x, ', ', y)
    print("   Pixel Coords:", x*render_w, ', ', y*render_h)
    print("")
    return x, y
##############################################
def get_all_visible_2d_coordinates():
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
