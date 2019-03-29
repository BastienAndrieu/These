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
    u = co_2d.x - 2.*cam.data.shift_x
    v = co_2d.y - 2.*float(render_w)/float(render_h)*cam.data.shift_y
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
##############################################
def fit_camera_to_meshes(meshes):
    scn = bpy.context.scene
    cam = scn.camera
    ratio = scn.render.resolution_x/scn.render.resolution_y
    shx = cam.data.shift_x
    shy = cam.data.shift_y
    ### compute original bounding box in image space
    xmin = 1
    xmax = 0
    ymin = 1
    ymax = 0
    for me in meshes:
        for v in me.vertices:
            co_obj = v.co
            co_img = bpy_extras.object_utils.world_to_camera_view(scn, cam, co_obj)
            x = co_img.x - shx
            y = co_img.y - shy*ratio
            xmin = min(xmin, x)
            xmax = max(xmax, x)
            ymin = min(ymin, y)
            ymax = max(ymax, y)
    xctr = 0.5*(xmin + xmax)
    xrng = 0.5*(xmax - xmin)
    yctr = 0.5*(ymin + ymax)
    yrng = 0.5*(ymax - ymin)
    #print('center :', xctr, ',', yctr)
    #print('ranges :', xrng, ',', yrng)
    ### adjust camera sensor size to fit 
    scl = 0.5/max(xrng, yrng)
    #print('scl =',scl)
    cam.data.sensor_width /= scl
    ### adjust camera shift to align the center of the view with
    #   the center of the bounding box
    cam.data.shift_x = scl*(xctr - 0.5)
    cam.data.shift_y = scl*(yctr - 0.5)/ratio
    return
##############################################
def get_scene_xyz_AABB():
    BIG = 1e6
    aabb = [BIG*Vector([1,1,1]), -BIG*Vector([1,1,1])]
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            bbox_corners = [obj.matrix_world * Vector(corner) for corner in obj.bound_box]
            for corner in bbox_corners:
                for i in range(3):
                    aabb[0][i] = min(aabb[0][i], corner[i])
                    aabb[1][i] = max(aabb[1][i], corner[i])
    return aabb
