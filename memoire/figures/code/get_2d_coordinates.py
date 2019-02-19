# Test the function using the active object (which must be a camera)
# and the 3D cursor as the location to find.
import bpy
import bpy_extras

scene = bpy.context.scene
obj = bpy.context.object

# select 3d location
#co = bpy.context.scene.cursor_location
co = bpy.data.objects["Sphere.006"].location

co_2d = bpy_extras.object_utils.world_to_camera_view(scene, obj, co)
print("2D Coords:", co_2d.x, ', ', co_2d.y)

# If you want pixel coords
render_scale = scene.render.resolution_percentage / 100
render_size = (
        int(scene.render.resolution_x * render_scale),
        int(scene.render.resolution_y * render_scale),
        )
print("Pixel Coords:", (
      co_2d.x * render_size[0],#round(co_2d.x * render_size[0]),
      co_2d.y * render_size[1],#round(co_2d.y * render_size[1]),
      ))
