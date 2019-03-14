import bpy
import numpy
from mathutils import Vector

def is_visible_point(xyz, cam, tol=1.e-3, nrs=32, clean=True):
    bpy.ops.mesh.primitive_uv_sphere_add(location=xyz, size=tol, segments=nrs, ring_count=nrs)
    obj = bpy.context.active_object
    result = bpy.context.scene.ray_cast(start=cam.location, end=Vector(xyz))
    visible = (result[1] == obj)
    if clean:
        bpy.ops.object.select_all(action='DESELECT')
        obj.select = True
        bpy.ops.object.delete()
    return visible

def cut_into_constant_visibility_segments(polyline, camera, tol=1.e-3, eps=1.e-3, nrs=32, clean=True):
    #
    visible = []
    hidden = []
    #
    segment = numpy.empty((0,3))
    for i, x in enumerate(polyline):
        vis = is_visible_point(x, camera, tol, nrs, clean)
        if i == 0:
            segment = numpy.vstack((segment, x)) # start the first segment
        else:
            if vis == visprev:
                if i < len(polyline)-1:
                    segment = numpy.vstack((segment, x)) # continue the current segment
            else:
                xL = numpy.copy(segment[-1])
                xR = numpy.copy(x)
                vL = visprev
                vR = vis
                while True:
                    xM = 0.5*(xL + xR)
                    dist = numpy.sqrt(numpy.sum(numpy.power(xL - xR,2)))
                    if dist > 2.*eps: break
                    vM = is_visible_point(xM, camera, tol, nrs, clean)
                    if vM == vL:
                        xL = numpy.copy(xM)
                    else:
                        xR = numpy.copy(xM)
                # end the current segment, ...
                segment = numpy.vstack((segment, xM))
                # ... add it to the appropriate list, ...
                if visprev:
                    visible.append(segment)
                else:
                    hidden.append(segment)
                # ... and start a new segment
                segment = numpy.empty((0,3))
                segment = numpy.vstack((segment, xM))
        visprev = vis
    segment = numpy.vstack((segment, polyline[-1]))
    if vis:
        visible.append(segment)
    else:
        hidden.append(segment)
    return visible, hidden
                
