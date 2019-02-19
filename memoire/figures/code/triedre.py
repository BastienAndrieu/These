import bpy
import sys
sys.path.append('/stck/bandrieu/Bureau/Python/mylibs/')
import my_lib as myl
import numpy as np


a = 1.0
thck = a/100.

o = bpy.context.scene.cursor_location
o = np.array([o.x, o.y, o.z])
o[2] = bpy.data.objects["Plane"].location[2] + thck

e = np.eye(3)
zero = np.zeros(3)
ob = myl.addEmpty(name="O", location=o)
names = ["Ox","Oy","Oz"]
for i in range(3):
    #myl.addPolyline(np.vstack([o, o + a*e[i]]), e[i], thck, 0)
    myl.addPolyline(np.vstack([zero, a*e[i]]), e[i], thck, 0)
    if i == 0:
        name = "ObjCurve"
    else:
        name = "ObjCurve."+format(i,'03')
    bpy.data.objects[name].location = o
    ob = myl.addEmpty(name=names[i], location=o + a*e[i])
