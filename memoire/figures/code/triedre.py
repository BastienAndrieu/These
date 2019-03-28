import bpy
import sys
sys.path.append('/stck/bandrieu/Bureau/Python/mylibs/')
import my_lib as myl
import numpy as np

sys.path.append('/d/bandrieu/GitHub/These/memoire/figures/code/')
import lib_fig


a = 1.0
thck = a/100.

o = bpy.context.scene.cursor_location
o = np.array([o.x, o.y, o.z])
o[2] = bpy.data.objects["Plane"].location[2] + thck


x, y = lib_fig.convert_3d_to_2d_coords(o)
print('O :', x, ',', y)

e = np.eye(3)
zero = np.zeros(3)
ob = myl.addEmpty(name="O", location=o)
names = ["Ox","Oy","Oz"]
for i in range(3):
    obj = myl.addPolyline(np.vstack([zero, a*e[i]]), e[i], thck, 0)
    obj.location = o
    ob = myl.addEmpty(name=names[i], location=o + a*e[i])
    x, y = lib_fig.convert_3d_to_2d_coords(o + a*e[i])
    print(names[i], ':', x, ',', y)
