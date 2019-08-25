import bpy
import numpy
from numpy.polynomial.chebyshev import chebgrid2d

import sys
sys.path.append('/home/bastien/GitHub/Code/Python/')
import lib_blender_util as lbu
import lib_blender_edit as lbe
import lib_EoS as leos
import lib_chebyshev as lcheb
from lib_linalg import matmul

#####################################################
def radius_function(x, y, z):
    return 0.8*(1.0 + 0.05*numpy.cos(5.0*x+y+z))
#####################################################
def div_colormap(v, vmin=-1, vmax=1): # -1 < v < 1
    v = max(vmin, min(vmax, v))
    #col_stops = numpy.array([[0,0,1], [1,1,1], [1,0,0]])
    col_stops = numpy.array([[18,117,171], [248,243,232], [160,55,55]])/255.
    if v > 0:
        v /= vmax
        v = numpy.sqrt(v)
        return (1 - v)*col_stops[1] + v*col_stops[2]
    else:
        v /= vmin
        v = numpy.sqrt(v)
        return (1 - v)*col_stops[1] + v*col_stops[0]
#####################################################



m = 50
n = 50
u = numpy.linspace(-1,1,m)
v = numpy.linspace(-1,1,n)
ucgl = lcheb.cgl_nodes(3)

chebpoly = []
nappes = []
nappes_eos = []

XYZR = []
E = []
for inappe in range(2):
    # convert Bezier -> Chebyshev 
    surf = bpy.data.objects['SurfPatch'+str(inappe)]
    cxyz = lbu.bezier_surface_to_chebyshev(surf)
    chebpoly.append(cxyz)

    # add mesh
    xyz = chebgrid2d(u, v, cxyz)
    verts, faces = lbu.tensor_product_mesh_vf(xyz[0], xyz[1], xyz[2])
    obj = lbu.pydata_to_mesh(verts, faces, edges=None, name='nappe'+str(inappe))
    lbe.set_smooth(obj)
    obj.hide_render = True
    nappes.append(obj)

    # make envelope of spheres
    cxyz_u, cxyz_v = lcheb.diff2(cxyz)
    xyz_u = chebgrid2d(u, v, cxyz_u)
    xyz_v = chebgrid2d(u, v, cxyz_v)
    
    xyzcgl = chebgrid2d(ucgl, ucgl, cxyz)
    rcgl = radius_function(xyzcgl[0], xyzcgl[1], xyzcgl[2])
    cr = lcheb.fcht(lcheb.fcht(rcgl).T).T
    cr_u, cr_v = lcheb.diff2(cr)
    r = chebgrid2d(u, v, cr)
    r_u = chebgrid2d(u, v, cr_u)
    r_v = chebgrid2d(u, v, cr_v)
    e = leos.two_parameter_EoS(xyz, xyz_u, xyz_v, r, r_u, r_v)
    #e = numpy.moveaxis(e[1], -1, 0)
    e = numpy.swapaxes(e[1].T, -1, -2)
    
    XYZR.append([xyz, r])
    E.append(e)

    verts, faces = lbu.tensor_product_mesh_vf(e[0], e[1], e[2])
    obj = lbu.pydata_to_mesh(verts, faces, edges=None, name='eos'+str(inappe))
    lbe.set_smooth(obj)
    nappes_eos.append(obj)
    
    mat = bpy.data.materials.new('mat_'+'eos'+str(inappe))
    mat.use_vertex_color_paint = True
    mat.use_shadeless = True
    obj.data.materials.append(mat)
    
    obj = lbe.unwrap_uv_tensor_product(obj, 0.5*(u+1), 0.5*(v+1))

    
for obj in nappes + nappes_eos:
    obj.layers[1] = True
    obj.layers[0] = False

LS = []
huge = 1e6
mnLS = 0.0
mxLS = 0.0
for inappe in range(2):
    jnappe = (inappe+1)%2
    F = huge*numpy.ones((m,n))
    for i in range(m):
        for j in range(n):
            xyzij = XYZR[jnappe][0][:,i,j]
            rij = XYZR[jnappe][1][i,j]
            Fij = (- rij**2)*numpy.ones((m,n))
            for k in range(3):
                Fij = Fij + numpy.power(E[inappe][k] - xyzij[k], 2)
            F = numpy.minimum(F, Fij)
    mnLS = min(mnLS, numpy.amin(F))
    mxLS = max(mxLS, numpy.amax(F))
    LS.append(F)

mLS = max(-mnLS,mxLS)

for inappe in range(2):
    F = LS[inappe]
    obj = nappes_eos[inappe]
    msh = obj.data
    msh.vertex_colors.new()
    vertexcolor = msh.vertex_colors[0].data
    for f in msh.polygons:
        v = f.vertices
        for p in range(f.loop_total):
            l = f.loop_start + p
            k = v[p]
            i = k%m
            j = (k-i)/m
            vertexcolor[l].color = div_colormap(F[i,j], -mLS, mLS)