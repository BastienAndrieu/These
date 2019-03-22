import bpy
import sys
sys.path.append('/stck/bandrieu/Bureau/Python/mylibs/')
sys.path.append('/d/bandrieu/GitHub/These/memoire/figures/code/')
import my_lib as myl
import my_lib1 as mylb
import chebyshev as cheb
import chebyshev_lib as cheblib
import lib_fig
import numpy
from numpy.polynomial.chebyshev import chebval, chebval2d, chebgrid2d
###########################################################

pth = '/d/bandrieu/GitHub/These/memoire/figures/data/'
nuvcells = 6
nfaceperuvcell = 20

###########################################################
## Convert Bezier control points to Chebyshev coefficients
bpy.data.objects["SurfPatch"].hide = True
bezier_poly = bpy.data.curves["SurfPatch"]
bezier_poly.splines[0].use_endpoint_u = True
bezier_poly.splines[0].use_endpoint_v = True
mb = bezier_poly.splines[0].point_count_u #order_u
nb = bezier_poly.splines[0].point_count_v #order_v
B = numpy.zeros((mb,nb,3))
for l, p in enumerate(bezier_poly.splines[0].points):
        i = l%mb
        j = int((l-i)/mb)
        for k in range(3):
            B[i,j,k] = p.co[k]
M = cheb.B2Cmatrix(mb)
N = cheb.B2Cmatrix(nb)
c = numpy.zeros((mb,nb,3))
for k in range(3):
    c[:,:,k] = mylb.mymatmul(mylb.mymatmul(M, B[:,:,k]), N.T)
    # flip u and v
    c[:,:,k] = c[:,:,k].T

duc, dvc = cheblib.diff2(c)

## Add smooth surface patch
m = nfaceperuvcell*nuvcells + 1
#patch = myl.addPatch(c, m, [1,1,1], 1)
u = numpy.linspace(-1.0,1.0,m)
patch = myl.addTensorProductPatch(chebgrid2d(u, u, c[:,:,0]),
                                  chebgrid2d(u, u, c[:,:,1]),
                                  chebgrid2d(u, u, c[:,:,2]),
                                  name="patch",
                                  location=[0,0,0],
                                  smooth=True,
                                  color=[1,1,1],
                                  alpha=1)


## Add point, tangents and normal
tchkvec = 5.e-3
lennor = 1.0

#uv = -1.0 + 2.*numpy.array([6,2])/9.
uv = -1.0 + 2.*numpy.array([4,1])/6.
s  = chebval2d(uv[0], uv[1], c)
su = chebval2d(uv[0], uv[1], duc)
sv = chebval2d(uv[0], uv[1], dvc)
sn = numpy.cross(su,sv)
scl = lennor / numpy.sqrt(numpy.sum(numpy.power(sn,2)))


    
## Add ds
duv = numpy.ones(2)/3.0#[0.3,0.3]#[0.2,0.1]
ds = duv[0]*su + duv[1]*sv

myl.addPolyline(p=numpy.vstack((s,s+ds)), clr=[0,0,0], thickness=tchkvec)


f = open(pth + 'fig_diffgeom_point_on_surface.dat', 'w')

f.write(str(uv[0]) + ', ' + str(uv[1]) + '\n')

qx, qy = lib_fig.convert_3d_to_2d_coords(s)
f.write(str(qx) + ', ' + str(qy) + '\n')

f.write(str(duv[0]) + ', ' + str(duv[1]) + '\n')

clvec = numpy.array([[1,0,0],[0,0,1],[0,0,0],[0,1,0]])
for ivec, vec in enumerate([duv[0]*su, duv[1]*sv, scl*sn, ds]):
    dest = s+vec
    qx, qy = lib_fig.convert_3d_to_2d_coords(dest)
    f.write(str(qx) + ', ' + str(qy) + '\n')
    myl.addPolyline(p=numpy.vstack((s,dest)), clr=clvec[ivec], thickness=tchkvec)
f.close()

## Add curve
cpsi = mylb.readCoeffs1(pth + 'fig_diffgeom_cpsi.cheb')
dwcpsi = cheblib.diff(cpsi)
uv = numpy.loadtxt(pth + 'fig_diffgeom_uvcurve.dat')

g = chebval2d(uv[:,0], uv[:,1], c).T
myl.addPolyline(p=g, clr=[0,0,0], thickness=tchkvec)
f = open(pth + 'fig_diffgeom_xyzcurve.dat', 'w')
for x in g:
    qx, qy = lib_fig.convert_3d_to_2d_coords(x)
    f.write(str(qx) + ' ' + str(qy) + '\n')
f.close()


w = -0.2
uv   = chebval(w, cpsi)
dwuv = chebval(w, dwcpsi)
print(uv, dwuv)

xyz   = chebval2d(uv[0], uv[1], c)
duxyz = chebval2d(uv[0], uv[1], duc)
dvxyz = chebval2d(uv[0], uv[1], dvc)
dwxyz = dwuv[0]*duxyz + dwuv[1]*dvxyz

myl.addPolyline(p=numpy.vstack((xyz,xyz + dwxyz)), clr=[1,1,0], thickness=tchkvec)



f = open(pth + 'fig_diffgeom_point_on_curve.dat', 'w')
f.write(str(w) + ', ' + str(0.0) + '\n')
f.write(str(uv[0]) + ', ' + str(uv[1]) + '\n')
f.write(str(dwuv[0]) + ', ' + str(dwuv[1]) + '\n')

for q in [xyz, xyz + dwxyz]:
    qx, qy = lib_fig.convert_3d_to_2d_coords(q)
    f.write(str(qx) + ', ' + str(qy) + '\n')
f.close()




### Edit vertex color to make a checker board texture
mesh = bpy.data.meshes["patch"]
mesh.vertex_colors.new()

vertexcolor = mesh.vertex_colors[0].data


color1 = numpy.zeros(3)
color2 = numpy.ones(3)

for j in range(m-1):
    jcell = int(j/nfaceperuvcell)
    for i in range(m-1):
        icell = int(i/nfaceperuvcell)
        k = (m-1)*j + i
        if (icell+jcell)%2 == 0:
            colorf = color1
        else:
            colorf = color2
        for l in range(4*k,4*k+4):
            vertexcolor[l].color = colorf
        
bpy.data.materials['mat_patch'].use_vertex_color_paint = True
