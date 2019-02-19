import bpy
import sys
sys.path.append('/stck/bandrieu/Bureau/Python/mylibs/')
sys.path.append('/d/bandrieu/GitHub/These/memoire/figures/code/')
import my_lib as myl
import my_lib1 as mylb
import chebyshev as cheb
import chebyshev_lib as cheblib
import numpy as np
from numpy.polynomial.chebyshev import chebval, chebgrid2d, chebval2d

def norm2(a):
    return np.sqrt(np.sum(np.power(a,2)))

mr = 2
nr = mr
r0 = 0.4

bezier_poly = bpy.data.objects["SurfPatch"]
bpy.context.scene.objects.active = bezier_poly
bezier_poly.select = True

bpy.ops.object.transform_apply(location=True, rotation=True)

bezier_poly = bpy.data.curves["SurfPatch"]
bezier_poly.splines[0].use_endpoint_u = True
bezier_poly.splines[0].use_endpoint_v = True
mb = bezier_poly.splines[0].point_count_u #order_u
nb = bezier_poly.splines[0].point_count_v #order_v
B = np.zeros((mb,nb,3))
for l, p in enumerate(bezier_poly.splines[0].points):
    i = l%mb
    j = int((l-i)/mb)
    for k in range(3):
        B[i,j,k] = p.co[k]

M = cheb.B2Cmatrix(mb)
N = cheb.B2Cmatrix(nb)
C = np.zeros((mb,nb,3))
for k in range(3):
    C[:,:,k] = mylb.mymatmul(mylb.mymatmul(M, B[:,:,k]), N.T)
duC, dvC = cheblib.diff2(C)
    
u = np.linspace(-1.,1.,mr)
v = np.linspace(-1.,1.,nr)

xr = chebgrid2d(u, v, C[:,:,0])
yr = chebgrid2d(u, v, C[:,:,1])
zr = chebgrid2d(u, v, C[:,:,2])


for j in range(nr):
    for i in range(mr):
        bpy.ops.mesh.primitive_uv_sphere_add(
            location=[xr[i,j], yr[i,j], zr[i,j]],
            segments=16,
            ring_count=16,
            size=r0)

###########################      

###########################

Br = np.zeros((mr,nr))
for j in range(nr):
    for i in range(mr):
        k = j*mr + i
        if k == 0:
            name = "Sphere"
        else:
            name = "Sphere."+format(k,'03')
        bpy.data.objects[name].hide_render = True
        Br[i,j] = r0*(1+np.random.rand())*bpy.data.objects[name].scale[0]

M = cheb.B2Cmatrix(mr)
N = cheb.B2Cmatrix(nr)
Cr = mylb.mymatmul(mylb.mymatmul(M, Br), N.T)
duCr, dvCr = cheblib.diff2(Cr)


"""
m = 15
n = m
u = np.linspace(-1.,1.,m)
v = np.linspace(-1.,1.,n)
x = np.polynomial.chebyshev.chebgrid2d(u, v, C[:,:,0])
y = np.polynomial.chebyshev.chebgrid2d(u, v, C[:,:,1])
z = np.polynomial.chebyshev.chebgrid2d(u, v, C[:,:,2])
r = np.polynomial.chebyshev.chebgrid2d(u, v, Cr)

verts = []
faces = []
for j in range(n):
    for i in range(m):
        verts.append([x[i,j], y[i,j], z[i,j]])
        bpy.ops.mesh.primitive_uv_sphere_add(
            location=[x[i,j], y[i,j], z[i,j]],
            segments=16,
            ring_count=16,
            size=r[i,j])
        if i < m-1 and j < n-1:
            faces.append([j*m+i, j*m+i+1, (j+1)*m+i+1, (j+1)*m+i])
#create mesh and object
mesh = bpy.data.meshes.new("mypatch")
object = bpy.data.objects.new("mypatch",mesh)
#set mesh location
object.location = [0,0,0]
bpy.context.scene.objects.link(object)
#create mesh from python data
mesh.from_pydata(verts,[],faces)
mesh.update(calc_edges=True)
# show mesh as smooth
object.data.use_auto_smooth = 1
#bpy.ops.object.shade_smooth()
mypolys = mesh.polygons
for q in mypolys:
    q.use_smooth = True
"""
###########################


# add surface
m = 101
n = m
u = np.linspace(-1.,1.,m)
v = np.linspace(-1.,1.,n)
ns = chebgrid2d(u,v,Cr)
dur = chebgrid2d(u,v,duCr)
dvr = chebgrid2d(u,v,dvCr)
s = chebgrid2d(u,v,C)
dus = chebgrid2d(u,v,duC)
dvs = chebgrid2d(u,v,dvC)
verts = []
faces = []
for j in range(n):
    for i in range(m):
        verts.append([s[0,i,j], s[1,i,j], s[2,i,j]])
        if j < n-1 and i < m-1:
            faces.append([j*m + i, j*m + i + 1, (j+1)*m + i + 1, (j+1)*m + i])
mesh = bpy.data.meshes.new("surface")
object = bpy.data.objects.new("surface",mesh)

#set mesh location
object.location = [0,0,0]
bpy.context.scene.objects.link(object)

#create mesh from python data
mesh.from_pydata(verts,[],faces)
mesh.update(calc_edges=True)

# show mesh as smooth
object.data.use_auto_smooth = 1
mypolys = mesh.polygons
for q in mypolys:
    q.use_smooth = True

###########################
i = int(m/2)
j = int(n/2)

u0 = u[i]
v0 = v[j]
ns0 = chebval2d(u0,v0,Cr)

tau = 1.

r0 = tau*ns0

s0 = s[:,i,j]
r0 = tau*ns[i,j]
rs = 0.999*r0
bpy.ops.mesh.primitive_uv_sphere_add(
    location=s0, segments=100, ring_count=100, size=0.995*r0)
myl.setSmooth(bpy.context.object)

mat = bpy.data.materials.new("mat_sph")
mat.diffuse_color = [0.800, 0.494, 0.317]
mat.diffuse_intensity = 1
mat.specular_intensity = 0.5
mat.specular_hardness = 20.
mat.use_transparency = True
mat.raytrace_transparency.fresnel = 2.2#2.1
mat.alpha = 1
mat.emit = 1.#1.2
mat.use_raytrace = False
mat.use_shadows = False
mat.use_cast_buffer_shadows = False
myl.setMaterial(bpy.context.object, mat)

###########################
# add envelope +/-
mat = bpy.data.materials.new("mat_enve")
mat.diffuse_color = [0.527, 0.800, 0.213]
mat.diffuse_intensity = 1
mat.specular_intensity = 0.3#0
mat.specular_hardness = 20.
mat.use_transparency = True
mat.raytrace_transparency.fresnel = 2.0#2.7
mat.alpha = 0.5#0.68
mat.emit = 0.5#1.0
mat.use_raytrace = False
mat.use_shadows = False
mat.use_cast_buffer_shadows = False

dm = 50
dn = dm
nm = 2*dm + 1
for k in range(2):
    verts = []
    faces = []
    l = -1
    for jj in range(j-dn,j+dn+1):
        for ii in range(i-dm,i+dm+1):
            nor = np.cross(dus[:,ii,jj],dvs[:,ii,jj])
            nor = nor / norm2(nor)
            E = np.dot(dus[:,ii,jj], dus[:,ii,jj])
            F = np.dot(dus[:,ii,jj], dvs[:,ii,jj])
            G = np.dot(dvs[:,ii,jj], dvs[:,ii,jj])
            w = -((dur[ii,jj]*G - dvr[ii,jj]*F)*dus[:,ii,jj] +
              (dvr[ii,jj]*E - dur[ii,jj]*F)*dvs[:,ii,jj])
            w /= E*G - F**2
            sqrt1mtau2w2 = np.sqrt(1. - np.sum(np.power(w,2))*tau**2)
            r = tau*ns[ii,jj]
            vij = s[:,ii,jj] + r*(tau*w + sqrt1mtau2w2*nor*(-1.)**k)
            verts.append([vij[0], vij[1], vij[2]])
            l += 1
            if jj < j+dn and ii < i+dm:
                faces.append([l, l+1, l+nm+1, l+nm])
    mesh = bpy.data.meshes.new("envelope_"+str(k))
    object = bpy.data.objects.new("envelope_"+str(k),mesh)
    #set mesh location
    object.location = [0,0,0]
    bpy.context.scene.objects.link(object)
    #create mesh from python data
    mesh.from_pydata(verts,[],faces)
    mesh.update(calc_edges=True)
    # show mesh as smooth
    object.data.use_auto_smooth = 1
    mypolys = mesh.polygons
    for q in mypolys:
        q.use_smooth = True
    # assign material
    myl.setMaterial(object, mat)
