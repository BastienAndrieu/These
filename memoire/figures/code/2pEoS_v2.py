import bpy
import numpy as np
from numpy.polynomial.chebyshev import chebval, chebgrid2d, chebval2d
import sys
sys.path.append('/stck/bandrieu/Bureau/Python/mylibs/')
sys.path.append('/d/bandrieu/GitHub/These/memoire/figures/code/')
import chebyshev_lib as cheblib
import my_lib as myl
import my_lib1 as mylb
############################################
def norm2(a):
    return np.sqrt(np.sum(np.power(a,2)))

def fns(x,y,z):
    return 1. + 0.9*x + 0.5*y + 0.6*z - 0.1*x*y
    #return 1. + 0.5*np.sin(4.*y)*np.cos(6.*x) + x
    #return 1. + x - 0.2*np.power(x-1.,2) + 0.5*np.sin(7.*y) - 0.2*z
############################################

ps = mylb.readCoeffs2('/d/bandrieu/GitHub/These/memoire/figures/code/parametric_patch.cheb')
dups, dvps = cheblib.diff2(ps)

M = ps.shape[0]
N = ps.shape[1]

u = cheblib.cgl_nodes(M-1)
v = cheblib.cgl_nodes(N-1)

xc = chebgrid2d(u, v, ps[:,:,0])
yc = chebgrid2d(u, v, ps[:,:,1])
zc = chebgrid2d(u, v, ps[:,:,2])

nsc = fns(xc,yc,zc)
pns = cheblib.fcht(cheblib.fcht(nsc).T).T
dupns, dvpns = cheblib.diff2(pns)


u0 = 0.
v0 = 0.

ns0 = chebval2d(u0,v0,pns)
duns0 = chebval2d(u0,v0,dupns)
dvns0 = chebval2d(u0,v0,dvpns)

s0 = chebval2d(u0,v0,ps)
dus0 = chebval2d(u0,v0,dups)
dvs0 = chebval2d(u0,v0,dvps)

n0 = np.cross(dus0,dvs0)
n0 = n0 / norm2(n0)

E = np.dot(dus0, dus0)
F = np.dot(dus0, dvs0)
G = np.dot(dvs0, dvs0)

w0 = -((duns0*G - dvns0*F)*dus0 + (dvns0*E - duns0*F)*dvs0)
w0 /= E*G - F**2
tau = 0.2/norm2(w0)

r0 = tau*ns0

##################################################
myl.blankScene()

# add surface
m = 101
n = m
u = np.linspace(-1.,1.,m)
v = np.linspace(-1.,1.,n)
ns = chebgrid2d(u,v,pns)
duns = chebgrid2d(u,v,dupns)
dvns = chebgrid2d(u,v,dvpns)
s = chebgrid2d(u,v,ps)
dus = chebgrid2d(u,v,dups)
dvs = chebgrid2d(u,v,dvps)
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



##########################################
# add sphere
i = int(m/2)
j = int(n/2)
s0 = s[:,i,j]
r0 = tau*ns[i,j]
rs = 0.999*r0

bpy.ops.mesh.primitive_uv_sphere_add(
    location=s0, segments=100, ring_count=100, size=0.995*r0)
myl.setSmooth(bpy.context.object)

##########################################
# add envelope +/-
"""
mat = bpy.data.materials.new("mat_enve")
mat.diffuse_color = [0.527, 0.800, 0.213]
mat.diffuse_intensity = 1
mat.specular_intensity = 1#0
mat.specular_hardness = 30
mat.use_transparency = True
mat.raytrace_transparency.fresnel = 2.0#2.7
mat.alpha = 1.0#0.68
mat.emit = 1.0
"""
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
            w = -((duns[ii,jj]*G - dvns[ii,jj]*F)*dus[:,ii,jj] +
              (dvns[ii,jj]*E - duns[ii,jj]*F)*dvs[:,ii,jj])
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
    #myl.setMaterial(object, mat)
##########################################
