import bpy
import bpy_extras
from mathutils import Vector
import numpy
from numpy import *
from numpy.polynomial.chebyshev import chebgrid2d, chebval2d

import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python/')
import lib_blender_edit as lbe
import lib_blender_util as lbu
import lib_blender_figures as lbf
import lib_chebyshev as lcheb

############################################
def norm2(a):
    return sqrt(sum(a**2))
############################################
def normal_speed_function(x, y, z):
    return 1 + 0.5*sin(4*y)*cos(6*x) + x
############################################


pthin = '/d/bandrieu/GitHub/These/memoire/figures/data/EdS_propre_carreau/'

#################################################
# CHECKER TEXTURE
imgchecker = bpy.data.images.load(filepath='/d/bandrieu/GitHub/These/memoire/figures/code/BRep/checker.png')
texchecker = bpy.data.textures.new('texture_checker', 'IMAGE')
texchecker.image = imgchecker
#################################################


#################################################
## Set Scene
scene = bpy.context.scene
cam = scene.camera
lbu.clear_scene(meshes=True, lamps=True, cameras=False)

resx = 1400
resy = resx*5./8.
lbu.set_scene(
    resolution_x=resx,
    resolution_y=resy,
    resolution_percentage=100,
    alpha_mode='SKY',
    horizon_color=[1,1,1],
    use_environment_light=False
)
light_settings = scene.world.light_settings
light_settings.use_ambient_occlusion = True
light_settings.ao_factor = 0.5
light_settings.ao_blend_type = 'MULTIPLY'

## Set Camera
cam.location = (5.20956, -3.24223, 2.99055)
cam.rotation_euler = radians([63.2, 0.598, 56.5])
cam.data.lens_unit = "FOV"
cam.data.angle = radians(11.0)

## Set Lighting
fl = 2.0
lamp = lbu.add_point_light(
    name="lamp",
    energy=2.2,
    shadow_method='NOSHADOW',
    location=Vector((-0.46846,-3.60067,4.38))*fl
)
#################################################


#################################################
ps = lcheb.read_polynomial2(pthin + 'surface.cheb')
ps[:,:,2] = 1.3*ps[:,:,2]
angle = radians(6.0)
cosangle = cos(angle)
sinangle = sin(angle)
X = cosangle*ps[:,:,0] + sinangle*ps[:,:,2]
Z = -sinangle*ps[:,:,0] + cosangle*ps[:,:,2]
ps[:,:,0] = X
ps[:,:,2] = Z
dups, dvps = lcheb.diff2(ps)

M = ps.shape[0]
N = ps.shape[1]

u = lcheb.cgl_nodes(M-1)
v = lcheb.cgl_nodes(N-1)

xyzc = chebgrid2d(u, v, ps)

nsc = normal_speed_function(xyzc[0], xyzc[1], xyzc[2])
pns = lcheb.fcht(lcheb.fcht(nsc).T).T
dupns, dvpns = lcheb.diff2(pns)

u0 = 0.1
v0 = 0.25

ns0 = chebval2d(u0,v0,pns)
duns0 = chebval2d(u0,v0,dupns)
dvns0 = chebval2d(u0,v0,dvpns)

s0 = chebval2d(u0,v0,ps)
#bpy.ops.object.empty_add(location=s0)
dus0 = chebval2d(u0,v0,dups)
dvs0 = chebval2d(u0,v0,dvps)

n0 = cross(dus0,dvs0)
n0 = n0 / norm2(n0)

E = dot(dus0, dus0)
F = dot(dus0, dvs0)
G = dot(dvs0, dvs0)

w0 = ((dvns0*F - duns0*G)*dus0 + (duns0*F - dvns0*E)*dvs0)
w0 /= E*G - F**2
tau = 0.2/norm2(w0)

r0 = tau*ns0
#################################################


#################################################
# ADD SURFACE
m = 101
n = m
u = linspace(-0.65, 0.92, m)
v = linspace(-0.64, 1.1, n)

ns = chebgrid2d(u,v,pns)
duns = chebgrid2d(u,v,dupns)
dvns = chebgrid2d(u,v,dvpns)
s = chebgrid2d(u,v,ps)
dus = chebgrid2d(u,v,dups)
dvs = chebgrid2d(u,v,dvps)

verts, faces = lbu.tensor_product_mesh_vf(s[0], s[1], s[2])
surf = lbu.pydata_to_mesh(
    verts,
    faces,
    edges=None,
    name='carreau'
)

lbe.set_smooth(surf)

bpy.ops.object.select_all(action='DESELECT')
scene.objects.active = surf
surf.select = True
bpy.ops.object.mode_set(mode='EDIT')
print('   unwrap UVs...')
bpy.ops.uv.unwrap(
    method='ANGLE_BASED',
    fill_holes=True,
    correct_aspect=True,
    use_subsurf_data=False,
    margin=0.001
)
bpy.ops.object.mode_set(mode='OBJECT')
uvlayer = surf.data.uv_layers.active
print('   edit UVs...')
uu = linspace(0,1,m)
vv = linspace(0,1,n)
for j in range(n-1):
    for i in range(m-1):
        k = i + j*(m-1)
        f = surf.data.polygons[k]
        for l in [0,3]:
            uvlayer.data[f.loop_start + l].uv[0] = uu[i]
        for l in [1,2]:
            uvlayer.data[f.loop_start + l].uv[0] = uu[i+1]
        for l in [0,1]:
            uvlayer.data[f.loop_start + l].uv[1] = vv[j]
        for l in [2,3]:
            uvlayer.data[f.loop_start + l].uv[1] = vv[j+1]

mat_surf = bpy.data.materials.new('mat_carreau')
mat_surf.diffuse_color = (0.330, 0.413, 0.500)
mat_surf.diffuse_intensity = 1
mat_surf.emit = 0.8
mat_surf.specular_intensity = 1
mat_surf.specular_hardness = 20.0
mat_surf.use_transparency = True
mat_surf.raytrace_transparency.fresnel = 2.

slot = mat_surf.texture_slots.add()
slot.texture = texchecker
slot.texture_coords = 'UV'
slot.diffuse_color_factor = 0.09

surf.data.materials.append(mat_surf)
#################################################


#################################################
# SPHERE DATA & MATERIAL
i = int(m/2)
j = int(n/2)
s0 = s[:,i,j]
r0 = tau*ns[i,j]
rs = 0.999*r0

mat_sphere = bpy.data.materials.new('mat_sphere')
mat_sphere.diffuse_color = (0.800, 0.494, 0.317)
mat_sphere.diffuse_intensity = 1
mat_sphere.specular_intensity = 1
mat_sphere.specular_hardness = 30.
mat_sphere.use_transparency = True
mat_sphere.raytrace_transparency.fresnel = 2.2
mat_sphere.alpha = 1
mat_sphere.emit = 1.0
#################################################


#################################################
# POINTS ON ENVELOPE
ns0 = ns[i,j]
duns0 = duns[i,j]
dvns0 = dvns[i,j]

dus0 = dus[:,i,j]
dvs0 = dvs[:,i,j]

n0 = cross(dus0,dvs0)
n0 = n0 / norm2(n0)

E = dot(dus0, dus0)
F = dot(dus0, dvs0)
G = dot(dvs0, dvs0)

w0 = ((dvns0*F - duns0*G)*dus0 + (duns0*F - dvns0*E)*dvs0)
w0 /= E*G - F**2

r0 = tau*ns0

nn = sqrt(1 - sum(w0**2)*tau**2)

eospoints = []
f = open(pthin + 'envelope_points.dat', 'w')
for k in range(2):
    e0k = s0 + r0*(tau*w0 + nn*n0*(-1)**(k+1))
    #bpy.ops.object.empty_add(location=e0k)
    eospoints.append(e0k)
    vx, vy = lbf.convert_3d_to_2d_coords(e0k, normalize=True)
    f.write('%s, %s\n' % (vx, vy))
f.close()

# NORMALS TO ENVELOPE
veclw = 0.14
f = open(pthin + 'envelope_normals.dat', 'w')
for k in range(2):
    vec = eospoints[k] - s0
    vec = vec/r0
    print(norm2(vec))
    vx, vy = lbf.convert_3d_to_2d_coords(eospoints[k] + veclw*vec, normalize=True)
    bpy.ops.object.empty_add(location=eospoints[k] + veclw*vec)
    f.write('%s, %s\n' % (vx, vy))
f.close()
#################################################

#################################################
# DUMMY MATERIAL
mat_dummy = bpy.data.materials.new('mat_dummy')
mat_dummy.diffuse_color = (0.800, 0.494, 0.317)
mat_dummy.specular_intensity = 0
mat_dummy.use_transparency = True
mat_dummy.alpha = 0
mat_dummy.use_raytrace = False
mat_dummy.use_shadows = False
mat_dummy.use_cast_buffer_shadows = True
#################################################


#################################################
# ADD CHARACTERISTIC CIRCLES
bu0 = ones(3, dtype=float)

# /u
dg0 = dus[:,i,j]
normdg = norm2(dg0)
dr0 = tau*duns[i,j]
sina = dr0/normdg
occ = s0 - r0*sina*dg0/normdg
rcc = r0*sqrt(1. - sina**2)
bw = dg0/normdg
bu = bu0 - dot(bu0,bw)*bw
bu = bu / norm2(bu)
bv = cross(bw,bu)

verts = []
faces = []
nc = 100
tc = pi*linspace(-1.,1.,nc)
for k in range(nc):
    vk = occ + rcc*(cos(tc[k])*bu + sin(tc[k])*bv)
    verts.append([vk[0], vk[1], vk[2]])
    faces.append([k, (k+1)%nc, nc])
verts.append([occ[0], occ[1], occ[2]])

circle_u = lbu.pydata_to_mesh(
    verts,
    faces,
    edges=None,
    name='cercle_u'
)
circle_u.data.materials.append(mat_dummy)


# /v
dg0 = dvs[:,i,j]
normdg = norm2(dg0)
dr0 = tau*dvns[i,j]
sina = dr0/normdg
occ = s0 - r0*sina*dg0/normdg
rcc = r0*sqrt(1. - sina**2)
bw = dg0/normdg
bu = bu0 - dot(bu0,bw)*bw
bu = bu / norm2(bu)
bv = cross(bw,bu)

verts = []
faces = []
nc = 100
tc = pi*linspace(-1.,1.,nc)
for k in range(nc):
    vk = occ + rcc*(cos(tc[k])*bu + sin(tc[k])*bv)
    verts.append([vk[0], vk[1], vk[2]])
    faces.append([k, (k+1)%nc, nc])
verts.append([occ[0], occ[1], occ[2]])

circle_v = lbu.pydata_to_mesh(
    verts,
    faces,
    edges=None,
    name='cercle_v'
)
circle_v.data.materials.append(mat_dummy)
#################################################



#################################################
# TANGENT AND NORMAL VECTORS
factng = 1.5
tngu = factng*dvs[:,i,j]
tngv = factng*dus[:,i,j]
nor0 = cross(tngu, tngv)
if nor0[2] < 0: nor0 = -nor0
nor0 = nor0/norm2(nor0)
print(norm2(nor0))

f = open(pthin + 'surface_point.dat', 'w')
vx, vy = lbf.convert_3d_to_2d_coords(s0, normalize=True)
f.write('%s, %s' % (vx, vy))
f.close()

f = open(pthin + 'surface_vectors.dat', 'w')
for vec in [tngu, tngv, nor0]:
    vx, vy = lbf.convert_3d_to_2d_coords(s0 + veclw*vec, normalize=True)
    #bpy.ops.object.empty_add(location=s0 + veclw*vec)
    f.write('%s, %s\n' % (vx, vy))
f.close()
#################################################



#################################################
# SPHERE-SURFACE INTERSECTION
ni = 100
ti = linspace(0,2*pi,ni+1)
ti = ti + ti[0]

ruv = 0.5
uvi = vstack([u[i] + ruv*cos(ti[:ni]), v[j] + ruv*sin(ti[:ni])]).T
si = zeros((ni,3))

tol3 = 1.e-6
tol2 = 1.e-7
itmax = 20
r2 = r0**2
for k in range(ni):
    if abs(uvi[k,0] - u[i]) < abs(uvi[k,1] - v[j]):
        mode = 1
        b = (uvi[k,0] - u[i])/(uvi[k,1] - v[j])
    else:
        mode = 2
        a = (uvi[k,1] - v[j])/(uvi[k,0] - u[i])
    conv = False
    print('pt #',k+1,'/',ni)
    for it in range(itmax):
        print('   it.#',it+1)
        q = chebval2d(uvi[k,0], uvi[k,1], ps) - s0
        su = chebval2d(uvi[k,0], uvi[k,1], dups)
        sv = chebval2d(uvi[k,0], uvi[k,1], dvps)
        if mode == 1:
            dq = b*su + sv
        else:
            dq = su + a*sv
        qtq = sum(power(q,2))
        res = abs(qtq - r2)
        print('      res =',res)
        if res < tol3:
            conv = True
            break
        dqtq = dot(dq,q)
        ds = 0.5*(r2 - qtq)/dqtq
        #print('      ds =',ds)
        if abs(ds) < tol2:
            break
        if mode == 1:
            uvi[k,0] += b*ds
            uvi[k,1] += ds
        else:
            uvi[k,0] += ds
            uvi[k,1] += a*ds
        #print('')
    if conv:
        si[k] = chebval2d(uvi[k,0], uvi[k,1], ps)
    else:
        #print("pt #",k,", res=",res)
        print('failed to converge :(')
        si[k] = s0

verts = []
faces = []
epsz = 0.#1.e-3
for k in range(ni):
    vk = [si[k,0], si[k,1], si[k,2]+epsz]
    verts.append([vk[0], vk[1], vk[2]])
    faces.append([k, (k+1)%ni, ni])
verts.append([s0[0], s0[1], s0[2]])

intersection = lbu.pydata_to_mesh(
    verts,
    faces,
    edges=None,
    name='intersection'
)
intersection.data.materials.append(mat_dummy)
#################################################



#################################################
# ADD TOP & BOTTOM HEMISPHERES
# base (intersection)
thetai = arctan2(si[:,1]-s0[1], si[:,0]-s0[0])
phii = arctan2(si[:,2]-s0[2], hypot(si[:,0]-s0[0], si[:,1]-s0[1]))

nr = 50
hs = zeros((ni,nr,3))
hemispheres = []
for k in range(2):
    sgnhalfpi = 0.5*pi*(-1.)**k
    for jj in range(nr):
        fj = float(jj)/float(nr+1)
        for ii in range(ni):
            ph = (1. - fj)*phii[ii] + fj*sgnhalfpi
            cph = cos(ph)
            th = thetai[ii]
            hs[ii,jj] = s0 + rs*array([cos(th)*cph, sin(th)*cph, sin(ph)])
    verts = []
    faces = []
    for jj in range(nr):
        for ii in range(ni):
            verts.append([hs[ii,jj,0], hs[ii,jj,1], hs[ii,jj,2]])
            if jj < nr-1:
                if k == 0:
                    faces.append([jj*ni + ii, jj*ni + (ii+1)%ni, (jj+1)*ni + (ii+1)%ni, (jj+1)*ni + ii])
                else:
                    faces.append([jj*ni + ii, (jj+1)*ni + ii, (jj+1)*ni + (ii+1)%ni, jj*ni + (ii+1)%ni])
                    
    verts.append([s0[0], s0[1], s0[2]+rs*(-1.)**k])
    lastv = len(verts)-1
    for ii in range(ni):
        if k == 0:
            faces.append([(nr-1)*ni + ii, (nr-1)*ni + (ii+1)%ni, lastv])
        else:
            faces.append([(nr-1)*ni + ii, lastv, (nr-1)*ni + (ii+1)%ni])

    hemisphere = lbu.pydata_to_mesh(
        verts,
        faces,
        edges=None,
        name='hemisphere_'+str(k)
    )
    lbe.set_smooth(hemisphere)
    hemisphere.data.materials.append(mat_sphere)

    hemispheres.append(hemisphere)
#################################################



#################################################
# MAKE GROUPS
bpy.ops.object.select_all(action='DESELECT')
circle_u.select = True
bpy.ops.group.create(name='group_cercle_u')

bpy.ops.object.select_all(action='DESELECT')
circle_v.select = True
bpy.ops.group.create(name='group_cercle_v')

bpy.ops.object.select_all(action='DESELECT')
surf.select = True
bpy.ops.group.create(name='group_carreau')

bpy.ops.object.select_all(action='DESELECT')
intersection.select = True
bpy.ops.group.create(name='group_intersection')

bpy.ops.object.select_all(action='DESELECT')
hemispheres[0].select = True
bpy.ops.group.create(name='group_hemisphere_0')

bpy.ops.object.select_all(action='DESELECT')
hemispheres[1].select = True
bpy.ops.group.create(name='group_hemisphere_1')
#################################################



################################################
# FREESTYLE SETTINGS
scene.render.use_freestyle = True
freestyle = scene.render.layers.active.freestyle_settings
freestyle.use_smoothness = True

cc_lw = 3.5
cc_dl = 10
cc_gl = 7
ccu_cl = (0,0,1)
ccv_cl = (1,0,0)

sp_cl = [0.799,0.396,0.159]
sp_lw = 2.5
sp_dl = 12
sp_gl = sp_dl


# Circle u, visible
lineset = freestyle.linesets['LineSet']
lineset.name = 'lineset_cercle_u_vis'
lineset.select_silhouette = False
lineset.select_border = True
lineset.visibility = 'VISIBLE'
linestyle = bpy.data.linestyles['LineStyle']
linestyle.name = 'linestyle_cercle_u_vis'
linestyle.color = ccu_cl
linestyle.thickness = cc_lw
lineset.group = bpy.data.groups['group_cercle_u']


# Circle v, visible
bpy.ops.scene.freestyle_lineset_add()
lineset = freestyle.linesets['LineSet 2']
lineset.name = 'lineset_cercle_v_vis'
lineset.select_silhouette = False
lineset.select_border = True
lineset.visibility = 'VISIBLE'
linestyle = bpy.data.linestyles['LineStyle']
linestyle.name = 'linestyle_cercle_v_vis'
linestyle.color = ccv_cl
linestyle.thickness = cc_lw
lineset.group = bpy.data.groups['group_cercle_v']


# Hemisphere 0 (top)
bpy.ops.scene.freestyle_lineset_add()
lineset = freestyle.linesets['LineSet 3']
lineset.name = 'lineset_hemisphere_0'
lineset.select_silhouette = True
lineset.select_border = False
lineset.visibility = 'RANGE'
linestyle = bpy.data.linestyles['LineStyle']
linestyle.name = 'linestyle_hemisphere_0'
linestyle.color = sp_cl
linestyle.thickness = sp_lw
lineset.group = bpy.data.groups['group_hemisphere_0']


# Hemisphere 1 (bottom)
bpy.ops.scene.freestyle_lineset_add()
lineset = freestyle.linesets['LineSet 4']
lineset.name = 'lineset_hemisphere_1'
lineset.select_silhouette = True
lineset.select_border = False
lineset.visibility = 'RANGE'
linestyle = bpy.data.linestyles['LineStyle']
linestyle.name = 'linestyle_hemisphere_1'
linestyle.color = sp_cl
linestyle.thickness = sp_lw
linestyle.use_dashed_line = True
linestyle.dash1 = sp_dl
linestyle.gap1 = sp_gl
lineset.group = bpy.data.groups['group_hemisphere_1']


# Circle u, hidden
bpy.ops.scene.freestyle_lineset_add()
lineset = freestyle.linesets['LineSet 5']
lineset.name = 'lineset_cercle_u_hid'
lineset.select_silhouette = False
lineset.select_border = True
lineset.visibility = 'HIDDEN'
linestyle = bpy.data.linestyles['LineStyle']
linestyle.name = 'linestyle_cercle_u_hid'
linestyle.color = ccu_cl
linestyle.thickness = cc_lw
linestyle.use_chaining = True
linestyle.use_dashed_line = True
linestyle.dash1 = cc_dl
linestyle.gap1 = cc_gl
lineset.group = bpy.data.groups['group_cercle_u']


# Circle v, hidden
bpy.ops.scene.freestyle_lineset_add()
lineset = freestyle.linesets['LineSet 6']
lineset.name = 'lineset_cercle_v_hid'
lineset.select_silhouette = False
lineset.select_border = True
lineset.visibility = 'HIDDEN'
linestyle = bpy.data.linestyles['LineStyle']
linestyle.name = 'linestyle_cercle_v_hid'
linestyle.color = ccv_cl
linestyle.thickness = cc_lw
linestyle.use_chaining = True
linestyle.use_dashed_line = True
linestyle.dash1 = cc_dl
linestyle.gap1 = cc_gl
lineset.group = bpy.data.groups['group_cercle_v']


# Surface
bpy.ops.scene.freestyle_lineset_add()
lineset = freestyle.linesets['LineSet 7']
lineset.name = 'lineset_carreau'
lineset.select_silhouette = False
lineset.select_border = True
lineset.visibility = 'VISIBLE'
linestyle = bpy.data.linestyles['LineStyle']
linestyle.name = 'linestyle_carreau'
linestyle.color = (0.330,0.413,0.500)
linestyle.thickness = 1.5
lineset.group = bpy.data.groups['group_carreau']


for lineset in freestyle.linesets:
    lineset.select_by_group = True
    lineset.select_crease = False
    
for linestyle in bpy.data.linestyles:
    linestyle.caps = 'ROUND'
    linestyle.geometry_modifiers["Sampling"].sampling = 0.1
################################################

