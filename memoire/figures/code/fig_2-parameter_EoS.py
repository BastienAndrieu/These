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
    #return 1. + 0.9*x + 0.5*y + 0.6*z - 0.1*x*y
    return 1. + 0.5*np.sin(4.*y)*np.cos(6.*x) + x
    #return 1. + x - 0.2*np.power(x-1.,2) + 0.5*np.sin(7.*y) - 0.2*z
############################################

ps = mylb.readCoeffs2('/d/bandrieu/GitHub/These/memoire/figures/code/fig_2-parameter_EoS_surf.cheb')
ps[:,:,2] = 1.3*ps[:,:,2]
angle = 6.*np.pi/180.
cosangle = np.cos(angle)
sinangle = np.sin(angle)
X = cosangle*ps[:,:,0] + sinangle*ps[:,:,2]
Z = -sinangle*ps[:,:,0] + cosangle*ps[:,:,2]
ps[:,:,0] = X
ps[:,:,2] = Z
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




u0 = 0.1
v0 = 0.25

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

myl.blankScene()

# add surface
m = 101
n = m
u = np.linspace(-0.65,0.92,m)#np.linspace(-0.67,0.9,m)#np.linspace(-1.,1.,m)
v = np.linspace(-0.64,1.1,n)#np.linspace(-0.62,1.,n)#np.linspace(-1.,1.,n)
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
    
# assign material
mat = myl.surfaceMaterial("mat_surface", [0.330,0.413,0.500], 1, 0.8)
mat.specular_intensity = 1.0
mat.specular_hardness = 20.0
mat.use_transparency = True
mat.raytrace_transparency.fresnel = 2.
myl.setMaterial(object, mat)


# add dummy surface #########
mesh = bpy.data.meshes.new("surface_dummy")
object = bpy.data.objects.new("surface_dummy",mesh)
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
mat = myl.surfaceMaterial("mat_surface_dummy",np.ones(3),0,0)
mat.specular_intensity = 0
mat.use_raytrace = False
mat.use_shadows = False
mat.use_cast_buffer_shadows = False
myl.setMaterial(object, mat)



###################################
# add sphere
i = int(m/2)
j = int(n/2)
s0 = s[:,i,j]
r0 = tau*ns[i,j]
rs = 0.999*r0#0.995*r0

mat = bpy.data.materials.new("mat_sph")
mat.diffuse_color = [0.800, 0.494, 0.317]
mat.diffuse_intensity = 1
mat.specular_intensity = 1
mat.specular_hardness = 30.
mat.use_transparency = True
mat.raytrace_transparency.fresnel = 2.2#2.1
mat.alpha = 1
mat.emit = 1.#1.2
"""
bpy.ops.mesh.primitive_uv_sphere_add(
    location=s0, segments=100, ring_count=100, size=0.995*r0)
myl.setSmooth(bpy.context.object)
myl.setMaterial(bpy.context.object, mat)
"""
###################################
# add characterisitc circles
bu0 = np.array([1.,1.,1.])

# /u
dg0 = dus[:,i,j]
normdg = norm2(dg0)
dr0 = tau*duns[i,j]
sina = dr0/normdg
occ = s0 - r0*sina*dg0/normdg
rcc = r0*np.sqrt(1. - sina**2)
bw = dg0/normdg
bu = bu0 - np.dot(bu0,bw)*bw
bu = bu / norm2(bu)
bv = np.cross(bw,bu)

verts = []
faces = []
nc = 100
tc = np.pi*np.linspace(-1.,1.,nc)
for k in range(nc):
    vk = occ + rcc*(np.cos(tc[k])*bu + np.sin(tc[k])*bv)
    verts.append([vk[0], vk[1], vk[2]])
    faces.append([k, (k+1)%nc, nc])
verts.append([occ[0], occ[1], occ[2]])
#create mesh and object
mesh = bpy.data.meshes.new("ccircle_u")
object = bpy.data.objects.new("ccircle_u",mesh)

#set mesh location
object.location = bpy.context.scene.cursor_location
bpy.context.scene.objects.link(object)

#create mesh from python data
mesh.from_pydata(verts,[],faces)
myl.setMaterial(object, bpy.data.materials["mat_surface_dummy"])

# /v
dg0 = dvs[:,i,j]
normdg = norm2(dg0)
dr0 = tau*dvns[i,j]
sina = dr0/normdg
occ = s0 - r0*sina*dg0/normdg
rcc = r0*np.sqrt(1. - sina**2)
bw = dg0/normdg
bu = bu0 - np.dot(bu0,bw)*bw
bu = bu / norm2(bu)
bv = np.cross(bw,bu)

verts = []
faces = []
for k in range(nc):
    vk = occ + rcc*(np.cos(tc[k])*bu + np.sin(tc[k])*bv)
    verts.append([vk[0], vk[1], vk[2]])
    faces.append([k, (k+1)%nc, nc])
verts.append([occ[0], occ[1], occ[2]])
#create mesh and object
mesh = bpy.data.meshes.new("ccircle_v")
object = bpy.data.objects.new("ccircle_v",mesh)

#set mesh location
object.location = bpy.context.scene.cursor_location
bpy.context.scene.objects.link(object)

#create mesh from python data
mesh.from_pydata(verts,[],faces)
myl.setMaterial(object, bpy.data.materials["mat_surface_dummy"])

###################################
# add tangents and normal
tngu = np.vstack([s0, s0-dus[:,i,j]])
tngv = np.vstack([s0, s0+dvs[:,i,j]])
nor0 = -np.cross(dus[:,i,j], dvs[:,i,j])

ccu_cl = [0.,0.,1.]
ccv_cl = [1.,0.,0.]

veclw = 1.5e-3

myl.addPolyline(tngu, ccu_cl, veclw, 0)
myl.addPolyline(tngv, ccv_cl, veclw, 0)
myl.addPolyline(np.vstack([s0, s0+nor0]), [0,0,0], veclw, 0)
###################################

###################################
"""
# add iso-u curve
verts = []
for k in range(n):
    verts.append([s[0,i,k], s[1,i,k], s[2,i,k]])
myl.addPolyline(verts, ccv_cl, veclw, 0)

# add iso-v curve
verts = []
for k in range(n):
    verts.append([s[0,k,j], s[1,k,j], s[2,k,j]])
myl.addPolyline(verts, ccu_cl, veclw, 0)
"""
###################################


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
    myl.setMaterial(object, mat)
"""
##########################################


##########################################
# surface-sphere intersection
ni = 100
ti = np.linspace(0.,2.*np.pi,ni+1)
ti = ti + ti[0]

ruv = 0.5
uvi = np.vstack([u[i] + ruv*np.cos(ti[:ni]), v[j] + ruv*np.sin(ti[:ni])]).T
si = np.zeros((ni,3))

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
    #print('pt #',k+1,'/',ni)
    for it in range(itmax):
        #print('   it.#',it+1)
        q = chebval2d(uvi[k,0], uvi[k,1], ps) - s0
        su = chebval2d(uvi[k,0], uvi[k,1], dups)
        sv = chebval2d(uvi[k,0], uvi[k,1], dvps)
        if mode == 1:
            dq = b*su + sv
        else:
            dq = su + a*sv
        qtq = np.sum(np.power(q,2))
        res = abs(qtq - r2)
        #print('      res =',res)
        if res < tol3:
            conv = True
            break
        dqtq = np.dot(dq,q)
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
        print("pt #",k,", res=",res)
        si[k] = s0
        
verts = []
faces = []
epsz = 0.#1.e-3
for k in range(ni):
    vk = [si[k,0], si[k,1], si[k,2]+epsz]
    verts.append([vk[0], vk[1], vk[2]])
    faces.append([k, (k+1)%ni, ni])
verts.append([s0[0], s0[1], s0[2]])
#create mesh and object
mesh = bpy.data.meshes.new("intersection")
object = bpy.data.objects.new("intersection",mesh)

#set mesh location
object.location = bpy.context.scene.cursor_location
bpy.context.scene.objects.link(object)

#create mesh from python data
mesh.from_pydata(verts,[],faces)
myl.setMaterial(object, bpy.data.materials["mat_surface_dummy"])

##########################################


##########################################
# add hemispheres

# base (intersection)
thetai = np.arctan2(si[:,1]-s0[1], si[:,0]-s0[0])
phii = np.arctan2(si[:,2]-s0[2], np.hypot(si[:,0]-s0[0], si[:,1]-s0[1]))

nr = 50
hs = np.zeros((ni,nr,3))
for k in range(2):
    sgnhalfpi = 0.5*np.pi*(-1.)**k
    for jj in range(nr):
        fj = float(jj)/float(nr+1)
        for ii in range(ni):
            ph = (1. - fj)*phii[ii] + fj*sgnhalfpi
            cph = np.cos(ph)
            th = thetai[ii]
            hs[ii,jj] = s0 + rs*np.array([np.cos(th)*cph, np.sin(th)*cph, np.sin(ph)])
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
    #create mesh and object
    mesh = bpy.data.meshes.new("hemisphere_"+str(k))
    object = bpy.data.objects.new("hemisphere_"+str(k),mesh)
    
    #set mesh location
    object.location = bpy.context.scene.cursor_location
    bpy.context.scene.objects.link(object)
    
    #create mesh from python data
    mesh.from_pydata(verts,[],faces)

    # show mesh as smooth
    object.data.use_auto_smooth = 1
    mypolys = mesh.polygons
    for q in mypolys:
        q.use_smooth = True

    # assign material
    myl.setMaterial(object, bpy.data.materials["mat_sph"])
##########################################
# Mark iso-u edges
patch = bpy.data.objects["surface"]
bpy.context.scene.objects.active = patch
patch.select = True
mesh = bpy.data.meshes["surface"]

bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='DESELECT')
bpy.ops.mesh.select_mode(type="EDGE")
bpy.ops.object.mode_set(mode='OBJECT')

k = j*m + i
for e in mesh.edges:
    v1 = e.vertices[0]
    v2 = e.vertices[1]
    p = min(v1,v2)
    q = max(v1,v2)
    if q == p + m and p == k:
        e.select = True
        break
        
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.loop_multi_select(ring=False)
bpy.ops.mesh.mark_freestyle_edge(clear=False)
bpy.ops.object.mode_set(mode='OBJECT')
patch.select = False

# Mark iso-v edges
patch_dummy = bpy.data.objects["surface_dummy"]
bpy.context.scene.objects.active = patch_dummy
patch_dummy.select = True
mesh = bpy.data.meshes["surface_dummy"]

bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='DESELECT')
bpy.ops.mesh.select_mode(type="EDGE")
bpy.ops.object.mode_set(mode='OBJECT')

for e in mesh.edges:
    v1 = e.vertices[0]
    v2 = e.vertices[1]
    p = min(v1,v2)
    q = max(v1,v2)
    if q == p + 1 and p == k:
        e.select = True
        break

bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.loop_multi_select(ring=False)
bpy.ops.mesh.mark_freestyle_edge(clear=False)
bpy.ops.object.mode_set(mode='OBJECT')
patch_dummy.select = False

##########################################
# CAMERA, SCENE
scene = bpy.context.scene
scene.camera.location = [5.20956, -3.24223, 2.99055]
scene.camera.rotation_euler = np.array([63.2, 0.598, 56.5])*np.pi/180.0
bpy.data.cameras["Camera"].lens_unit = "FOV"
bpy.data.cameras["Camera"].angle = 11.*np.pi/180.0

scene.render.resolution_x = 1400
scene.render.resolution_y = scene.render.resolution_x*5./8.
scene.render.resolution_percentage = 100

bpy.data.worlds["World"].horizon_color = [1,1,1]
scene.render.use_freestyle = True
#scene.render.alpha_mode = 'TRANSPARENT'

bpy.data.worlds["World"].light_settings.use_ambient_occlusion = True
bpy.data.worlds["World"].light_settings.ao_factor = 0.5
bpy.data.worlds["World"].light_settings.ao_blend_type = 'MULTIPLY'
############################################################
# GROUPS
############################################################
for ob in scene.objects:
    ob.select = False


bpy.data.objects['ccircle_u'].select = True
bpy.ops.group.create(name="ccircle_u_group")
bpy.data.objects['ccircle_u'].select = False

bpy.data.objects['ccircle_v'].select = True
bpy.ops.group.create(name="ccircle_v_group")
bpy.data.objects['ccircle_v'].select = False


bpy.data.objects['surface'].select = True
bpy.ops.group.create(name="surface_group")
bpy.data.objects['surface'].select = False

bpy.data.objects['surface_dummy'].select = True
bpy.ops.group.create(name="surface_dummy_group")
bpy.data.objects['surface_dummy'].select = False

"""
bpy.data.objects['envelope_0'].select = True
bpy.data.objects['envelope_1'].select = True
bpy.ops.group.create(name="envelope_group")
bpy.data.objects['envelope_0'].select = False
bpy.data.objects['envelope_1'].select = False
"""

"""
bpy.data.objects['Sphere'].select = True
bpy.ops.group.create(name="sphere_group")
bpy.data.objects['Sphere'].select = False
"""

bpy.data.objects['intersection'].select = True
bpy.ops.group.create(name="intersection_group")
bpy.data.objects['intersection'].select = False

bpy.data.objects['hemisphere_0'].select = True
bpy.ops.group.create(name="hemisphere_0_group")
bpy.data.objects['hemisphere_0'].select = False

bpy.data.objects['hemisphere_1'].select = True
bpy.ops.group.create(name="hemisphere_1_group")
bpy.data.objects['hemisphere_1'].select = False

############################################################
# FREESTYLE
############################################################
bpy.ops.scene.freestyle_lineset_add() #2
bpy.ops.scene.freestyle_lineset_add() #3
bpy.ops.scene.freestyle_lineset_add() #4
bpy.ops.scene.freestyle_lineset_add() #5
bpy.ops.scene.freestyle_lineset_add() #6
bpy.ops.scene.freestyle_lineset_add() #7
bpy.ops.scene.freestyle_lineset_add() #8
bpy.ops.scene.freestyle_lineset_add() #9
bpy.ops.scene.freestyle_lineset_add() #10
bpy.ops.scene.freestyle_lineset_add() #11
bpy.ops.scene.freestyle_lineset_add() #12
bpy.ops.scene.freestyle_lineset_add() #13

freestyle = scene.render.layers.active.freestyle_settings
freestyle.use_smoothness = True

dash_len = 10
gap_len = 7

sp_cl = [0.799,0.396,0.159]
sp_lw = 2.5
sp_dl = 12
sp_gl = sp_dl
# line set 1 (sphere, silhouette, hidden)
lineset = freestyle.linesets["LineSet"]
lineset.select_silhouette = True
lineset.select_border = False
lineset.select_crease = False
lineset.select_by_group = True
lineset.visibility = 'RANGE'#'VISIBLE'
linestyle = bpy.data.linestyles["LineStyle"]
linestyle.caps = "ROUND"
linestyle.color = sp_cl
linestyle.thickness = sp_lw
lineset.group = bpy.data.groups["hemisphere_0_group"]#sphere_group"]

# line set 2 (sphere, silhouette, hidden)
lineset = freestyle.linesets["LineSet 2"]
lineset.select_silhouette = True
lineset.select_border = False
lineset.select_crease = False
lineset.select_by_group = True
lineset.visibility = 'RANGE'#'HIDDEN'
linestyle = bpy.data.linestyles["LineStyle.001"]
linestyle.caps = "ROUND"
linestyle.color = sp_cl
linestyle.thickness = sp_lw
linestyle.use_dashed_line = True
linestyle.dash1 = sp_dl
linestyle.gap1 = sp_gl
lineset.group = bpy.data.groups["hemisphere_1_group"]#"sphere_group"]

cc_lw = 3.5#4.5
# line set 3 (ccircle, border, visible)
lineset = freestyle.linesets["LineSet 3"]
lineset.select_silhouette = False
lineset.select_border = True
lineset.select_crease = False
lineset.select_by_group = True
lineset.visibility = 'VISIBLE'
linestyle = bpy.data.linestyles["LineStyle.002"]
linestyle.caps = "ROUND"
linestyle.use_chaining = False
linestyle.thickness = cc_lw
linestyle.color = ccu_cl
lineset.group = bpy.data.groups["ccircle_u_group"]

# line set 4 (ccircle, border, hidden)
lineset = freestyle.linesets["LineSet 4"]
lineset.select_silhouette = False
lineset.select_border = True
lineset.select_crease = False
lineset.select_by_group = True
lineset.visibility = 'HIDDEN'
linestyle = bpy.data.linestyles["LineStyle.003"]
linestyle.caps = "ROUND"
linestyle.use_chaining = True
linestyle.color = ccu_cl
linestyle.use_dashed_line = True
linestyle.dash1 = dash_len
linestyle.gap1 = gap_len
lineset.group = bpy.data.groups["ccircle_u_group"]


# line set 5 (ccircle, border, visible)
lineset = freestyle.linesets["LineSet 5"]
lineset.select_silhouette = False
lineset.select_border = True
lineset.select_crease = False
lineset.select_by_group = True
lineset.visibility = 'VISIBLE'
linestyle = bpy.data.linestyles["LineStyle.004"]
linestyle.caps = "ROUND"
linestyle.use_chaining = False
linestyle.thickness = cc_lw
linestyle.color = ccv_cl
lineset.group = bpy.data.groups["ccircle_v_group"]

# line set 6 (ccircle, border, hidden)
lineset = freestyle.linesets["LineSet 6"]
lineset.select_silhouette = False
lineset.select_border = True
lineset.select_crease = False
lineset.select_by_group = True
lineset.visibility = 'HIDDEN'
linestyle = bpy.data.linestyles["LineStyle.005"]
linestyle.caps = "ROUND"
linestyle.use_chaining = True
linestyle.color = ccv_cl
linestyle.use_dashed_line = True
linestyle.dash1 = dash_len
linestyle.gap1 = gap_len
lineset.group = bpy.data.groups["ccircle_v_group"]

# line set 7 (surface, border, visible)
lineset = freestyle.linesets["LineSet 7"]
lineset.select_silhouette = False
lineset.select_border = True
lineset.select_crease = False
lineset.select_by_group = True
lineset.visibility = 'VISIBLE'
linestyle = bpy.data.linestyles["LineStyle.006"]
linestyle.caps = "ROUND"
linestyle.use_chaining = True
linestyle.color = [0.330,0.413,0.500]
linestyle.thickness = 1.5
lineset.group = bpy.data.groups["surface_group"]

isouv_lw = 2.
isouv_dl = 5
isouv_gl = 7
# line set 8 (surface, marked edges, visible)
lineset = freestyle.linesets["LineSet 8"]
lineset.select_silhouette = False
lineset.select_border = False
lineset.select_crease = False
lineset.select_edge_mark = True
lineset.select_by_group = True
lineset.visibility = 'VISIBLE'
linestyle = bpy.data.linestyles["LineStyle.007"]
linestyle.caps = "ROUND"
linestyle.color = ccv_cl
linestyle.use_chaining = True
linestyle.thickness = isouv_lw
lineset.group = bpy.data.groups["surface_group"]

# line set 9 (surface, marked edges, hidden)
lineset = freestyle.linesets["LineSet 9"]
lineset.select_silhouette = False
lineset.select_border = False
lineset.select_crease = False
lineset.select_edge_mark = True
lineset.select_by_group = True
lineset.visibility = 'HIDDEN'
linestyle = bpy.data.linestyles["LineStyle.008"]
linestyle.caps = "ROUND"
linestyle.color = ccv_cl
linestyle.use_chaining = True
linestyle.thickness = isouv_lw
linestyle.use_dashed_line = True
linestyle.dash1 = isouv_dl
linestyle.gap1 = isouv_gl
lineset.group = bpy.data.groups["surface_group"]

# line set 10 (surface_dummy, marked edges, visible)
lineset = freestyle.linesets["LineSet 10"]
lineset.select_silhouette = False
lineset.select_border = False
lineset.select_crease = False
lineset.select_edge_mark = True
lineset.select_by_group = True
lineset.visibility = 'VISIBLE'
linestyle = bpy.data.linestyles["LineStyle.009"]
linestyle.caps = "ROUND"
linestyle.color = ccu_cl
linestyle.use_chaining = True
linestyle.thickness = isouv_lw
lineset.group = bpy.data.groups["surface_dummy_group"]

# line set 11 (surface_dummy, marked edges, hidden)
lineset = freestyle.linesets["LineSet 11"]
lineset.select_silhouette = False
lineset.select_border = False
lineset.select_crease = False
lineset.select_edge_mark = True
lineset.select_by_group = True
lineset.visibility = 'HIDDEN'
linestyle = bpy.data.linestyles["LineStyle.010"]
linestyle.caps = "ROUND"
linestyle.color = ccu_cl
linestyle.use_chaining = True
linestyle.thickness = isouv_lw
linestyle.use_dashed_line = True
linestyle.dash1 = isouv_dl
linestyle.gap1 = isouv_gl
lineset.group = bpy.data.groups["surface_dummy_group"]

inter_lw = 1.5
inter_al = 0.4
# line set 12 (intersection, visible)
lineset = freestyle.linesets["LineSet 12"]
lineset.select_silhouette = False
lineset.select_border = True
lineset.select_crease = False
lineset.select_by_group = True
lineset.visibility = 'VISIBLE'
linestyle = bpy.data.linestyles["LineStyle.011"]
linestyle.caps = "ROUND"
linestyle.use_chaining = True
linestyle.thickness = inter_lw
linestyle.alpha = inter_al
lineset.group = bpy.data.groups["intersection_group"]

# line set 12 (intersection, hidden)
lineset = freestyle.linesets["LineSet 13"]
lineset.select_silhouette = False
lineset.select_border = True
lineset.select_crease = False
lineset.select_by_group = True
lineset.visibility = 'HIDDEN'
linestyle = bpy.data.linestyles["LineStyle.012"]
linestyle.caps = "ROUND"
linestyle.use_chaining = True
linestyle.thickness = inter_lw
linestyle.alpha = inter_al
linestyle.use_dashed_line = True
linestyle.dash1 = dash_len
linestyle.gap1 = gap_len
lineset.group = bpy.data.groups["intersection_group"]


for linestyle in bpy.data.linestyles:
    linestyle.geometry_modifiers["Sampling"].sampling = 0.1

##########################################
# LIGHTING
##########################################
# Create new lamp datablock
lamp_data = bpy.data.lamps.new(name="New Lamp", type='POINT')
lamp_data.energy = 2.2
lamp_data.shadow_method = 'NOSHADOW'


# Create new object with our lamp datablock
lamp_object = bpy.data.objects.new(name="New Lamp", object_data=lamp_data)

# Link lamp object to the scene so it'll appear in this scene
bpy.context.scene.objects.link(lamp_object)

# Place lamp to a specified location
fl = 2.0
lamp_object.location = fl*np.array([-0.46846,-3.60067,4.38])
"""
az = -20.*np.pi/180.
el = 30.*np.pi/180.
rl = 6.
az = 1.5*np.pi + np.arctan2(scene.camera.location.y,scene.camera.location.x) - az
lamp_object.location.x = rl*np.cos(az)*np.cos(el)
lamp_object.location.y = rl*np.sin(az)*np.cos(el)
lamp_object.location.z = rl*np.sin(el)
-0.46846,-3.60067,4.38
lamp_object.location.x += scene.camera.location.x + rl*np.cos(az)*np.cos(el)
lamp_object.location.y += scene.camera.location.y + rl*np.sin(az)*np.cos(el)
lamp_object.location.z += scene.camera.location.z + rl*np.sin(el)
"""

# And finally select it make active
lamp_object.select = True
bpy.context.scene.objects.active = lamp_object









