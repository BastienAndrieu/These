import bpy
import numpy as np
from numpy.polynomial.chebyshev import chebval, chebval2d, chebgrid2d
import sys
sys.path.append('/stck/bandrieu/Bureau/Python/mylibs/')
sys.path.append('/d/bandrieu/GitHub/These/memoire/figures/code/')
import chebyshev_lib as cheblib
import lib_EoS as eos
import lib_fig as fig
import my_lib as myl
import my_lib1 as mylb

#########################################################
class Face:
    def __init__(self, xyz, edges):
        self.xyz = xyz
        self.edges = edges
#####################################
class Edge:
    def __init__(self, xyz, verts, faces):
        self.xyz = xyz
        self.verts = verts
        self.faces = faces
#####################################
class Vertex:
    def __init__(self, xyz, edges):
        self.xyz = xyz
        self.edges = edges
#########################################################
def border_id(b, m, n):
    b = b%4
    if b == 0:
        return np.arange(m),0
    elif b == 1:
        return m-1,np.arange(n)
    elif b == 2:
        return np.arange(m-1,-1,-1),n-1
    else:
        return 0,np.arange(n-1,-1,-1)
#########################################################
def corner_id(c, m, n):
    c = c%4
    if c == 0:
        return 0,0
    elif c == 1:
        return m-1,0
    elif c == 2:
        return m-1,n-1
    else:
        return 0,n-1
#########################################################
def fr(x,y,z):
    #return 0.15*np.ones(x.shape)
    #return 0.15*(1. - 0.4*(x+1.) + 0.12*(y+1.) + 0.15*(z+1.))
    return 0.15*(1. + 0.3*(x+1.) - 0.08*(y+1.) + 0.12*(z+1.))
#########################################################
scene = bpy.context.scene
myl.blankScene()

step = 6
niso = 16

m = step*niso + 1
n = m
u = np.linspace(-1.0,1.0,m)
v = np.linspace(-1.0,1.0,n)


clf = [1,1,1]
cle = [0.527, 0.800, 0.213]#[1.0000, 0.9648, 0.6695]#


Faces = []
EFaces = []
# EoS from faces
for i in range(6):
    ps = mylb.readCoeffs2('/d/bandrieu/GitHub/These/memoire/figures/code/EoS_brep/c_'+str(i+1)+'.cheb')
    s = chebgrid2d(u,v,ps)
    obj = myl.addTensorProductPatch(s[0], s[1], s[2],
                          name="face_"+str(i+1),
                          periodu=False, periodv=False,
                          location=[0,0,0],
                          smooth=True,
                          color=clf, alpha=1, emit=0)
    if i > 0:
        scene.objects.active = obj
        bpy.ops.object.material_slot_remove()
        myl.setMaterial(obj, bpy.data.materials["mat_face_1"])
    #----------------------------------------
    dups, dvps = cheblib.diff2(ps)
    M = ps.shape[0]
    N = ps.shape[1]
    uc = cheblib.cgl_nodes(M-1)
    vc = cheblib.cgl_nodes(N-1)
    sc = chebgrid2d(uc,vc,ps)
    rc = fr(sc[0], sc[1], sc[2])
    pr = cheblib.fcht(cheblib.fcht(rc).T).T
    dupr, dvpr = cheblib.diff2(pr)
    r = chebgrid2d(u,v,pr)
    Faces.append(Face(xyz=s, edges=[-1]*4))
    dur = chebgrid2d(u,v,dupr)
    dvr = chebgrid2d(u,v,dvpr)
    dus = chebgrid2d(u,v,dups)
    dvs = chebgrid2d(u,v,dvps)
    e = eos.two_parameter_EoS(s, dus, dvs, r, dur, dvr)
    EFaces.append(Face(xyz=e[0], edges=[]))
    obj = myl.addTensorProductPatch(e[0,:,:,0], e[0,:,:,1], e[0,:,:,2],
                                    name="enve_"+str(i+1),
                                    periodu=False, periodv=False,
                                    location=[0,0,0],
                                    smooth=True,
                                    color=cle, alpha=1, emit=0.2)
    if i > 0:
        scene.objects.active = obj
        bpy.ops.object.material_slot_remove()
        myl.setMaterial(obj, bpy.data.materials["mat_enve_1"])
    obj.layers[1] = True
    obj.layers[0] = False


# Edges and Vertices
Edges = []
Verts = []
eps = 1.e-5 
for ifr in range(5):
    for ifl in range(ifr+1,6):
        for j in range(4):
            ir1, jr1 = corner_id(j,m,n)
            ir2, jr2 = corner_id(j-1,m,n)
            for i in range(4):
                il1, jl1 = corner_id(i,m,n)
                il2, jl2 = corner_id(i+1,m,n)
                if eos.norm2(Faces[ifr].xyz[:,ir1,jr1] - Faces[ifl].xyz[:,il1,jl1]) < eps and eos.norm2(Faces[ifr].xyz[:,ir2,jr2] - Faces[ifl].xyz[:,il2,jl2]) < eps:
                    ir, jr = border_id(j-1,m,n)
                    il, jl = border_id(i,m,n)
                    sr = Faces[ifr].xyz[:,ir,jr]
                    sl = Faces[ifl].xyz[:,il,jl]
                    
                    g = 0.5*(sr[:,::-1] + sl)
                    iv = [-1]*2
                    for k in [0,-1]:
                        for ive,ve in enumerate(Verts):
                            if eos.norm2(g[:,k] - ve.xyz) < eps:
                                iv[k] = ive
                                break
                        if iv[k] < 0:
                            # add vertex
                            Verts.append(Vertex(xyz=g[:,k], edges=[[len(Edges),k]]))
                        else:
                            Verts[iv[k]].edges.append([len(Edges),k])
                    Edges.append(Edge(xyz=g, faces=[[ifl,i],[ifr,j-1]], verts=iv))
                    Faces[ifl].edges[i] = len(Edges)
                    Faces[ifr].edges[j-1] = -len(Edges)
                    break


# EoS from Edges
for ied, ed, in enumerate(Edges):
    g = ed.xyz
    r = fr(g[0], g[1], g[2])
    ir, jr = border_id(ed.faces[1][1],m,n)
    er = EFaces[ed.faces[1][0]].xyz[ir,jr].T
    er = er[:,::-1] - g
    il, jl = border_id(ed.faces[0][1],m,n)
    el = EFaces[ed.faces[0][0]].xyz[il,jl].T - g
    if True:
        M = int(0.25*len(r))
        t = np.linspace(-1.0,1.0,len(r))
        pg = cheblib.chebfit(t, g.T, M)
        dpg = cheblib.diff(pg)
        dg = chebval(t,dpg)
        e = np.zeros((len(r),n,3))
        for i in range(len(r)):
            bw = dg[:,i]/eos.norm2(dg[:,i])
            w = 0.5*(np.dot(er[:,i],bw) + np.dot(el[:,i],bw))
            ewr = er[:,i] - w*bw
            ewl = el[:,i] - w*bw
            a = eos.angle3d(ewr,ewl)
            rwr = eos.norm2(ewr)
            rwl = eos.norm2(ewl)
            rw = 0.5*(rwr + rwl)
            uw = ewr/rwr
            vw = ewl - np.dot(ewl,uw)*uw
            vw = vw/eos.norm2(vw)
            gw = g[:,i] + w*bw 
            for j in range(n):
                aj = 0.5*(v[j] + 1.)*a
                e[i,j] = gw + rw*(uw*np.cos(aj) + vw*np.sin(aj))
    else:
        M = int(0.25*len(r))
        t = np.linspace(-1.0,1.0,len(r))
        pg = cheblib.chebfit(u, g.T, M)
        pr = cheblib.chebfit(u, r.T, M)
        dpg = cheblib.diff(pg)
        dpr = cheblib.diff(pr)
        dg = chebval(u,dpg)
        dr = chebval(u,dpr)
        e = eos.trimmed_canal_surface(g, el, er, dg, r, dr, v)
    EFaces.append(Face(xyz=e, edges=[]))
    obj = myl.addTensorProductPatch(e[:,:,0], e[:,:,1], e[:,:,2],
                                    name="enve_edge_"+str(ied),
                                    periodu=False, periodv=False,
                                    location=[0,0,0],
                                    smooth=True,
                                    color=cle, alpha=1, emit=0)
    scene.objects.active = obj
    bpy.ops.object.material_slot_remove()
    myl.setMaterial(obj, bpy.data.materials["mat_enve_1"])
    obj.layers[1] = True
    obj.layers[0] = False
    
# new Edges
for ef in EFaces[6:]:
    e = ef.xyz
    Edges.append(Edge(xyz=e[:,0].T, verts=[], faces=[]))
    Edges.append(Edge(xyz=e[0,::-1].T, verts=[], faces=[]))
    Edges.append(Edge(xyz=e[-1].T, verts=[], faces=[]))
    Edges.append(Edge(xyz=e[::-1,-1].T, verts=[], faces=[]))
    
    ef.edges = [len(Edges)-4,len(Edges)-3,len(Edges)-2,len(Edges)-1]

# EoS from Verts
for ive, ve in enumerate(Verts):
    x, y, z = ve.xyz
    bpy.ops.mesh.primitive_uv_sphere_add(
                        location=ve.xyz,
                        segments=m,
                        ring_count=n,
                        size=fr(x,y,z))
    if ive == 0:
        name = "Sphere"
    else:
        name = "Sphere."+format(ive,'03')
    obj = bpy.data.objects[name]
    obj.layers[1] = True
    obj.layers[0] = False
    myl.setSmooth(obj)
    myl.setMaterial(obj, bpy.data.materials["mat_enve_1"])

# trace Edges
if False:
    # Freestyle
    for ied, ed in enumerate(Edges):
        scene.render.use_freestyle = False
        ename = "edge_"+format(ied,'03')
        obj, lset, lsty = myl.addFreestylePolyline(ed.xyz.T,
                                 name=ename,
                                 location=[0,0,0],
                                 color=[0,0,0],
                                 thickness=3.0,
                                 dashed=False,
                                 dash=0,
                                 gap=0)
        bpy.data.groups.remove(bpy.data.groups["group_"+ename])
        if ied > 11:
            obj.layers[1] = True
            obj.layers[0] = False

    for ob in scene.objects:
        ob.select = False

    for ied in range(len(Edges)):
        ename = "edge_"+format(ied,'03')
        bpy.data.objects[ename].select = True
        bpy.ops.group.create(name="group_edges")

    freestyle = scene.render.layers.active.freestyle_settings
    lineset = freestyle.linesets["LineSet"]
    lineset.select_by_group = False

    linestyle = bpy.data.linestyles["LineStyle"]
    linestyle.thickness = 1.8
    linestyle.geometry_modifiers["Sampling"].sampling = 0.1

    for ob in scene.objects:
        ob.select = False
else:
    # Mesh
    for ied, ed in enumerate(Edges):
        if ied > 11:
            lay = 1
        else:
            lay = 0
        myl.addPolyline(ed.xyz.T, [0,0,0], 1.25e-3, lay)



#########################################################
mat = myl.surfaceMaterial("mat_dummy",[1,0,0],0,0)
mat.specular_intensity = 0
mat.use_raytrace = False
mat.use_shadows = False
mat.use_cast_buffer_shadows = False

ied = 8
ed = Edges[ied]
g = ed.xyz
r = fr(g[0], g[1], g[2])
ir, jr = border_id(ed.faces[1][1],m,n)
er = EFaces[ed.faces[1][0]].xyz[ir,jr].T
er = er[:,::-1] - g
il, jl = border_id(ed.faces[0][1],m,n)
el = EFaces[ed.faces[0][0]].xyz[il,jl].T - g
M = int(0.25*len(r))
t = np.linspace(-1.0,1.0,len(r))
pg = cheblib.chebfit(t, g.T, M)
dpg = cheblib.diff(pg)
dg = chebval(t,dpg)
e = np.zeros((2,len(r),n,3))
for i in range(len(r)):
    bw = dg[:,i]/eos.norm2(dg[:,i])
    w = 0.5*(np.dot(er[:,i],bw) + np.dot(el[:,i],bw))
    ewr = er[:,i] - w*bw
    ewl = el[:,i] - w*bw
    a = eos.angle3d(ewr,ewl)
    rwr = eos.norm2(ewr)
    rwl = eos.norm2(ewl)
    rw = 0.5*(rwr + rwl)
    uw = ewr/rwr
    vw = ewl - np.dot(ewl,uw)*uw
    vw = vw/eos.norm2(vw)
    gw = g[:,i] + w*bw
    for j in range(n):
        aj = 0.5*(1. + v[j])*a
        bj = 0.5*(1. - v[j])*a + 0.5*(1. + v[j])*2.0*np.pi
        e[0,i,j] = gw + rw*(uw*np.cos(aj) + vw*np.sin(aj))
        e[1,i,j] = gw + rw*(uw*np.cos(bj) + vw*np.sin(bj))

for k in range(2):
    obj = myl.addTensorProductPatch(e[k,:,:,0], e[k,:,:,1], e[k,:,:,2],
                                    name="canal_surface_"+str(k),
                                    periodu=False, periodv=False,
                                    location=[0,0,0],
                                    smooth=True,
                                    color=[1,1,1],
                                    alpha=1, emit=0)
    scene.objects.active = obj
    bpy.ops.object.material_slot_remove()
    myl.setMaterial(obj, bpy.data.materials["mat_dummy"])
#########################################################
freestyle = scene.render.layers.active.freestyle_settings
freestyle.use_smoothness = True
scene.render.use_freestyle = True

for ob in scene.objects:
    ob.select = False
    
for k in range(2):
    name = "canal_surface_"+str(k)
    obj = bpy.data.objects[name]
    scene.objects.active = obj
    obj.select = True
    bpy.ops.group.create(name="group_"+name)
    # mark freestyle iso-u edges
    mesh = bpy.data.meshes[name]
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_mode(type="EDGE")
    bpy.ops.object.mode_set(mode='OBJECT')

    for e in mesh.edges:
        v1 = e.vertices[0]
        v2 = e.vertices[1]
        p = min(v1,v2)
        q = max(v1,v2)
        if k > 0:
            if q == p + m and p == 0:
                e.select = True
        else:
            if q == p + m and (p%step) == 0 and p < m:
                e.select = True
    
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.loop_multi_select(ring=False)
    bpy.ops.mesh.mark_freestyle_edge(clear=False)
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')
    obj.select = False

    bpy.ops.scene.freestyle_lineset_add()
    lineset = freestyle.linesets["LineSet "+str(k+2)]
    linestyle = bpy.data.linestyles["LineStyle."+format(k+1,'03')]
    lineset.select_silhouette = False
    lineset.select_border = False
    lineset.select_crease = False
    lineset.select_edge_mark = True
    lineset.select_by_group = True
    lineset.group = bpy.data.groups["group_"+name]
    lineset.visibility = 'VISIBLE'
    linestyle.color = [0,0,1]
    linestyle.thickness = 1.7
    linestyle.caps = 'ROUND'
    if k > 0:
        linestyle.use_dashed_line = True
        linestyle.dash1 = 7
        linestyle.gap1 = 7

"""
for k in range(2):
    for i in range(0,m,step):
        obj, lset, lsty = myl.addFreestylePolyline(e[k,i,:,:],
                                                   name="ccirc_"+str(k)+"_"+str(i),
                                                   location=[0,0,0],
                                                   color=[0,0,1],
                                                   thickness=1.7,
                                                   dashed=(k>0),
                                                   dash=7,
                                                   gap=7)
        if k > 0:
            break
"""
for linestyle in bpy.data.linestyles:
    linestyle.geometry_modifiers["Sampling"].sampling = 0.1
#########################################################




#########################################################
mat = bpy.data.materials["mat_face_1"]
mat.specular_intensity = 0.1
mat.specular_hardness = 30.
mat.emit = 0.2

mat = bpy.data.materials["mat_enve_1"]
mat.specular_intensity = 0.1
mat.specular_hardness = 30.
mat.emit = 0.2
#########################################################
"""
cz = campos - camtrg
cz = cz/eos.norm2(cz)

cy0 = np.array((0.0,0.0,1.0))
cy = cy0 - np.dot(cy0,cz)*cz
cy = cy/eos.norm2(cy)

cx = np.cross(cy,cz)

rotmat = np.vstack((cx,cy,cz)).T
"""

scene.render.resolution_percentage = 100
bpy.data.cameras["Camera"].lens_unit = "FOV"

cam_settings = 2
if cam_settings == 0:
    scene.camera.location = [1.62388, 1.5668, 1.16487]
    scene.camera.rotation_euler = np.array((61.013, 0, 132.995))*np.pi/180.0
    bpy.data.cameras["Camera"].angle = 38.*np.pi/180.0
    bpy.data.cameras["Camera"].shift_x = -0.01
    scene.render.resolution_x = 800
    scene.render.resolution_y = scene.render.resolution_x
elif cam_settings == 1:
    scene.camera.location = [1.27846, 1.90248, 1.16732]
    scene.camera.rotation_euler = np.array((61.414, 0, 143.825))*np.pi/180.0#
    bpy.data.cameras["Camera"].angle = 37.*np.pi/180.0
    scene.render.resolution_x = 800
    scene.render.resolution_y = 750
elif cam_settings == 2:
    scene.camera.location = [1.37984, 1.68634, 1.30219]
    scene.camera.rotation_euler = np.array((57.403, 0., 139.813))*np.pi/180.0
    bpy.data.cameras["Camera"].angle = 41.*np.pi/180.0
    bpy.data.cameras["Camera"].shift_x = -0.01
    bpy.data.cameras["Camera"].shift_y = -0.015
    scene.render.resolution_x = 800
    scene.render.resolution_y = scene.render.resolution_x

bpy.data.worlds["World"].horizon_color = [1,1,1]
#########################################################

##########################################
# LIGHTING
##########################################
# Create new lamp datablock
lamp_data = bpy.data.lamps.new(name="KeyLight", type='POINT')
lamp_data.energy = 2.2
lamp_data.shadow_method = 'NOSHADOW'


# Create new object with our lamp datablock
lamp_object = bpy.data.objects.new(name="KeyLight", object_data=lamp_data)

# Link lamp object to the scene so it'll appear in this scene
scene.objects.link(lamp_object)

# Place lamp to a specified location
"""
flk = 1.5
lamp_object.location = flk*np.array([-3.36343,1.56752,3.00455])
"""
flk = 1.
lamp_object.location = flk*np.array([-3.24893,3.21769,4.50682])

# And finally select it make active
lamp_object.select = True
scene.objects.active = lamp_object




# Create new lamp datablock
lamp_data = bpy.data.lamps.new(name="FillLight", type='POINT')
lamp_data.energy = 0.2
lamp_data.shadow_method = 'NOSHADOW'


# Create new object with our lamp datablock
lamp_object = bpy.data.objects.new(name="FillLight", object_data=lamp_data)

# Link lamp object to the scene so it'll appear in this scene
scene.objects.link(lamp_object)

# Place lamp to a specified location
flf = 1.0
#lamp_object.location = flf*np.array([5.98033,-2.47243,-0.44377])
lamp_object.location = flf*np.array([2.76828,-4.33204,0.76075])

# And finally select it make active
lamp_object.select = True
scene.objects.active = lamp_object



##################################################
