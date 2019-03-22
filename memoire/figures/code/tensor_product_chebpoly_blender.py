import bpy
import sys
import numpy
from numpy.polynomial.chebyshev import chebgrid2d

sys.path.append('/stck/bandrieu/Bureau/Python/mylibs/')
import my_lib as myl

sys.path.append('/d/bandrieu/GitHub/These/memoire/figures/code/')
import lib_color as mycolors
import lib_fig

###############################################
cn = numpy.array([18,117,171])/255.
c0 = numpy.array([248,243,232])/255.
#cn = numpy.array([55,116,160])/255.
#c0 = numpy.array([252,235,186])/255.
cp = numpy.array([160,55,55])/255.
def div_cmap(z):
    if z < 0:
        return (1 + z)*c0 - z*cn
    else:
        return (1 - z)*c0 + z*cp
###############################################


###############################################
nsamp = 200
zsamp = numpy.linspace(-1,1,nsamp+1)
cmap = mycolors.sample_colormap('myBuRd', nsamp+1)
def div_cmap2(z):
    i = numpy.floor(0.5*(z + 1)*nsamp)
    if i == nsamp:
         return cmap[-1]
    else:
        t = (z - zsamp[i])/(zsamp[i+1] - zsamp[i])
        return cmap[i]*(1 - t) + cmap[i+1]*t
###############################################


args = sys.argv

if len(args) < 4:
    M = 4
else:
    M = int(args[3])
    
if len(args) < 5:
    N = 4
else:
    N = int(args[4])

if len(args) < 6:
    samples = 101
else:
    samples = int(args[5])

###############################################
scene = bpy.context.scene
cam = scene.camera
myl.blankScene()

## Set scene
scene.render.resolution_x = 512
scene.render.resolution_y = scene.render.resolution_x
scene.render.resolution_percentage = 100
scene.render.alpha_mode = 'TRANSPARENT'
bpy.data.worlds["World"].horizon_color = [1,1,1]

cam.data.type = 'ORTHO'

az = numpy.radians(180+90-37.5)
el = numpy.radians(30.)

co = cam.location
rcam = 10.
co.x = rcam*numpy.cos(az)*numpy.cos(el)
co.y = rcam*numpy.sin(az)*numpy.cos(el)
co.z = rcam*numpy.sin(el)

rot = cam.rotation_euler
rot.x = 0.5*numpy.pi - el
rot.y = 0
rot.z = az + 0.5*numpy.pi

cam.data.ortho_scale = 3.3

scene.render.image_settings.file_format='PNG'
pthimg = '/d/bandrieu/GitHub/These/memoire/figures/tensor_product_chebpoly/'


###############################################
# grid data
nticks = 3
ticks = numpy.linspace(-1,1,nticks)

"""
for a in [[[-1,-1,-1],[ 1,-1,-1],[1,0,0]],
          [[-1,-1,-1],[-1, 1,-1],[0,1,0]],
          [[-1, 1,-1],[-1, 1, 1],[0,0,1]]]:
    myl.addPolyline(p=a[0:2], clr=a[2], thickness=1e-2)
"""


#exit()




###############################################
u = numpy.linspace(-1.,1.,samples)
y = numpy.tile(u, (samples,1))
x = y.T
patch = myl.addTensorProductPatch(x, y, numpy.zeros((samples,samples)),
                                  name="patch",
                                  location=[0,0,0],
                                  smooth=True,
                                  color=[1,1,1],
                                  alpha=1)
mat = bpy.data.materials['mat_patch']
mat.use_vertex_color_paint = True
mat.use_shadeless = True

mesh = patch.data
verts = mesh.vertices
faces = mesh.polygons


mesh.vertex_colors.new()
vertexcolor = mesh.vertex_colors[0].data

colors = numpy.zeros((len(verts),3))









for n in range(N+1):
    for m in range(M+1):
        c = numpy.zeros((M+1,N+1))
        c[m,n] = 1

        # update elevation
        z = chebgrid2d(u, u, c)
        for j in range(samples):
            for i in range(samples):
                k = samples*j + i
                verts[k].co[2] = z[i,j]
                #colors[k,:] = 0.5*(z[i,j] + 1)
                colors[k,:] = div_cmap2(z[i,j])

        # update vertex colors
        for k, f in enumerate(faces):
            v = f.vertices
            for p in range(f.loop_total):
                l = f.loop_start + p
                vertexcolor[l].color = colors[v[p]]
     
        # Render
        scene.render.filepath = pthimg + 'T_' + str(m) + '_' + str(n)
        bpy.ops.render.render( write_still=True )



"""
f = open('/d/bandrieu/GitHub/These/memoire/figures/data/tensor_product_chebpoly/grid_ticks.dat','w')
for co in [[(-1,-1,-1), ( 1,-1,-1)],
           [(-1,-1,-1), (-1, 1,-1)],
           [(-1, 1,-1), (-1, 1, 1)]]:
    u, v = lib_fig.convert_3d_to_2d_coords(co[0])
    s, t = lib_fig.convert_3d_to_2d_coords(co[1])
    print(u,v,s,t)
    f.write(str(u) + ', ' + str(v) + ', ' + str(s) + ', ' + str(t) + '\n')
f.close()




print('\n\n\n')



x = [0,0,0]
f = open('/d/bandrieu/GitHub/These/memoire/figures/data/tensor_product_chebpoly/grid_lines.dat', 'w')
for idim, co in enumerate([[( 0,-1,-1), ( 0, 1,-1), ( 0, 1, 1)],
                           [(-1, 0,-1), ( 1, 0,-1), ( 1, 0, 1)],
                           [(-1, 1, 0), ( 1, 1, 0), ( 1,-1, 0)]]):
    for w in ticks:
        for y in co:
            x[0] = y[0]
            x[1] = y[1]
            x[2] = y[2]
            x[idim] = w
            u, v = lib_fig.convert_3d_to_2d_coords(x)
            f.write(str(u) + ', ' + str(v) + ', ')
    f.write('\n')
f.close()



f = open('/d/bandrieu/GitHub/These/memoire/figures/data/tensor_product_chebpoly/frame.dat', 'w')
u, v = lib_fig.convert_3d_to_2d_coords([0,0,0])
f.write(str(u) + ', ' + str(v) + '\n')
for vec in numpy.eye(3):
    u, v = lib_fig.convert_3d_to_2d_coords(vec)
    f.write(str(u) + ', ' + str(v) + '\n')
f.close()

"""

