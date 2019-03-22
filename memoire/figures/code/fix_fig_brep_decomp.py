import numpy


#####################################################
fname = '../data/fig_brep_faces/edges_008.dat'
dat = numpy.loadtxt(fname, delimiter=',')
i = 1
#dat[i,1] += 0.1
#dat[i,2] += 0.05
dat[i,1] = 4.928507566452026145e-01
dat[i,2] = 2.191462174057960399e-01

numpy.savetxt(fname, dat, delimiter=',')
#####################################################




#####################################################
from PIL import Image, ImageDraw

ifaces = [2,7,8]
eps = 0.01
radius = 0.05
"""
# Determine position of vertex labels in xyz-view
for iface in ifaces:
    strf = format(iface,'03')
    dat = numpy.loadtxt('../data/fig_brep_faces/verts_'+strf+'.dat', delimiter=',')
    pixeldata = []
    for imagename in ['face', 'edges_vis', 'edges_hid']:
        image = Image.open('../fig_brep_faces/'+imagename+'_'+strf+'.png')
        pixeldata.append(list(image.getdata()))
    width = image.size[0]
    height = image.size[1]
    radiuspx = int(round(radius*width))
    modif = numpy.zeros((width,height,4), dtype=float)
    modif[:,:,0] = 255.

    f = open('../data/fig_brep_faces/verts_dxy_'+strf+'.dat', 'w')
    cs = numpy.zeros((len(dat),2))
    for k, v in enumerate(dat):
        iv = int(round((1.0 - v[2])*height))
        jv = int(round((v[1])*width))
        imin = max(0,iv-radiuspx)
        imax = min(height,iv+radiuspx)
        jmin = max(0,jv-radiuspx)
        jmax = min(width,jv+radiuspx)
        c = 0.
        s = 0.
        for i in range(imin,imax):
            y = 1.0 - float(i)/height
            for j in range(jmin,jmax):
                x = float(j)/width
                dist = numpy.hypot(x - v[1], y - v[2])
                if dist <= radius and dist >= eps:
                    a = 0
                    for p in pixeldata:
                        a += p[i*width+j][3]
                    modif[i,j,3] = float(a)
                    c += a*(x - v[1])/dist
                    s += a*(y - v[2])/dist
        h = numpy.hypot(c,s)
        c /= -h
        s /= -h
        cs[k,0] = c
        cs[k,1] = s
        f.write(str(int(v[0])) + ',' +
                str(v[1]) + ',' +
                str(v[2]) + ',' +
                str(v[3]) + ',' +
                str(c) + ',' +
                str(s) + '\n')

    f.close()

    modif[:,:,3] = numpy.around(255.*modif[:,:,3]/numpy.amax(modif[:,:,3]))
    modif = tuple(map(tuple,numpy.reshape(modif.astype(int), [width*height,4])))
    image.putdata(modif)
    draw = ImageDraw.Draw(image)
    for k, v in enumerate(dat):
        draw.line((v[1],v[2], v[1]+radiuspx*cs[k,0],v[2]+radiuspx*cs[k,1]), fill=128, width=2)
    image.save('../fig_brep_faces/output_'+strf+'.png')
#####################################################
"""



stride = 10

seuil = 50
def threshold_alpha(alpha):
    if alpha < seuil:
        return 0
    else:
        return 255


import sys
sys.path.append('/d/bandrieu/GitHub/These/memoire/figures/code/')
import lib_color as mycolors

fid = numpy.loadtxt('../data/fig_brep_faces/faces_id.dat', dtype=int)
# Determine position of vertex labels in xyz-view
for iface in ifaces:
    strf = format(iface,'03')
    image = Image.open('../fig_brep_faces/face_'+strf+'.png')
    width = image.size[0]
    height = image.size[1]
    npx = width*height
    pixeldata = list(image.getdata())
    modif = numpy.zeros((npx,4), dtype=float)
    isinterior = numpy.zeros(len(pixeldata), dtype=bool)
    visited = numpy.zeros(len(pixeldata), dtype=bool)
    countvis = 0
    interior = []
    exterior = []
    boundary = []
    regions = []
    while countvis < npx:
        #print 'visited :',len(interior) + len(exterior), '/',npx
        for iseed in range(len(pixeldata)):
            if visited[iseed]: continue
            #print 'seed :',iseed, ', alpha =',pixeldata[iseed][3]
            alphaseed = pixeldata[iseed][3]
            front = [iseed]
            region = []
            while True:
                #print 'front :',front
                fronttmp = []
                for ifront in front:
                    j = ifront%width
                    i = (ifront-j)/width
                    #print '   ifront :',ifront,', i,j=',i,',',j
                    visited[ifront] = True
                    countvis += 1
                    #print countvis, npx
                    region.append(ifront)
                    isonboundary = False
                    neighbors = []
                    if j < width-1:# east
                        neighbors.append(ifront+1)
                    if i > 0:# north
                        neighbors.append(ifront-width)
                    if j > 0:# west
                        neighbors.append(ifront-1)                   
                    if i < height-1:# south
                        neighbors.append(ifront+width)
                    
                    for ingb in neighbors:
                        if threshold_alpha(pixeldata[ingb][3]) != threshold_alpha(alphaseed):
                            isonboundary = True
                        else:
                            if not visited[ingb] and ingb not in front and ingb not in fronttmp:
                                fronttmp.append(ingb)
                    if isonboundary: boundary.append([i,j])
                if len(fronttmp) == 0: break
                front = fronttmp
            if pixeldata[iseed][3] > 25:
                interior.extend(region)
                isinterior[region] = True
            else:
                exterior.extend(region)
            regions.append(region)
            break

    # Pole of inaccessibility
    dmax = 0.
    imax = int(npx/2)
    nb = len(boundary)
    diagsqr = height**2 + width**2
    if nb > 0:
        for i in range(0,height,stride):
            for j in range(0,width,stride):
                k = i*width + j
                if isinterior[k]:
                    dk = diagsqr
                    for ijb in boundary:
                        dk = min(dk, (ijb[0] - i)**2 + (ijb[1] - j)**2)
                    if dmax < dk:
                        imax = k
                        jpia = j
                        ipia = i
                        dmax = dk

    f = open('../data/fig_brep_faces/face_xyzlabel_' + strf + '.dat', 'w')
    f.write(str(fid[iface-1]+1) + ', ' +
            str(float(jpia)/width) + ', ' +
            str(1.0 - float(ipia)/height))
    f.close()
    
                
    clr = numpy.around(255*mycolors.sample_colormap('IR', len(regions)))
    clr = clr.astype(int)
    for j, r in enumerate(regions):
        if threshold_alpha(pixeldata[r[0]][3]) == 0:
            a = 128
        else:
            a = 255
        for i in r:
            for k in range(3):
                modif[i,k] = clr[j,k]
            modif[i,3] = a
    for ij in boundary:
        modif[ij[0]*width+ij[1],0:3] = 0


    modif[imax] = [0,0,255,255]
    
    modif = tuple(map(tuple,numpy.reshape(modif.astype(int), [width*height,4])))
    image.putdata(modif)

    image.save('../fig_brep_faces/output2_'+strf+'.png')

    dmax = numpy.sqrt(dmax)
    print dmax
    draw = ImageDraw.Draw(image)
    draw.ellipse((jpia-dmax,ipia-dmax,jpia+dmax,ipia+dmax),
                 fill=None, outline='blue')
    image.show()
#####################################################
