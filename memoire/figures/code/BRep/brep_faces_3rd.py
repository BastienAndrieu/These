import numpy
from PIL import Image

import sys
sys.path.append('/d/bandrieu/GitHub/Code/Python/')
import lib_pole_of_inaccessibility as lpia

#####################################################
fname = '../../data/BRep/faces/edges_008.dat'
dat = numpy.loadtxt(fname, delimiter=',')
i = 1
#dat[i,1] += 0.1
#dat[i,2] += 0.05
dat[i,1] = 4.928507566452026145e-01
dat[i,2] = 2.191462174057960399e-01

numpy.savetxt(fname, dat, delimiter=',')
#####################################################



ifaces = [2,7,8]



#####################################################
# Determine position of vertex labels in xyz-view
eps = 0.01
radius = 0.05

for iface in ifaces:
    print "Vertex label location in face #", iface
    strf = format(iface,'03')
    dat = numpy.loadtxt('../../data/BRep/faces/verts_'+strf+'.dat', delimiter=',')
    pixeldata = []
    for imagename in ['face', 'edges_vis', 'edges_hid']:
        image = Image.open('../../images/BRep/faces/'+imagename+'_'+strf+'.png')
        pixeldata.append(list(image.getdata()))
    width = image.size[0]
    height = image.size[1]
    radiuspx = int(round(radius*width))

    f = open('../../data/BRep/faces/verts_dxy_'+strf+'.dat', 'w')
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
#####################################################


#####################################################
# Determine position of face labels in xyz-view
stride = 5
seuil = 128

fid = numpy.loadtxt('../../data/BRep/faces/faces_id.dat', dtype=int)

for iface in ifaces:
    print "Face label location (face #", iface,")"
    strf = format(iface,'03')
    image = Image.open('../../images/BRep/faces/face_'+strf+'.png')
    width = image.size[0]
    height = image.size[1]
    npx = width*height

    iPIA, jPIA, dPIA = lpia.brute_force_PIA_pixel(image,
                                                  stride=stride,
                                                  seuil_alpha=seuil)

    f = open('../../data/BRep/faces/face_xyzlabel_' + strf + '.dat', 'w')
    f.write(str(fid[iface-1]+1) + ', ' +
            str(float(jPIA)/width) + ', ' +
            str(1.0 - float(iPIA)/height))
    f.close()

    
    
    
