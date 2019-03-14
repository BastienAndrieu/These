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



