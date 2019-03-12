import numpy as np
from lib_color import rgb2hsv
#########################################
class Vertex:
    def __init__(self, x, y, neighbors, color):
        self.x = x
        self.y = y
        self.neighbors = neighbors
        self.color = color
#########################################
def diff_angle(a1, a2):
    c1 = np.cos(a1)
    s1 = np.sin(a1)
    c2 = np.cos(a2)
    s2 = np.sin(a2)
    return np.arctan2(s1*c2 - c1*s2,c1*c2 + s1*s2)
############################################
def distance_color(hsv1, hsv2):
    return abs(diff_angle(2.*np.pi*hsv1[0], 2.*np.pi*hsv2[0]))
#########################################
def graph_coloring_contrast(verts, colors):
    nverts = len(verts)
    if len(colors) < nverts:
        colors = np.tile(colors, [int(np.ceil(nverts/len(colors))),1])
        
    ncolors = len(colors)
    hsv = rgb2hsv(colors)
    
    icolor = -np.ones(nverts, dtype=int)
    used = np.zeros(ncolors, dtype=bool)

    for i, v in enumerate(verts):
        if max(icolor[v.neighbors]) < 0:
            for j in np.random.permutation(ncolors):
                if not used[j]: break
            icolor[i] = j
        else:
            kmax = 0
            dmax = 0.
            for k, c in enumerate(hsv):
                if used[k]: continue
                dmin = 1000.
                for j in v.neighbors:
                    if icolor[j] >= 0:
                        djk = distance_color(c, hsv[icolor[j]])
                        if djk < dmin:
                            dmin = djk
                if dmin > dmax:
                    kmax = k
                    dmax = dmin
            icolor[i] = kmax
        used[icolor[i]] = True
        v.color = colors[icolor[i]]
    return v, icolor, used
