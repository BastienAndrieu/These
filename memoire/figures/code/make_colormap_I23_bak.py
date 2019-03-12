import numpy as np
import sys
sys.path.append('/d/bandrieu/GitHub/These/memoire/figures/code/')
from lib_color import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import colorsys

def pou(x, m=0.5, w=0.5):
    w = min(w, m, 1.-m)
    y = np.zeros(x.shape)
    d = 0.5*w
    x1 = m - d
    x2 = m + d
    for i, xi in enumerate(x):
        if xi < x1:
            y[i] = 0.0
        elif xi > x2:
            y[i] = 1.0
        else:
            y[i] = 0.5*(1.0 - np.cos(0.5*np.pi*(xi - x1)/d))
    return y




frac = 0#0.6
n = 100
m = int(frac*float(n))


rgb10 = sample_colormap('I3',n)
rgb10 = rgb10[-30::-1]
rgb20 = sample_colormap('I2',n)

rgb1 = np.vstack([rgb10, np.tile(rgb10[-1],(2*n-m-len(rgb10),1))])
rgb2 = np.vstack([np.tile(rgb20[0],(2*n-m-len(rgb20),1)), rgb20])
#rgb1 = np.vstack([rgb10, rgb20[0:m]])
#rgb2 = np.vstack([rgb10[0:m], rgb20])

q = len(rgb1)
t = np.linspace(0,1,q)

hsl12 = np.zeros((2,q,3))
for i in range(2):
    for j in range(q):
        if i == 0:
            r,g,b = rgb1[j]
        else:
            r,g,b = rgb2[j]
        h,l,s = colorsys.rgb_to_hls(r, g, b)
        hsl12[i,j,0] = h
        hsl12[i,j,1] = s
        hsl12[i,j,2] = l

y = pou(t,0.5,2.0*frac)
hsl = np.zeros((q,3))
rgb = np.zeros((q,3))
for i in range(q):
    hsl[i] = (1.0 - y[i])*hsl12[0,i] + y[i]*hsl12[1,i]
    h,s,l = hsl[i]
    r,g,b = colorsys.hls_to_rgb(h, l, s)
    rgb[i,0] = r
    rgb[i,1] = g
    rgb[i,2] = b

np.savetxt('colormaps/I23.dat',rgb)

lbl = ['hue','saturation','lightness']
fig, ax = plt.subplots(1,4)
for j in range(3):
    for i in range(q):
        #ax[j].plot(t[i], hsl12[0,i,j], 'o',color=rgb1[i],markeredgecolor='none')
        #ax[j].plot(t[i], hsl12[1,i,j], 'o',color=rgb2[i],markeredgecolor='none')
        ax[j].plot(t[i], hsl[i,j], 'o',color=rgb[i],markeredgecolor='none')
    #for i in [n-m,n+m]:
    #    ax[j].plot([t[i],t[i]],[0,1],'k--')
    #ax[j].plot(t, y, 'k')
    ax[j].plot([0.5-0.5*frac,0.5-0.5*frac], [0,1], 'k--')
    ax[j].plot([0.5+0.5*frac,0.5+0.5*frac], [0,1], 'k--')
    ax[j].set_xlim([0,1])
    ax[j].set_ylim([0,1])
h = 1.0/float(q)
for i in range(q):
    if i < len(rgb10):
        ax[3].add_patch(patches.Rectangle((0,i*h),1,h,linewidth=0,edgecolor=rgb1[i],facecolor=rgb1[i]))
    if i > m:
        ax[3].add_patch(patches.Rectangle((1,i*h),1,h,linewidth=0,edgecolor=rgb2[i],facecolor=rgb2[i]))
    ax[3].add_patch(patches.Rectangle((2,i*h),1,h,linewidth=0,edgecolor=rgb[i],facecolor=rgb[i]))
    #ax[3].plot(t[i],t[i],'o',color=rgb[i],markeredgecolor='none')
ax[3].set_xlim([0,3])
ax[3].set_ylim([0,1])
plt.show()
