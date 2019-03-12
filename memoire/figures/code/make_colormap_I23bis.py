import numpy as np
import sys
sys.path.append('/d/bandrieu/GitHub/These/memoire/figures/code/')
from lib_color import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import colorsys

def pou(x):
    return 0.5*(1.0 - np.cos(np.pi*x))

m = 45

L = sample_colormap('I3',100)
L = L[-40:20:-1]
nL = len(L)

R = sample_colormap('I2',150)
nR = len(R)

"""
print 255*L[0]
print 255*R[0]
print 255*R[-1]


print 255*sample_colormap('I2',6)
"""
print R[int(0.68*len(R))]

q = nL + nR + m


t = np.linspace(0.,1.,q)

hsl = np.zeros((q,3))
for i in range(q):
    if i < nL:
        r,g,b = L[i]
    elif i >= nL + m:
        r,g,b = R[i-nL-m]
    else:
        continue
    h,l,s = colorsys.rgb_to_hls(r, g, b)
    hsl[i,0] = h
    hsl[i,1] = s
    hsl[i,2] = l


y = pou(np.linspace(0,1,m))
for i in range(m):
    hsl[nL+i] = (1. - y[i])*hsl[nL-1] + y[i]*hsl[nL+m]


rgb = np.zeros((q,3))
for i in range(q):
    h,s,l = hsl[i]
    r,g,b = colorsys.hls_to_rgb(h, l, s)
    rgb[i,0] = r
    rgb[i,1] = g
    rgb[i,2] = b

np.savetxt('colormaps/I23.dat',rgb)

c = sample_colormap('IR',q)
#c = c[::-1]

lbl = ['hue','saturation','lightness']
fig, ax = plt.subplots(1,4)
for j in range(3):
    for i in range(q):
        ax[j].plot(t[i], hsl[i,j], 'o',color=rgb[i],markeredgecolor='none')
    ax[j].plot(t[nL:nL+m], y, 'k')
    ax[j].set_xlim([0,1])
    ax[j].set_ylim([0,1])
    ax[j].set_ylabel(lbl[j])
h = 1.0/float(q)
for i in range(nL):
    ax[3].add_patch(patches.Rectangle((0,i*h),1,h,linewidth=0,edgecolor=L[i],facecolor=L[i]))
for i in range(nR):
    ax[3].add_patch(patches.Rectangle((0,(i+nL+m)*h),1,h,linewidth=0,edgecolor=R[i],facecolor=R[i]))
for i in range(q):
    ax[3].add_patch(patches.Rectangle((1,i*h),1,h,linewidth=0,edgecolor=rgb[i],facecolor=rgb[i]))
    ax[3].add_patch(patches.Rectangle((2,i*h),1,h,linewidth=0,edgecolor=c[i],facecolor=c[i]))
ax[3].set_xlim([0,3])
ax[3].set_ylim([0,1])
plt.show()



######
