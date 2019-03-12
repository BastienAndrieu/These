import numpy as np
import sys
sys.path.append('/d/bandrieu/GitHub/These/memoire/figures/code/')
from lib_color import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import colorsys

def pou(x, m=0.5, w=0.5):
    w = min(w, 2.*m, 2.*(1.-m))
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

n = 100
m = 30


L = sample_colormap('I3',n)
L = L[-30::-1]
nL = len(L)

R = sample_colormap('I2',n)
nR = len(R)


q = nL + nR - m
#L = np.vstack([L, np.tile(L[-1], [nR - m,1])])
#R = np.vstack([np.tile(R[0], [nL - m,1]), R])
L = np.vstack([L, np.ones((nR - m,3))])
R = np.vstack([np.ones((nL - m,3)), R])

t = np.linspace(0.,1.,q)

hslLR = np.zeros((2,q,3))
for i in range(2):
    for j in range(q):
        if i == 0:
            r,g,b = L[j]
        else:
            r,g,b = R[j]
        h,l,s = colorsys.rgb_to_hls(r, g, b)
        hslLR[i,j,0] = h
        hslLR[i,j,1] = s
        hslLR[i,j,2] = l

mid = 0.5*(2.*nL - m - 1)/float(q-1)
wid = float(m)/float(q)
print mid, wid

y = pou(t,mid,wid)
hsl = np.zeros((q,3))
rgb = np.zeros((q,3))
for i in range(q):
    hsl[i] = (1.0 - y[i])*hslLR[0,i] + y[i]*hslLR[1,i]
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
        ax[j].plot(t[i], hslLR[0,i,j], 'o',color=L[i],markeredgecolor='none')
        ax[j].plot(t[i], hslLR[1,i,j], 'o',color=R[i],markeredgecolor='none')
        ax[j].plot(t[i], hsl[i,j], 'o',color=rgb[i],markeredgecolor='none')
    #for i in [n-m,n+m]:
    #    ax[j].plot([t[i],t[i]],[0,1],'k--')
    ax[j].plot(t, y, 'k')
    ax[j].plot([mid-0.5*wid,mid-0.5*wid], [0,1], 'k--')
    ax[j].plot([mid+0.5*wid,mid+0.5*wid], [0,1], 'k--')
    ax[j].set_xlim([0,1])
    ax[j].set_ylim([0,1])
    ax[j].set_ylabel(lbl[j])
h = 1.0/float(q)
for i in range(q):
    #if i < nL:
    ax[3].add_patch(patches.Rectangle((0,i*h),1,h,linewidth=0,edgecolor=L[i],facecolor=L[i]))
    #if i > m:
    ax[3].add_patch(patches.Rectangle((1,i*h),1,h,linewidth=0,edgecolor=R[i],facecolor=R[i]))
    ax[3].add_patch(patches.Rectangle((2,i*h),1,h,linewidth=0,edgecolor=rgb[i],facecolor=rgb[i]))
    ax[3].plot([0,2],[mid-0.5*wid,mid-0.5*wid],'k--',lw=0.05)
    ax[3].plot([0,2],[mid+0.5*wid,mid+0.5*wid],'k--',lw=0.05)
ax[3].set_xlim([0,3])
ax[3].set_ylim([0,1])
plt.show()
