#plot_hits

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

hits = hp.read_map("output/taurus_mission_maps/hits_cmb_ideal_taumis_gauss_150.fits")
mask = hp.ud_grade(hp.read_map("output/gal070_eq_smo2.fits"), 256)
cond = hp.read_map("output/taurus_mission_maps/cond_cmb_ideal_taumis_gauss_150.fits")

hits /= np.amax(hits)

plt.figure(1, figsize=(440/72, 2.5))
binary_mask = np.rint(mask)
hp.mollview(hits, norm="log", min= 1e-3, max=1, title="", fig=1, cmap="plasma")
hp.mollview(binary_mask, cmap="Blues_r", fig=1, cbar=False, 
    title="", min=-1, max=1, alpha=(1-binary_mask)*0.8, reuse_axes=True)
#A Duncan Watts special
CbAx = plt.gcf().get_children()[2]
CbAx.tick_params(labelsize=11)

plt.savefig("normalized_hitsmap_jcap.pdf", dpi=250)

#Not fully related, but computing the MASTER error estimate for our mask. 
#I need that plot for my thesis

npix = 12*256**2
lmax = 3*256-1
joined_mask = np.ones(npix)
joined_mask[cond>3]=0.
joined_mask[hits==0]=0.
joined_mask *= mask
fsky = np.mean(joined_mask)

print(fsky)
w2 = np.sum(joined_mask**2)/fsky/npix
w4 = np.sum(joined_mask**4)/fsky/npix
print(w2, w4)

ell = np.arange(2, lmax)
vl = (2*ell+1)*fsky*w2*w2/w4
binwidths = [1, 5, 10, 20]
cm = 1/2.54
fig = plt.figure(figsize=(12*cm, 6*cm))
for bw in binwidths:
    plt.plot(ell, np.sqrt(2/vl/bw), label=rf"$\Delta \ell={bw}$")
plt.xlabel(r"Multipole $\ell$", fontsize=11)
plt.ylabel(r"$\Delta C_\ell /C_\ell$", fontsize=11)
plt.xlim(2, 200)
plt.yscale("log")
plt.legend(ncol=2, frameon=False, fontsize=11)
fig.tight_layout()
plt.savefig("taurus_mask_pcl_limit_taumis.pdf", dpi=180)

f3 = plt.figure(3, figsize=(440/72, 2.))

hp.mollview(hits, norm="log", min= 1e-3, max=1, title=None, fig=3, cmap="plasma", sub=121)
hp.mollview(binary_mask, cmap="Blues_r", fig=3, cbar=False, sub=121,
    title="", min=-1, max=1, alpha=(1-binary_mask)*0.8, reuse_axes=True)

hp.mollview(cond, min=2, max=3., title=None, fig=3, sub=122)
hp.mollview(binary_mask, cmap="Blues_r", fig=3, cbar=False, sub=122,
    title="", min=-1, max=1, alpha=(1-binary_mask)*0.8, reuse_axes=True)
cbax1 = plt.gcf().get_children()[2]
cbax1.tick_params(labelsize=11)
cbax2 = plt.gcf().get_children()[4]
cbax2.tick_params(labelsize=11)

for i, ax in enumerate(f3.get_axes(), start=1):
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0 + 0.1, pos.width, pos.height])
plt.savefig("hits_and_cond3_jcap.pdf", dpi=250)
plt.show()