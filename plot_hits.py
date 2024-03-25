#plot_hits

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

hits = hp.read_map("output/cmb_sims/hits_cmb_ideal_33gauss_150.fits")
mask = hp.ud_grade(hp.read_map("output/gal070_eq_smo2.fits"), 256)
cond = hp.read_map("output/cmb_sims/cond_cmb_ideal_33gauss_150.fits")

hits /= np.amax(hits)

plt.figure(1, figsize=(10/3, 2.))
binary_mask = np.rint(mask)
hp.mollview(hits, norm="log", min= 1e-3, max=1, title="", fig=1)
hp.mollview(binary_mask, cmap="Oranges_r", fig=1, cbar=False, 
	title="", min=-1, max=1, alpha=(1-binary_mask)*0.8, reuse_axes=True)
#A Duncan Watts special
CbAx = plt.gcf().get_children()[2]
CbAx.tick_params(labelsize=9)

plt.savefig("normalized_hitsmap_marked_mask.png", dpi=250)

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
plt.savefig("taurus_mask_pcl_limit.pdf", dpi=180)
plt.show()