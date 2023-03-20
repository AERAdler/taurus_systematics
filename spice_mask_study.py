import healpy as hp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from beamconv import tools
import pipeline_tools
import os
opj = os.path.join
plt.switch_backend("agg")

nside=256
lmax=700
#ell, TT, TE, EE, BB, PP
spp = np.loadtxt("planckCOMr3_spectra.txt", unpack=True)
ell = np.arange(lmax)
cl = np.zeros((4,lmax))
#Change order to match spice output, let mono and dipole be zero
cl[0,2:] = spp[1,:lmax-2]#TT
cl[1,2:] = spp[3,:lmax-2]#EE
cl[2,2:] = spp[4,:lmax-2]#BB
cl[3,2:] = spp[2,:lmax-2]#TE

cl[:,2:] = cl[:,2:]*2*np.pi/(ell[2:]*(ell[2:]+1))#Go from dl to cl

#Spice options
def get_default_spice_opts(lmax=700, fsky=None):

    if fsky is None:
        fsky = 1.0

    spice_opts = dict(nlmax=lmax,
        apodizetype=1,
        apodizesigma=180*fsky*0.8,
        thetamax=180*fsky,
        decouple=True,
        symmetric_cl=True,
        outroot=os.path.realpath(__file__),
        verbose=0,
        subav=False,
        subdipole=True)

    return spice_opts

#Create a mask we can use for a spectrum estimation
#It is the union of:
#1 A Planck-like 70% mask 
gal_mask = hp.read_map("gal070_smoothed2.fits")

#2 A mask based on the scan pattern
condition = hp.read_map("output/cond_groundnoside.fits")
hits = hp.read_map("output/hits_groundnoside.fits")
scan_mask = np.ones(12*nside*nside)
scan_mask[condition>2.5] = 0
scan_mask[hits==0] = 0
scan_mask = hp.smoothing(scan_mask, fwhm=np.radians(2))

mask = gal_mask*scan_mask
hp.mollview(mask)
plt.savefig("spice_mask.png", dpi=200)

fsky = np.sum(mask)/(12*nside**2)
print("fsky is: ", fsky)
#Spice options call
spice_opts= get_default_spice_opts(lmax-1, fsky)

#Skip spice steps if you already have your array

#Output Cl arrays
spice_cl = np.zeros((100, 6, lmax))
for i in range(100):
    np.random.seed(i)
    inmap = hp.synfast(cl, nside=nside, new=True)
    spice_cl[i] = pipeline_tools.spice(inmap, mask=gal_mask, **spice_opts)

np.save("spice_cl.npy", spice_cl)

spice_cl = np.load("spice_cl.npy")

spice_cl.sort(axis=0)
spice_fl = np.zeros((100, 4, 200))
for i in range(100):
    spice_fl[i]=spice_cl[i,:4,:200]/cl[:,:200]

spice_1s = np.abs([spice_fl[49] - spice_fl[15], spice_fl[83] - spice_fl[49]])

comp=["TT", "EE", "BB", "TE"]

twosigdict = dict(linestyle="none", marker="x", color="tab:blue")
onesigdict = dict(linestyle="none", marker=".", color="tab:blue",)
fig, axs = plt.subplots(2,2, sharex=True, figsize=(11,5))
plot_order = [0,3,1,2]
fig.text(0.45, 0.02, r"Multipole $\ell$", fontsize=12)
for k, ax in enumerate(axs.flat):
    j = plot_order[k]#I want TT,TE,EE,BB as plot order
    ax.set_xlim(2.5,20)
    ax.set_ylim(0,4)
    if j==2:
        ax.set_ylim(-149,151)
    #ax.set_title(comp[j])
    ax.set_ylabel(r"$F_\ell^{{{}}}$".format(comp[j]))
    ax.plot(ell[:20], np.ones(20), "k--")
    ax.errorbar(ell[:20], spice_fl[49, j, :20], yerr=spice_1s[:,j,:20], 
        label="68%", **onesigdict)
    ax.plot(ell[:20], spice_fl[2,j,:20], label="95%", **twosigdict)
    ax.plot(ell[:20], spice_fl[97,j,:20], **twosigdict)

    ax.spines["right"].set_visible(False)
    divider = make_axes_locatable(ax)
    axLin = divider.append_axes("right", size=1.0, pad=0)
    axLin.set_xlim((20, 200))

    fl_mid_bin = np.average(spice_fl[49,j,20:].reshape((9,20)), axis=1)
    fl_low_twostd = np.average(spice_fl[2,j,20:].reshape((9,20)), axis=1)
    fl_high_twostd = np.average(spice_fl[97,j,20:].reshape((9,20)), axis=1)
    low_onestd = np.average((spice_fl[49,j,20:] 
                                - spice_fl[15, j, 20:]).reshape(9,20), axis=1)
    high_onestd = np.average((spice_fl[83,j,20:] 
                                - spice_fl[49, j, 20:]).reshape(9,20), axis=1)
    binned_1s = np.abs([low_onestd, high_onestd])
    axLin.errorbar(ell[30:200:20], fl_mid_bin, yerr=binned_1s, 
        label="68%", **onesigdict)
    axLin.plot(ell[30:200:20], fl_low_twostd, **twosigdict)
    axLin.plot(ell[30:200:20], fl_high_twostd, **twosigdict)
    axLin.plot(ell[20:], np.ones(680), "k--")
    axLin.spines["left"].set_visible(False)
    axLin.yaxis.set_ticks_position("right")
    axLin.set_ylim(0,4)

    if j==2:
        axLin.set_ylim(0,2)
        ax.set_ylabel(r"$F_\ell^{{{}}}$".format(comp[j]) ,labelpad=-17)
    else:
        plt.setp(axLin.get_yticklabels(), visible=False)

axs[0,0].legend(frameon=False)
fig.suptitle(r"Transfer function for input best-fit Planck spectrum")
#fig.tight_layout()
plt.savefig("transfer_function.png", dpi=200)


