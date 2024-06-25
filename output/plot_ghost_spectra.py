#plot pointing error spectra
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import os
from matplotlib import colormaps as cm
from scipy import stats
opj = os.path.join

def bin_spectrum(cls, lmin=2, lmax=None, binwidth=18, return_error=False):
    """
    Bin a power spectrum (or array of power spectra) using the standard binning.

    Arguments
    ---------
    cls : array_like
        Cl spectrum or list of Cls
    lmin : scalar, optional
        Minimum ell of the first bin.  Default: 8.
    lmax : scalar, optional
        Maximum ell of the last bin.  Default: length of cls array
    binwidth : scalar, optional
        Width of each bin.  Default: 25.
    return_error : bool, optional
        If True, also return an error for each bin, calculated as the statistic

    Returns
    -------
    ellb : array_like
        Bin centers
    clsb : array_like
        Binned power spectra
    clse : array_like
        Error bars on each bin (if return_error is True)
    """

    cls = np.atleast_2d(cls)
    if lmax is None:
        lmax = cls.shape[-1] - 1
    ell = np.arange(lmax + 1)
    bins = np.arange(lmin, lmax + 1, binwidth)
    ellb = stats.binned_statistic(ell, ell, statistic=np.mean, bins=bins)[0]
    clsb = np.array([stats.binned_statistic(
            ell, C, statistic=np.mean, bins=bins)[0] for C in cls]).squeeze()
    if return_error:
        clse = np.array([stats.binned_statistic(
            ell, C, statistic=np.std, bins=bins)[0] for C in cls]).squeeze()
        return ellb, clsb, clse
    return ellb, clsb


binw = 5
input_spec = np.loadtxt("planckrelease3_spectra.txt")
sigma_tau = np.load(opj("..","sigma_tau_cl_target.npy"))
ll, cvEE = np.load(opj("..", "cv_EE_fullsky.npy"))
lldfac = 0.5*ll*(ll+1)/np.pi

beam_type = ["33gauss", "wingfit", "po", "poside"]
beam_names = ["Gaussian", "Fitted", "PO", "PO+sidelobe"]
color_scale =  ["tab:blue", "tab:orange", "tab:green", "tab:red"]
ms = ["o", "s", "v", "^"]
plt.figure(1)
for i , bt in enumerate(beam_type):

    fname = "cmb_ghost_1e-2_fp2_{}_cl.npy".format(bt)
    spec = np.load(opj("cmb_sims", "spectra", fname))
    ell = np.arange(spec.shape[1])
    dfac = 0.5*ell*(ell+1)/np.pi

    bell, bCl, bCle = bin_spectrum(spec*dfac, lmin=2, 
        binwidth=binw, return_error=True)
    bell_offset = (2*i-3)/9.*binw
    plt.errorbar(bell+bell_offset, bCl[1], yerr=bCle[1], 
        label=beam_names[i], ls="none", color=color_scale[i],  
        marker=ms[i], markersize=4)
plt.plot(input_spec[2:,0], input_spec[2:,3], "k", label="Input spectrum")
plt.xlim(2,100)
plt.ylim(5e-4, 1.)
plt.yscale("log")
plt.legend(frameon=False)
plt.xlabel(r"Multipole $\ell$")
plt.ylabel(r"$D_\ell^{EE}$ $(\mu K^2)$")
plt.tight_layout()
plt.savefig(opj("images", "ghost_spectra_cmb.pdf"), dpi=200)

plt.figure(2, figsize=(10./3, 10./3.))
for i , bt in enumerate(beam_type):

    diff_file = "cmb_ghost_1e-2_fp2_{}_diffideal_cl_calno.npy".format(bt)
    spec = np.load(opj("cmb_sims", "spectra", diff_file))
    ell = np.arange(spec.shape[1])
    dfac = 0.5*ell*(ell+1)/np.pi

    bell, bCl, bCle = bin_spectrum(spec*dfac, lmin=2, 
        binwidth=binw, return_error=True)
    bell_offset = (2*i-3)/9.*binw
    plt.errorbar(bell+bell_offset, bCl[1], yerr=bCle[1], label=beam_names[i],
            ls="none", color=color_scale[i], markersize=3, marker=ms[i])

plt.plot(ll, lldfac*cvEE, c="tab:purple")#, label=r"c.v., $f_{sky}=0.7$")
plt.xlim(2,100)
plt.ylim(5e-7, 0.05)
plt.yscale("log")
plt.legend(frameon=False, ncol=2, fontsize=9, loc="upper center", columnspacing=0.5)
plt.xlabel(r"Multipole $\ell$", fontsize=9)
plt.ylabel(r"$D_\ell^{EE}$ $(\mu K^2)$", fontsize=9)
plt.tight_layout()
plt.savefig(opj("images", "ghost_diffspectra_cmb_red.pdf"), dpi=200, bbox_inches="tight")

plt.figure(3)
for i , bt in enumerate(beam_type):

    fname = "dust_ghost_1e-2_fp2_{}_150avg_cl.npy".format(bt)
    spec = np.load(opj("dust_sims", "spectra", fname))
    ell = np.arange(spec.shape[1])
    dfac = 0.5*ell*(ell+1)/np.pi

    bell, bCl, bCle = bin_spectrum(spec*dfac, lmin=2, 
        binwidth=binw, return_error=True)
    bell_offset = (2*i-3)/9.*binw
    plt.errorbar(bell+bell_offset, bCl[1], yerr=bCle[1], 
        label=beam_names[i], ls="none", color=color_scale[i],  
        marker=ms[i], markersize=4)
#plt.plot(input_spec[2:,0], input_spec[2:,3], "k", label="Input spectrum")
plt.xlim(2,100)
plt.ylim(5e-4, 1.)
plt.yscale("log")
plt.legend(frameon=False)
plt.xlabel(r"Multipole $\ell$")
plt.ylabel(r"$D_\ell^{EE}$ $(\mu K^2)$")
plt.tight_layout()
plt.savefig(opj("images", "ghost_spectra_dust.pdf"), dpi=200)

plt.figure(4)
for i , bt in enumerate(beam_type):

    diff_file = "dust_ghost_1e-2_fp2_{}_150avg_diffideal_cl_calno.npy".format(bt)
    spec = np.load(opj("dust_sims", "spectra", diff_file))
    ell = np.arange(spec.shape[1])
    dfac = 0.5*ell*(ell+1)/np.pi

    bell, bCl, bCle = bin_spectrum(spec*dfac, lmin=2, 
        binwidth=binw, return_error=True)
    bell_offset = (2*i-3)/9.*binw
    plt.errorbar(bell+bell_offset, bCl[1], yerr=bCle[1], label=beam_names[i],
            ls="none", color=color_scale[i], markersize=4, marker=ms[i])

plt.plot(ll, lldfac*cvEE, c="tab:purple", label=r"c.v., $f_{sky}=0.7$")
plt.xlim(2,100)
plt.ylim(5e-7, 5e-2)
plt.yscale("log")
plt.legend(frameon=False, ncol=2)
plt.xlabel(r"Multipole $\ell$")
plt.ylabel(r"$D_\ell^{EE}$ $(\mu K^2)$")
plt.tight_layout()
plt.savefig(opj("images", "ghost_diffspectra_dust.pdf"), dpi=200)
plt.show()