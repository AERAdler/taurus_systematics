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


binw=10
input_spec = np.loadtxt("planckrelease3_spectra.txt")
sigma_tau = np.load(opj("..","sigma_tau_cl_target.npy"))

err_mode = ["fixed_azel", "randm_azel", "fixed_polang", "randm_polang"]
color_scale =  ["tab:blue", "tab:orange", "tab:green", "tab:red"]
err_labels = ["Common az-el offset", "Random az-el offset", 
    r"Common $\xi$ offset", r"Random $\xi$ offset"]
ms = ["o", "s", "v", "^"]
plt.figure(1)
for i , em in enumerate(err_mode):

    fname = "{}_point_fp2_gauss_cl.npy".format(em)
    spec = np.load(opj("spectra", fname))
    ell = np.arange(spec.shape[1])
    dfac = 0.5*ell*(ell+1)/np.pi

    bell, bCl, bCle = bin_spectrum(spec*dfac, lmin=2, 
        binwidth=binw, return_error=True)
    bell_offset = (2*i-3)/9.*binw
    plt.errorbar(bell+bell_offset, bCl[1], yerr=bCle[1], label=err_labels[i], 
            ls="none", color=color_scale[i],  marker=ms[i], markersize=4)
plt.plot(input_spec[2:,0], input_spec[2:,3], "k", label="Input spectrum")
plt.xlim(0,200)
plt.ylim(5e-4, 4.)
plt.yscale("log")
plt.legend(frameon=False)
plt.xlabel(r"Multipole $\ell$")
plt.ylabel(r"$D_\ell^{EE}$ $(\mu K^2)$")
plt.tight_layout()
plt.savefig(opj("images", "pointing_spectra.png"), dpi=200)

plt.figure(2)
for i , em in enumerate(err_mode):

    diff_file = "{}_point_fp2_gauss_diffideal_cl_calno.npy".format(em)
    spec = np.load(opj("spectra", diff_file))
    ell = np.arange(spec.shape[1])
    dfac = 0.5*ell*(ell+1)/np.pi

    bell, bCl, bCle = bin_spectrum(spec*dfac, lmin=2, 
        binwidth=binw, return_error=True)
    bell_offset = (2*i-3)/9.*binw
    plt.errorbar(bell+bell_offset, bCl[1], yerr=bCle[1], label=err_labels[i],
            ls="none", color=color_scale[i], markersize=4, marker=ms[i])

plt.plot(ell[2:], dfac[2:]*sigma_tau, c="tab:purple", label=r"$\sigma(\tau)=0.003$")
plt.xlim(0,200)
plt.ylim(1e-7, 0.1)
plt.yscale("log")
plt.legend(frameon=False, ncol=2)
plt.xlabel(r"Multipole $\ell$")
plt.ylabel(r"$D_\ell^{EE}$ $(\mu K^2)$")
plt.tight_layout()
plt.savefig(opj("images", "pointing_diffspectra.png"), dpi=200)
plt.show()