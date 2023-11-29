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


binw=5
input_spec = np.loadtxt("planckrelease3_spectra.txt")
ll, cvEE = np.load(opj("..", "cv_EE_fullsky.npy"))
lldfac = 0.5*ll*(ll+1)/np.pi

err_mode = ["fixed_azel", "randm_azel", "fixed_polang", "randm_polang"]
color_scale =  ["tab:blue", "tab:green", "tab:orange", "tab:red"]
err_labels = ["Common az-el offset", "Random az-el offset", 
    r"Common $\xi$ offset", r"Random $\xi$ offset"]
beam_type = ["33gauss", "po", "wingfit", "poside"]
beam_names_full = ["Gaussian", "PO", "Gaussian+sidelobe", "PO+sidelobe"]
ms = ["o", "s", "v", "^"]


fig, axes = plt.subplots(2, 2, figsize=(7,4), sharex=True, sharey=True)
for i, ax in enumerate(axes.flat):
    for j, bt in enumerate(beam_type):

        err = err_mode[i]
        file = "{}_point_fp2_{}_diffideal_cl_calno.npy".format(err, bt)
        spec = np.load(opj("cmb_sims", "spectra", file))
        ell = np.arange(spec.shape[1])
        dfac = 0.5*ell*(ell+1)/np.pi

        bell, bCl, bCle = bin_spectrum(spec*dfac, lmin=2, 
            binwidth=binw, return_error=True)
        bell_offset = (2*j-3)/9.*binw
        ax.errorbar(bell+bell_offset, bCl[1], yerr=bCle[1], 
            label=beam_names_full[j], ls="none", color=color_scale[j], 
            markersize=4, marker=ms[j])
    ax.plot(ll, lldfac*cvEE/0.7, c="tab:purple", label=r"c.v. $f_{sky}=0.7$")
    ax.text(20, 0.01, err_labels[i], fontweight="bold")
    ax.set_xlim(2,100)
    ax.set_yscale("log")
    ax.set_ylim(5e-7, 5e-2)

axes[1,1].legend(frameon=False, ncol=2, columnspacing=-3.)
axes[1,0].set_xlabel(r"Multipole $\ell$")
axes[1,1].set_xlabel(r"Multipole $\ell$")
axes[0,0].set_ylabel(r"$D_\ell^{EE}$ $(\mu K^2)$")
axes[1,0].set_ylabel(r"$D_\ell^{EE}$ $(\mu K^2)$")
plt.tight_layout()
plt.savefig(opj("images", "pointing_diffspectra.pdf"), 
    dpi=200, bbox_inches="tight")
plt.show()