#plot dust spectra
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import os
from matplotlib import colormaps as cm
from scipy import stats
opj = os.path.join
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, LogLocator, NullFormatter

plt.rc('xtick', labelsize=9)
plt.rc('ytick', labelsize=9)

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


hwp_n = [1,3,5]
beam_type = ["33gauss", "wingfit", "po", "poside"]
beam_names_full = ["Gaussian", "Fitted", "PO", "PO+sidelobe"]
ls = ["-", "--"]
cal_type = ["no", "TT", "EE"]
cal_label = ["None", "TT", "EE"]
cal_color = ["tab:blue", "tab:orange", "tab:green"]
cal_mkrs = ["o", "s", "v"]

beam_colors =  ["tab:blue", "tab:orange", "tab:green", "tab:red"]
beam_mkrs = ["o", "s", "v", "^"]

binw = 5
lmax = 767

ll, cvEE = np.load(opj("..", "cv_EE_fullsky.npy"))
lldfac = 0.5*ll*(ll+1)/np.pi


### Beam model error
f6, ax6 = plt.subplots(3, sharex=True, sharey=True, figsize=(10/3,3.5))
for i, ax in enumerate(ax6.flat):
    bt = beam_type[i+1]
    ax.text(10, 1e-2, beam_names_full[i+1], fontsize=9)
    ax.tick_params(labelsize=9)
    if i<2:
        ax.tick_params(axis="x", direction="in")
    for j, ct in enumerate(cal_type):
        if "po" in bt:
            filename = "dust_ideal_fp2_{}_150avg_diffidealgauss_cl_cal{}.npy".format(bt, ct)
        else:
            filename = "dust_ideal_{}_150avg_diffidealgauss_cl_cal{}.npy".format(bt, ct)
        spec = np.load(opj("dust_sims", "spectra", filename))
        ell = np.arange(spec.shape[1])
        dfac = 0.5*ell*(ell+1)/np.pi

        bell, bCl, bCle = bin_spectrum(spec*dfac, lmin=2, 
            binwidth=binw, return_error=True)
        bell_offset = (2*j-1)/9.*binw
        ax.errorbar(bell+bell_offset, bCl[1], yerr=bCle[1], ls="none", 
            color=cal_color[j], marker=cal_mkrs[j], markersize=2, label=cal_label[j])

    ax.plot(ll, lldfac*cvEE/0.7, c="tab:purple")

ax6[0].set_xlim(2.,102.)
ax6[0].set_ylim(5e-7, 5e-2)
ax6[0].set_yscale("log")
ax6[0].yaxis.set_major_locator(LogLocator(100))
ax6[0].yaxis.set_minor_locator(LogLocator(100, subs=(0.1,)))
ax6[0].yaxis.set_minor_formatter(NullFormatter())
ax6[2].legend(frameon=False, ncol=3, loc="lower right", fontsize=9, columnspacing=0.5, handletextpad=0.1, bbox_to_anchor=(0.99,-0.1))
f6.supxlabel(r"Multipole $\ell$", fontsize=9, x=0.55, y=0.07)
f6.supylabel(r"$D_\ell^{{EE}} (\mu K^2)$", fontsize=9, y=0.57, x=0.07)
f6.tight_layout(h_pad=0.2)
plt.savefig(opj("images","binned_dust_spectra_difference_from_gauss_red.pdf"), dpi=180, bbox_inches="tight")

### Ghost beam
plt.figure(7, figsize=(10./3, 10./3.))

for i , bt in enumerate(beam_type):

    diff_file = "dust_ghost_1e-2_fp2_{}_150avg_diffideal_cl_calno.npy".format(bt)
    spec = np.load(opj("dust_sims", "spectra", diff_file))
    ell = np.arange(spec.shape[1])
    dfac = 0.5*ell*(ell+1)/np.pi

    bell, bCl, bCle = bin_spectrum(spec*dfac, lmin=2, 
        binwidth=binw, return_error=True)
    bell_offset = (2*i-3)/9.*binw
    plt.errorbar(bell+bell_offset, bCl[1], yerr=bCle[1], label=beam_names_full[i],
            ls="none", color=beam_colors[i], markersize=3, marker=beam_mkrs[i])

plt.plot(ll, lldfac*cvEE, c="tab:purple")
plt.xlim(2,100)
plt.ylim(5e-7, 0.05)
plt.yscale("log")
plt.legend(frameon=False, ncol=2, fontsize=9, loc="upper center", columnspacing=0.5)
plt.xlabel(r"Multipole $\ell$", fontsize=9)
plt.ylabel(r"$D_\ell^{EE}$ $(\mu K^2)$", fontsize=9)
plt.tight_layout()
plt.savefig(opj("images", "ghost_diffspectra_dust_red.pdf"), dpi=200, bbox_inches="tight")

### Residual vs ideal HWP for dust maps
f9, ax9 = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(7,4.2))

for i, hw in enumerate(hwp_n):
    for j, bt in enumerate(beam_type):
        for k, ct in enumerate(cal_type):
            if "po" in bt:
                filename = "dust_BR{}_fp2_{}_150avg_diffideal_cl_cal{}.npy".format(
                                                                    hw, bt, ct)
            else:
                filename = "dust_BR{}_{}_150avg_diffideal_cl_cal{}.npy".format(
                                                                    hw, bt, ct)
            spec = np.load(opj("dust_sims", "spectra", filename))
            ell = np.arange(spec.shape[1])
            dfac = 0.5*ell*(ell+1)/np.pi

            bell, bCl, bCle = bin_spectrum(spec*dfac, lmin=2, 
                binwidth=binw, return_error=True)
            bell_offset = 0.25*(k-1)*binw
            ax9[j,i].errorbar(bell+bell_offset, bCl[1], yerr=bCle[1], ls="none", 
                color=cal_color[k], marker=cal_mkrs[k], markersize=2, 
                label=cal_label[k])

        ax9[j,i].plot(ll, lldfac*cvEE/0.7, c="tab:purple")
        ax9[j,i].tick_params(labelsize=9)

        if i==2:
            ax9[j,2].set_ylabel(beam_names_full[j], fontsize=9)
            ax9[j,2].yaxis.set_label_position("right")

ax9[0,0].set_xlim(2.,102.)
ax9[0,0].set_ylim(5e-7, 5e-2)
ax9[0,0].set_yscale("log")
ax9[0,0].yaxis.set_major_locator(LogLocator(100))
ax9[0,0].yaxis.set_minor_locator(LogLocator(100, subs=(0.1,)))
ax9[0,0].yaxis.set_minor_formatter(NullFormatter())
ax9[0,1].legend(frameon=False, ncol=3, fontsize=9, columnspacing=0.5, 
                handletextpad=0.1, borderaxespad=0.)
ax9[0,0].set_title("1BR", fontsize=9)
ax9[0,1].set_title("3BR", fontsize=9)
ax9[0,2].set_title("5BR", fontsize=9)
f9.supylabel(r"$D_\ell^{{EE}} (\mu K^2)$", fontsize=9, y=0.53, x=0.04)
f9.supxlabel(r"Multipole $\ell$", fontsize=9, x=0.55, y=0.05)
f9.tight_layout(w_pad=0.1, h_pad=0.2)
plt.savefig(opj("images","binned_dust_spectra_hwp_difference_red.pdf"), bbox_inches="tight")

### Difference from ideal HWP and Gaussian beam
f11, ax11 = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(7,4.2))
for i, hw in enumerate(hwp_n):
    for j, bt in enumerate(beam_type):
        for k, ct in enumerate(cal_type):
            if "po" in bt:
                filename = "dust_BR{}_fp2_{}_150avg_diffidealgauss_cl_cal{}.npy".format(
                                                                    hw, bt, ct)
            else:
                filename = "dust_BR{}_{}_150avg_diffidealgauss_cl_cal{}.npy".format(
                                                                    hw, bt, ct)
            spec = np.load(opj("dust_sims", "spectra", filename))
            ell = np.arange(spec.shape[1])
            dfac = 0.5*ell*(ell+1)/np.pi

            bell, bCl, bCle = bin_spectrum(spec*dfac, lmin=2, 
                binwidth=binw, return_error=True)
            bell_offset = 0.25*(k-1)*binw
            ax11[j,i].errorbar(bell+bell_offset, bCl[1], yerr=bCle[1], ls="none", 
                color=cal_color[k], marker=cal_mkrs[k], markersize=2, 
                label=cal_label[k])
        ax11[j,i].plot(ll, lldfac*cvEE/0.7, c="tab:purple") 

        if i==2:
            ax11[j,2].set_ylabel(beam_names_full[j], fontsize=9)
            ax11[j,2].yaxis.set_label_position("right")
        ax11[j,i].tick_params(labelsize=9)

ax11[0,0].set_xlim(2.,102.)
ax11[0,0].set_ylim(5e-7, 5e-2)
ax11[0,0].set_yscale("log")
ax11[0,0].yaxis.set_major_locator(LogLocator(100))
ax11[0,0].yaxis.set_minor_locator(LogLocator(100, subs=(0.1,)))
ax11[0,0].yaxis.set_minor_formatter(NullFormatter())
ax11[0,1].legend(frameon=False, ncol=3, fontsize=9, columnspacing=0.5, 
                handletextpad=0.1, borderaxespad=0.)
ax11[0,0].set_title("1BR", fontsize=9)
ax11[0,1].set_title("3BR", fontsize=9)
ax11[0,2].set_title("5BR", fontsize=9)
f11.supxlabel(r"Multipole $\ell$", fontsize=9, x=0.55, y=0.05)
f11.supylabel(r"$D_\ell^{{EE}} (\mu K^2)$", fontsize=9, y=0.53, x=0.04)
f11.tight_layout(w_pad=0.1, h_pad=0.2)
plt.savefig(opj("images","binned_dust_spectra_difference_from_idegauss_red.pdf"), 
    dpi=180, bbox_inches="tight")

plt.show()