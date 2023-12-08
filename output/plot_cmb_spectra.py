#plot cmb spectra
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import os
from matplotlib import colormaps as cm
from scipy import stats
opj = os.path.join
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, LogLocator, NullFormatter


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
f6, ax6 = plt.subplots(3, sharex=True, sharey=True, figsize=(10/3,10/3))
for i, ax in enumerate(ax6.flat):
    bt = beam_type[i+1]
    ax.text(10, 1e-2, beam_names_full[i+1], fontsize=9)
    ax.tick_params(labelsize=9)
    if i<2:
        ax.tick_params(axis="x", direction="in")
    for j, ct in enumerate(cal_type):
            if "po" in bt:
                filename = "cmb_ideal_fp2_{}_150_diffidealgauss_cl_cal{}.npy".format(bt, ct)
            else:
                filename = "cmb_ideal_{}_150_diffidealgauss_cl_cal{}.npy".format(bt, ct)
            spec = np.load(opj("cmb_sims", "spectra", filename))
            ell = np.arange(spec.shape[1])
            dfac = 0.5*ell*(ell+1)/np.pi

            bell, bCl, bCle = bin_spectrum(spec*dfac, lmin=2, 
                binwidth=binw, return_error=True)
            bell_offset = (2*j-1)/9.*binw
            ax.errorbar(bell+bell_offset, bCl[1], yerr=bCle[1], ls="none", 
                color=cal_color[j], marker=cal_mkrs[j], markersize=2, 
                label=cal_label[j])
    ax.plot(ll, lldfac*cvEE/0.7, c="tab:purple")

    ax.set_ylabel(r"$D_\ell^{{EE}} (\mu K^2)$", fontsize=9)

ax6[2].set_xlabel(r"Multipole $\ell$", fontsize=9)
ax6[0].set_xlim(2.,102.)
ax6[0].set_ylim(5e-7, 5e-2)
ax6[0].set_yscale("log")
ax6[0].yaxis.set_major_locator(LogLocator(100))
ax6[0].yaxis.set_minor_locator(LogLocator(100, subs=(0.1,)))
ax6[0].yaxis.set_minor_formatter(NullFormatter())
ax6[2].legend(frameon=False, ncol=3, fontsize=9, columnspacing=0.5, handletextpad=0.1)
f6.tight_layout(h_pad=0.2)
plt.savefig(opj("images","binned_cmb_spectra_difference_from_gauss_red.pdf"), dpi=180, bbox_inches="tight")


### Ghost beam
plt.figure(7, figsize=(10./3, 10./3.))

for i , bt in enumerate(beam_type):

    diff_file = "cmb_ghost_1e-2_fp2_{}_diffideal_cl_calno.npy".format(bt)
    spec = np.load(opj("cmb_sims", "spectra", diff_file))
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
plt.savefig(opj("images", "ghost_diffspectra_cmb_red.pdf"), dpi=200, bbox_inches="tight")


### Pointing
err_mode = ["fixed_azel", "randm_azel", "fixed_polang", "randm_polang"]
err_labels = ["Common az-el offset", "Random az-el offset", 
    r"Common $\xi$ offset", r"Random $\xi$ offset"]

f8, ax8 = plt.subplots(2, 2, figsize=(7,4.2), sharex=True, sharey=True)
for i, ax in enumerate(ax8.flat):
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
            label=beam_names_full[j], ls="none", color=beam_colors[j], 
            markersize=4, marker=beam_mkrs[j])
    ax.plot(ll, lldfac*cvEE/0.7, c="tab:purple")
    ax.text(20, 0.01, err_labels[i], fontsize=9)
    ax.set_xlim(2,100)
    ax.set_yscale("log")
    ax.set_ylim(5e-7, 5e-2)

ax8[1,1].legend(frameon=False, ncol=2, fontsize=9, handletextpad=0.1)
ax8[1,0].set_xlabel(r"Multipole $\ell$", fontsize=9)
ax8[1,1].set_xlabel(r"Multipole $\ell$", fontsize=9)
ax8[0,0].set_ylabel(r"$D_\ell^{EE}$ $(\mu K^2)$", fontsize=9)
ax8[1,0].set_ylabel(r"$D_\ell^{EE}$ $(\mu K^2)$", fontsize=9)
ax8[0,0].yaxis.set_major_locator(LogLocator(100))
ax8[0,0].yaxis.set_minor_locator(LogLocator(100, subs=(0.1,)))
ax8[0,0].yaxis.set_minor_formatter(NullFormatter())

f8.tight_layout(w_pad=0.2, h_pad=0.2)
plt.savefig(opj("images", "pointing_diffspectra.pdf"), 
    dpi=200, bbox_inches="tight")


### Residual vs ideal HWP for CMB maps
f9, ax9 = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(7,4.2))

for i, hw in enumerate(hwp_n):
    for j, bt in enumerate(beam_type):
        for k, ct in enumerate(cal_type):
            if "po" in bt:
                filename = "cmb_BR{}_fp2_{}_150avg_diffideal_cl_cal{}.npy".format(
                                                                    hw, bt, ct)
            else:
                filename = "cmb_BR{}_{}_150avg_diffideal_cl_cal{}.npy".format(
                                                                    hw, bt, ct)
            spec = np.load(opj("cmb_sims", "spectra", filename))
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
        if i==0:
            ax9[j,0].set_ylabel(r"$D_\ell^{{EE}} (\mu K^2)$", fontsize=9)
        if i==2:
            ax9[j,2].set_ylabel(beam_names_full[j], fontsize=9)
            ax9[j,2].yaxis.set_label_position("right")

ax9[0,0].set_xlim(2.,102.)
ax9[0,0].set_ylim(5e-7, 5e-2)
ax9[0,0].set_yscale("log")
ax9[0,0].yaxis.set_major_locator(LogLocator(100))
ax9[0,0].yaxis.set_minor_locator(LogLocator(100, subs=(0.1,)))
ax9[0,0].yaxis.set_minor_formatter(NullFormatter())
ax9[0,1].legend(frameon=False, ncol=2, fontsize=9, columnspacing=0.5, 
                handletextpad=0.1, borderaxespad=0.)
ax9[0,0].set_title("1BR", fontsize=9)
ax9[0,1].set_title("3BR", fontsize=9)
ax9[0,2].set_title("5BR", fontsize=9)
ax9[3,0].set_xlabel(r"Multipole $\ell$", fontsize=9)
ax9[3,1].set_xlabel(r"Multipole $\ell$", fontsize=9)
ax9[3,2].set_xlabel(r"Multipole $\ell$", fontsize=9)
f9.tight_layout(w_pad=0.1, h_pad=0.2)
plt.savefig(opj("images","binned_cmb_spectra_hwp_difference_red.pdf"), bbox_inches="tight")


### HWP rotation angle error
f10, ax10 = plt.subplots(3, sharex=True, sharey=True, figsize=(10/3., 10/3.))

for i, hw in enumerate(hwp_n):
    for j, ct in enumerate(cal_type):

            filename = "cmb_BR{}_misang_150avg_diffideal_cl_cal{}.npy".format(
                                                                    hw, ct)
            spec = np.load(opj("cmb_sims", "spectra", filename))
            ell = np.arange(spec.shape[1])
            dfac = 0.5*ell*(ell+1)/np.pi

            bell, bCl, bCle = bin_spectrum(spec*dfac, lmin=2, 
                binwidth=binw, return_error=True)
            bell_offset = 0.25*(j-1)*binw
            ax10[i].errorbar(bell+bell_offset, bCl[1], yerr=bCle[1], ls="none", 
                color=cal_color[j], marker=cal_mkrs[j], markersize=2, 
                label=cal_label[j])
    ax10[i].text(12, 0.007, str(hw)+"BR", fontsize=9)
    ax10[i].plot(ll, lldfac*cvEE/0.7, c="tab:purple")
    ax10[i].set_ylabel(r"$D_\ell^{{EE}} (\mu K^2)$", fontsize=9)
    ax10[i].tick_params(labelsize=9)
    ax10[i].set_xlim(2.,102.)
    ax10[i].set_ylim(5e-7, 5e-2)
    ax10[i].set_yscale("log")
    ax10[i].yaxis.set_major_locator(LogLocator(100))
    ax10[i].yaxis.set_minor_locator(LogLocator(100, subs=(0.1,)))
    ax10[i].yaxis.set_minor_formatter(NullFormatter())

ax10[1].legend(frameon=False, ncol=3, fontsize=9, columnspacing=0.5, handletextpad=0.1)
ax10[2].set_xlabel(r"Multipole $\ell$", fontsize=9)
f10.tight_layout(h_pad=0.2)
plt.savefig(opj("images","binned_cmb_spectra_difference_hwp_misang.pdf"), dpi=180, bbox_inches="tight")


### Difference from ideal HWP and Gaussian beam
f11, ax11 = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(7,4.2))
for i, hw in enumerate(hwp_n):
    for j, bt in enumerate(beam_type):
        for k, ct in enumerate(cal_type):
            if "po" in bt:
                filename = "cmb_BR{}_fp2_{}_150avg_diffidealgauss_cl_cal{}.npy".format(
                                                                    hw, bt, ct)
            else:
                filename = "cmb_BR{}_{}_150avg_diffidealgauss_cl_cal{}.npy".format(
                                                                    hw, bt, ct)
            spec = np.load(opj("cmb_sims", "spectra", filename))
            ell = np.arange(spec.shape[1])
            dfac = 0.5*ell*(ell+1)/np.pi

            bell, bCl, bCle = bin_spectrum(spec*dfac, lmin=2, 
                binwidth=binw, return_error=True)
            bell_offset = 0.25*(k-1)*binw
            ax11[j,i].errorbar(bell+bell_offset, bCl[1], yerr=bCle[1], ls="none", 
                color=cal_color[k], marker=cal_mkrs[k], markersize=2, 
                label=cal_label[k])
        ax11[j,i].plot(ll, lldfac*cvEE/0.7, c="tab:purple") 
            #, label=r"c.v., $f_{sky}=0.7$")
        if i==0:
            ax11[j,0].set_ylabel(r"$D_\ell^{{EE}} (\mu K^2)$", fontsize=9)
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
ax11[0,1].legend(frameon=False, ncol=2, fontsize=9, columnspacing=0.5, 
                handletextpad=0.1, borderaxespad=0.)
ax11[0,0].set_title("1BR", fontsize=9)
ax11[0,1].set_title("3BR", fontsize=9)
ax11[0,2].set_title("5BR", fontsize=9)
ax11[3,0].set_xlabel(r"Multipole $\ell$", fontsize=9)
ax11[3,1].set_xlabel(r"Multipole $\ell$", fontsize=9)
ax11[3,2].set_xlabel(r"Multipole $\ell$", fontsize=9)
f11.tight_layout(w_pad=0.1, h_pad=0.2)
plt.savefig(opj("images","binned_cmb_spectra_difference_from_idegauss_red.pdf"), 
    dpi=180, bbox_inches="tight")

plt.show()