#plot dust spectra
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import os
from matplotlib import colormaps as cm
from scipy import stats
opj = os.path.join

def bin_cl(cl, binw):

    lmax = cl.shape[0]-1
    bins = np.arange(2, lmax, binw)
    indxs = np.digitize(ell, bins)
    binned_cl = np.zeros_like(bins, dtype=float)
    binned_cl[indxs-1] += cl/binw
    return binned_cl

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
beam_type = ["gauss", "gaussside", "po", "poside"]
beam_names_full = ["Gaussian", "Gaussian+sidelobe", "PO", "PO+sidelobe"]
ls = ["-", "--"]
plt.figure(1, figsize=(12,6))
for i, hw in enumerate(hwp_n):
    for j, bt in enumerate(beam_type):
        filename = "dust_BR{}_fp2_".format(hw) + bt + "_150avg_cl.npy"
        label = str(hw)+" layer, "+beam_names_full[j]
        spec = np.load(opj("dust_sims", "spectra", filename))
        ell = np.arange(spec.shape[1])
        dfac = 0.5*ell*(ell+1)/np.pi
        plt.plot(ell[2:], dfac[2:]*spec[1,2:], c=cm["tab20"](i/10 + 0.05*(j>1)), 
            ls=ls[j%2], label=label)

input_spec = np.load(opj("dust_sims", "spectra", "dustinput_150avg_cl.npy"))
plt.plot(ell[2:], dfac[2:]*input_spec[1,2:], "k:", label="Input spectrum")
ideal_spec = np.load(opj("dust_sims", "spectra", "dust_ideal_fp2_gauss_150avg_cl.npy"))
plt.plot(ell[2:], dfac[2:]*ideal_spec[1,2:], "k-.", label="Ideal simulation")

plt.legend(ncol=3, frameon=False)
plt.xlim(0.,40)
plt.ylim(-0.1,1.4)
plt.xlabel(r"Multipole $\ell$")
plt.ylabel(r"$D_\ell^{{EE}}$")
plt.savefig(opj("images","dust_spectra.png"), dpi=180)


plt.figure(2, figsize=(12,6))
for i, hw in enumerate(hwp_n):
    for j, bt in enumerate(beam_type):
        filename = "dust_BR{}_fp2_".format(hw) + bt + "_150avg_diffideal_cl_calno.npy"
        label = str(hw)+" layer, "+beam_names_full[j]
        spec = np.load(opj("dust_sims", "spectra", filename))
        ell = np.arange(spec.shape[1])
        dfac = 0.5*ell*(ell+1)/np.pi
        plt.plot(ell[2:], dfac[2:]*spec[1,2:], c=cm["tab20"](i/10 + 0.05*(j>1)), 
            ls=ls[j%2], label=label)

plt.legend(ncol=3, frameon=False)
plt.xlim(0.,40)
plt.ylim(-0.005,0.015)
plt.xlabel(r"Multipole $\ell$")
plt.ylabel(r"$D_\ell^{{EE}}$")
plt.title("Power spectrum of difference maps, no gain calibration")
plt.savefig(opj("images","dustdiff_spectra.png"), dpi=180)


plt.figure(3, figsize=(12,6))
for i, hw in enumerate(hwp_n):
    for j, bt in enumerate(beam_type):
        filename = "dust_BR{}_fp2_".format(hw) + bt + "_150avg_diffideal_cl_calTT.npy"
        label = str(hw)+" layer, "+beam_names_full[j]
        spec = np.load(opj("dust_sims", "spectra", filename))
        ell = np.arange(spec.shape[1])
        dfac = 0.5*ell*(ell+1)/np.pi
        plt.plot(ell[2:], dfac[2:]*spec[1,2:], c=cm["tab20"](i/10 + 0.05*(j>1)), 
            ls=ls[j%2], label=label)

plt.legend(ncol=3, frameon=False)
plt.xlim(0.,40)
plt.ylim(-0.005,0.01)
plt.xlabel(r"Multipole $\ell$")
plt.ylabel(r"$D_\ell^{{EE}}$")
plt.title("Power spectrum of difference maps, TT gain calibration")
plt.savefig(opj("images","dustdiffTT_spectra.png"), dpi=180)


plt.figure(4, figsize=(12,6))
for i, hw in enumerate(hwp_n):
    for j, bt in enumerate(beam_type):
        filename = "dust_BR{}_fp2_".format(hw) + bt + "_150avg_diffideal_cl_calEE.npy"
        label = str(hw)+" layer, "+beam_names_full[j]
        spec = np.load(opj("dust_sims", "spectra", filename))
        ell = np.arange(spec.shape[1])
        dfac = 0.5*ell*(ell+1)/np.pi
        plt.plot(ell[2:], dfac[2:]*spec[1,2:], c=cm["tab20"](i/10 + 0.05*(j>1)), 
            ls=ls[j%2], label=label)

plt.legend(ncol=3, frameon=False)
plt.xlim(0.,40)
plt.ylim(-0.003,0.003)
plt.xlabel(r"Multipole $\ell$")
plt.ylabel(r"$D_\ell^{{EE}}$")
plt.title("Power spectrum of difference maps, EE gain calibration")
plt.savefig(opj("images","dustdiffEE_spectra.png"), dpi=180)


####Binned

binw = 10
lmax = 767
Bl = hp.gauss_beam(np.radians(.5), lmax=lmax)
sigma_tau = np.load(opj("..","sigma_tau_cl_target.npy"))
f5, ax5 = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(12,8))

cal_type = ["no", "TT", "EE"]
cal_label = ["None", "TT", "EE"]
cal_color = ["tab:blue", "tab:orange", "tab:green"]
mkrs = ["o", "s", "v"]
for i, hw in enumerate(hwp_n):
    for j, bt in enumerate(beam_type):
        for k, ct in enumerate(cal_type):
            filename = "dust_BR{}_fp2_{}_150avg_diffideal_cl_cal{}.npy".format(
                                                                    hw, bt, ct)
            spec = np.load(opj("dust_sims", "spectra", filename))
            ell = np.arange(spec.shape[1])
            dfac = 0.5*ell*(ell+1)/np.pi

            bell, bCl, bCle = bin_spectrum(spec*dfac, lmin=2, 
                binwidth=binw, return_error=True)
            bell_offset = 0.25*(k-1)*binw
            ax5[j,i].errorbar(bell+bell_offset, bCl[1], yerr=bCle[1], ls="none", 
                color=cal_color[k], marker=mkrs[k], markersize=4, 
                label=cal_label[k])
        ax5[j,i].plot(ell[2:], dfac[2:]*sigma_tau, c="tab:purple", 
            label=r"$\sigma(\tau)=0.003$")
        if i==0:
            ax5[j,0].set_ylabel(r"$D_\ell^{{EE}}$")
        if i==2:
            ax5[j,2].set_ylabel(beam_names_full[j])
            ax5[j,2].yaxis.set_label_position("right")

ax5[0,0].set_xlim(2.,200.)
ax5[0,0].set_ylim(1e-8, 1e-1)
ax5[0,0].set_yscale("log")
ax5[0,2].legend(frameon=False, ncol=2)
ax5[0,0].set_title("1BR")
ax5[0,1].set_title("3BR")
ax5[0,2].set_title("5BR")
ax5[3,0].set_xlabel(r"Multipole $\ell$")
ax5[3,1].set_xlabel(r"Multipole $\ell$")
ax5[3,2].set_xlabel(r"Multipole $\ell$")
f5.suptitle("Residual vs ideal HWP for Dust maps")
f5.tight_layout()
plt.savefig(opj("images","binned_dust_spectra_hwp_difference.png"), dpi=180)

f6, ax6 = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12,8))
hwp_color = ["tab:blue", "tab:orange", "tab:green"]
mkrs = ["o", "s", "v"]
input_spec = np.load(opj("dust_sims", "spectra", "dustinput_150avg_cl.npy"))

for i, ax in enumerate(ax6.flat):
    bt = beam_type[i]
    ax.set_title(beam_names_full[i])
    
    ideal_filename = "dust_ideal_fp2_{}_150avg_cl.npy".format(bt)
    ideal_spec = np.load(opj("dust_sims", "spectra", ideal_filename))
    ell = np.arange(ideal_spec.shape[1])
    ax.plot(ell, 0.11*(ell/80.)**-0.42, "k", label=r"$0.11(\ell/80)^{{-0.42}}$")
    dfac = 0.5*ell*(ell+1)/np.pi
    belli, bCli, bClie = bin_spectrum(ideal_spec*dfac, lmin=2, 
            binwidth=binw, return_error=True)
    ax.errorbar(belli-binw/3., bCli[1], yerr=bClie[1], ls="none", 
            color="gray", marker="*", markersize=4, label="Ideal HWP")

    for j, hw in enumerate(hwp_n):
        filename = "dust_BR{}_fp2_{}_150avg_cl.npy".format(hw, bt)
        spec = np.load(opj("dust_sims", "spectra", filename))
        bell, bCl, bCle = bin_spectrum(spec*dfac, lmin=2, 
            binwidth=binw, return_error=True)

        bell_offset = (2*j-1)/9.*binw
        ax.errorbar(bell+bell_offset, bCl[1], yerr=bCle[1], ls="none", 
            color=hwp_color[j], marker=mkrs[j], markersize=4, label=str(hw)+"BR")

    if i%2==0:
        ax.set_ylabel(r"$D_\ell^{{EE}}$")
    if i/2==1:
        ax.set_xlabel(r"Multipole $\ell$")

    


ax6[0,0].set_xlim(2.,200.)
ax6[0,0].set_ylim(0.01, 1.)
ax6[0,0].set_yscale("log")
ax6[1,1].legend(frameon=False, ncol=1)
f6.suptitle("Dust simulations")
f6.tight_layout()

plt.savefig(opj("images","binned_dust_spectra_hwp_compare.png"), dpi=180)

#Difference from ideal HWP with Gaussian beam
f7, ax7 = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(12,8))
for i, hw in enumerate(hwp_n):
    for j, bt in enumerate(beam_type):
        for k, ct in enumerate(cal_type):
            filename = "dust_BR{}_fp2_{}_150avg_diffidealgauss_cl_cal{}.npy".format(
                                                                    hw, bt, ct)
            spec = np.load(opj("dust_sims", "spectra", filename))
            ell = np.arange(spec.shape[1])
            dfac = 0.5*ell*(ell+1)/np.pi

            bell, bCl, bCle = bin_spectrum(spec*dfac, lmin=2, 
                binwidth=binw, return_error=True)
            bell_offset = 0.25*(k-1)*binw
            ax7[j,i].errorbar(bell+bell_offset, bCl[1], yerr=bCle[1], ls="none", 
                color=cal_color[k], marker=mkrs[k], markersize=4, 
                label=cal_label[k])
        ax7[j,i].plot(ell[2:], dfac[2:]*sigma_tau, c="tab:purple", 
            label=r"$\sigma(\tau)=0.003$")
        if i==0:
            ax7[j,0].set_ylabel(r"$D_\ell^{{EE}}$")
        if i==2:
            ax7[j,2].set_ylabel(beam_names_full[j])
            ax7[j,2].yaxis.set_label_position("right")

ax7[0,0].set_xlim(2.,200.)
ax7[0,0].set_ylim(1e-8, 1e-1)
ax7[0,0].set_yscale("log")
ax7[0,2].legend(frameon=False, ncol=3)
ax7[0,0].set_title("1BR")
ax7[0,1].set_title("3BR")
ax7[0,2].set_title("5BR")
ax7[3,0].set_xlabel(r"Multipole $\ell$")
ax7[3,1].set_xlabel(r"Multipole $\ell$")
ax7[3,2].set_xlabel(r"Multipole $\ell$")
f7.suptitle("Residual vs ideal HWP and gaussian beam for dust maps")
f7.tight_layout()
plt.savefig(opj("images","binned_dust_spectra_hwp_difference_from_gauss.png"), dpi=180)

"""
plt.figure(5, figsize=(12,6))
for i, hw in enumerate(hwp_n):
    for j, bt in enumerate(beam_type):
        filename = "dust_BR{}_".format(hw) + bt + "_150avg_diffideal_cl_calEB.npy"
        label = str(hw)+" layer, "+beam_names_full[j]
        spec = np.load(opj("dust_sims", "spectra", filename))
        ell = np.arange(spec.shape[1])
        dfac = 0.5*ell*(ell+1)/np.pi
        plt.plot(ell[2:], dfac[2:]*spec[1,2:], c=cm["tab20"](i/10 + 0.05*(j>1)), 
            ls=ls[j%2], label=label)

plt.legend(ncol=3, frameon=False)
plt.xlim(0.,40)
plt.ylim(-0.2,0.4)
plt.xlabel(r"Multipole $\ell$")
plt.ylabel(r"$D_\ell^{{EE}}$")
plt.title("Power spectrum of difference maps, EB gain calibration")
plt.savefig(opj("images","dustdiffEB_spectra.png"), dpi=180)
"""

plt.show()