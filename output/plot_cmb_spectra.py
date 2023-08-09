#plot cmb spectra
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
        filename = "cmb_BR{}_fp2_".format(hw) + bt + "_150avg_cl.npy"
        label = str(hw)+" layer, "+beam_names_full[j]
        spec = np.load(opj("cmb_sims", "spectra", filename))
        ell = np.arange(spec.shape[1])
        dfac = 0.5*ell*(ell+1)/np.pi
        plt.plot(ell[2:], dfac[2:]*spec[1,2:], ls=ls[j%2], label=label,
                c=cm["tab20"](i/10 + 0.05*(j>1)))

input_spec = np.loadtxt("planckrelease3_spectra.txt")
plt.plot(input_spec[2:,0], input_spec[2:,3], "k:", label="Input spectrum")
ideal_spec = np.load(opj("cmb_sims", "spectra", "cmb_ideal_fp2_gauss_150_cl.npy"))
plt.plot(ell[2:], dfac[2:]*ideal_spec[1,2:], "k-.", label="Ideal simulation")

plt.legend(ncol=3, frameon=False)
plt.xlim(0.,40)
plt.ylim(-0.05,0.2)
plt.xlabel(r"Multipole $\ell$")
plt.ylabel(r"$D_\ell^{{EE}} (\mu K^2)$")
plt.savefig(opj("images","cmb_spectra.png"), dpi=180)

sigma_tau = np.load(opj("..","sigma_tau_cl_target.npy"))


plt.figure(2, figsize=(12,6))
for i, hw in enumerate(hwp_n):
    for j, bt in enumerate(beam_type):
        filename = "cmb_BR{}_fp2_".format(hw)+bt+"_150avg_diffideal_cl_calno.npy"
        label = str(hw)+" layer, "+beam_names_full[j]
        spec = np.load(opj("cmb_sims", "spectra", filename))
        ell = np.arange(spec.shape[1])
        dfac = 0.5*ell*(ell+1)/np.pi
        plt.plot(ell[2:], dfac[2:]*spec[1,2:], ls=ls[j%2], label=label,
                c=cm["tab20"](i/10 + 0.05*(j>1)))

plt.plot(ell[2:], dfac[2:]*sigma_tau, c="tab:purple", 
    label=r"$\sigma(\tau)=0.003$")
plt.plot(ell[2:], -dfac[2:]*sigma_tau, c="tab:purple")
plt.legend(ncol=3, frameon=False)
plt.xlim(0.,40)
plt.ylim(-0.01,0.004)
plt.xlabel(r"Multipole $\ell$")
plt.ylabel(r"$D_\ell^{{EE}} (\mu K^2)$")
plt.title("Power spectrum of difference maps, no gain calibration")
plt.savefig(opj("images","cmbdiff_spectra.png"), dpi=180)


plt.figure(3, figsize=(12,6))
for i, hw in enumerate(hwp_n):
    for j, bt in enumerate(beam_type):
        filename = "cmb_BR{}_fp2_".format(hw) +bt+"_150avg_diffideal_cl_calTT.npy"
        label = str(hw)+" layer, "+beam_names_full[j]
        spec = np.load(opj("cmb_sims", "spectra", filename))
        ell = np.arange(spec.shape[1])
        dfac = 0.5*ell*(ell+1)/np.pi
        plt.plot(ell[2:], dfac[2:]*spec[1,2:], ls=ls[j%2], label=label, 
                c=cm["tab20"](i/10 + 0.05*(j>1)))

plt.plot(ell[2:], dfac[2:]*sigma_tau, c="tab:purple", 
    label=r"$\sigma(\tau)=0.003$")
plt.plot(ell[2:], -dfac[2:]*sigma_tau, c="tab:purple")
plt.legend(ncol=3, frameon=False)
plt.xlim(0.,40)
plt.ylim(-0.01,0.004)
plt.xlabel(r"Multipole $\ell$")
plt.ylabel(r"$D_\ell^{{EE}} (\mu K^2)$")
plt.title("Power spectrum of difference maps, TT gain calibration")
plt.savefig(opj("images","cmbdiffTT_spectra.png"), dpi=180)


plt.figure(4, figsize=(12,6))
for i, hw in enumerate(hwp_n):
    for j, bt in enumerate(beam_type):
        filename = "cmb_BR{}_fp2_".format(hw)+bt+"_150avg_diffideal_cl_calEE.npy"
        label = str(hw)+" layer, "+beam_names_full[j]
        spec = np.load(opj("cmb_sims", "spectra", filename))
        ell = np.arange(spec.shape[1])
        dfac = 0.5*ell*(ell+1)/np.pi
        plt.plot(ell[2:], dfac[2:]*spec[1,2:], ls=ls[j%2], label=label, 
                c=cm["tab20"](i/10 + 0.05*(j>1)))

plt.plot(ell[2:], dfac[2:]*sigma_tau, c="tab:purple", label=r"$\sigma(\tau)$")
plt.plot(ell[2:], -dfac[2:]*sigma_tau, c="tab:purple")
plt.legend(ncol=3, frameon=False)
plt.xlim(0.,40)
plt.ylim(-0.005,0.003)
plt.xlabel(r"Multipole $\ell$")
plt.ylabel(r"$D_\ell^{{EE}} (\mu K^2)$")
plt.title("Power spectrum of difference maps, EE gain calibration")
plt.savefig(opj("images","cmbdiffEE_spectra.png"), dpi=180)

####Binned

binw = 10
lmax = 767
Bl = hp.gauss_beam(np.radians(.5), lmax=lmax)
sigma_tau = np.load(opj("..","sigma_tau_cl_target.npy"))
f6, ax6 = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(12,8))

cal_type = ["no", "TT", "EE"]
cal_label = ["None", "TT", "EE"]
cal_color = ["tab:blue", "tab:orange", "tab:green"]
mkrs = ["o", "s", "v"]
for i, hw in enumerate(hwp_n):
    for j, bt in enumerate(beam_type):
        for k, ct in enumerate(cal_type):
            filename = "cmb_BR{}_fp2_{}_150avg_diffideal_cl_cal{}.npy".format(
                                                                    hw, bt, ct)
            spec = np.load(opj("cmb_sims", "spectra", filename))
            ell = np.arange(spec.shape[1])
            dfac = 0.5*ell*(ell+1)/np.pi

            bell, bCl, bCle = bin_spectrum(spec*dfac, lmin=2, 
                binwidth=binw, return_error=True)
            bell_offset = 0.25*(k-1)*binw
            ax6[j,i].errorbar(bell+bell_offset, bCl[1], yerr=bCle[1], ls="none", 
                color=cal_color[k], marker=mkrs[k], markersize=4, 
                label=cal_label[k])
        ax6[j,i].plot(ell[2:], dfac[2:]*sigma_tau, c="tab:purple", 
            label=r"$\sigma(\tau)=0.003$")
        if i==0:
            ax6[j,0].set_ylabel(r"$D_\ell^{{EE}} (\mu K^2)$")
        if i==2:
            ax6[j,2].set_ylabel(beam_names_full[j])
            ax6[j,2].yaxis.set_label_position("right")

ax6[0,0].set_xlim(2.,200.)
ax6[0,0].set_ylim(1e-6, 1e-1)
ax6[0,0].set_yscale("log")
ax6[0,2].legend(frameon=False, ncol=3)
ax6[0,0].set_title("1BR")
ax6[0,1].set_title("3BR")
ax6[0,2].set_title("5BR")
ax6[3,0].set_xlabel(r"Multipole $\ell$")
ax6[3,1].set_xlabel(r"Multipole $\ell$")
ax6[3,2].set_xlabel(r"Multipole $\ell$")
f6.suptitle("Residual vs ideal HWP for CMB maps")
f6.tight_layout()
plt.savefig(opj("images","binned_cmb_spectra_hwp_difference.png"), dpi=180)

f7, ax7 = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12,8))
hwp_color = ["tab:blue", "tab:orange", "tab:green"]
mkrs = ["o", "s", "v"]
input_spec = np.loadtxt("planckrelease3_spectra.txt")

for i, ax in enumerate(ax7.flat):
    bt = beam_type[i]
    ax.set_title(beam_names_full[i])
    ax.plot(input_spec[2:,0], input_spec[2:,3], "k", label="Input spectrum")
    ideal_filename = "cmb_ideal_fp2_{}_150_cl.npy".format(bt)
    ideal_spec = np.load(opj("cmb_sims", "spectra", ideal_filename))
    ell = np.arange(ideal_spec.shape[1])
    dfac = 0.5*ell*(ell+1)/np.pi
    belli, bCli, bClie = bin_spectrum(ideal_spec*dfac, lmin=2, 
            binwidth=binw, return_error=True)
    ax.errorbar(belli-binw/3., bCli[1], yerr=bClie[1], ls="none", 
            color="gray", marker="*", markersize=4, label="Ideal HWP")

    for j, hw in enumerate(hwp_n):
        filename = "cmb_BR{}_fp2_{}_150avg_cl.npy".format(hw, bt)
        spec = np.load(opj("cmb_sims", "spectra", filename))
        bell, bCl, bCle = bin_spectrum(spec*dfac, lmin=2, 
            binwidth=binw, return_error=True)

        bell_offset = (2*j-1)/9.*binw
        ax.errorbar(bell+bell_offset, bCl[1], yerr=bCle[1], ls="none", 
            color=hwp_color[j], marker=mkrs[j], markersize=4, label=str(hw)+"BR")

    if i%2==0:
        ax.set_ylabel(r"$D_\ell^{{EE}} (\mu K^2)$")
    if i>1:
        ax.set_xlabel(r"Multipole $\ell$")

    


ax7[0,0].set_xlim(2.,200.)
ax7[0,0].set_ylim(5e-4, 4.)
ax7[0,0].set_yscale("log")
ax7[1,1].legend(frameon=False, ncol=1)
f7.suptitle("CMB simulations")
f7.tight_layout()

plt.savefig(opj("images","binned_cmb_spectra_hwp_compare.png"), dpi=180)


#Difference from ideal HWP with Gaussian beam
f8, ax8 = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(12,8))
for i, hw in enumerate(hwp_n):
    for j, bt in enumerate(beam_type):
        for k, ct in enumerate(cal_type):
            filename = "cmb_BR{}_fp2_{}_150avg_diffidealgauss_cl_cal{}.npy".format(
                                                                    hw, bt, ct)
            spec = np.load(opj("cmb_sims", "spectra", filename))
            ell = np.arange(spec.shape[1])
            dfac = 0.5*ell*(ell+1)/np.pi

            bell, bCl, bCle = bin_spectrum(spec*dfac, lmin=2, 
                binwidth=binw, return_error=True)
            bell_offset = 0.25*(k-1)*binw
            ax8[j,i].errorbar(bell+bell_offset, bCl[1], yerr=bCle[1], ls="none", 
                color=cal_color[k], marker=mkrs[k], markersize=4, 
                label=cal_label[k])
        ax8[j,i].plot(ell[2:], dfac[2:]*sigma_tau, c="tab:purple", 
            label=r"$\sigma(\tau)=0.003$")
        if i==0:
            ax8[j,0].set_ylabel(r"$D_\ell^{{EE}} (\mu K^2)$")
        if i==2:
            ax8[j,2].set_ylabel(beam_names_full[j])
            ax8[j,2].yaxis.set_label_position("right")

ax8[0,0].set_xlim(2.,200.)
ax8[0,0].set_ylim(1e-6, 1e-1)
ax8[0,0].set_yscale("log")
ax8[0,2].legend(frameon=False, ncol=3)
ax8[0,0].set_title("1BR")
ax8[0,1].set_title("3BR")
ax8[0,2].set_title("5BR")
ax8[3,0].set_xlabel(r"Multipole $\ell$")
ax8[3,1].set_xlabel(r"Multipole $\ell$")
ax8[3,2].set_xlabel(r"Multipole $\ell$")
f8.suptitle("Residual vs ideal HWP and gaussian beam for CMB maps")
f8.tight_layout()
plt.savefig(opj("images","binned_cmb_spectra_hwp_difference_from_gauss.png"), dpi=180)


###Ideal HWP everytime, just beam shapes

f9, ax9 = plt.subplots(3, sharex=True, sharey=True, figsize=(8,8))
for i, ax in enumerate(ax9.flat):
    bt = beam_type[i+1]
    ax.set_title(beam_names_full[i+1])
    ax.plot(ell[2:], dfac[2:]*sigma_tau, c="tab:purple", 
            label=r"$\sigma(\tau)=0.003$")
    for j, ct in enumerate(cal_type):
            filename = "cmb_ideal_fp2_{}_150_diffidealgauss_cl_cal{}.npy".format(
                                                                     bt, ct)
            spec = np.load(opj("spectra", filename))
            ell = np.arange(spec.shape[1])
            dfac = 0.5*ell*(ell+1)/np.pi

            bell, bCl, bCle = bin_spectrum(spec*dfac, lmin=2, 
                binwidth=binw, return_error=True)
            bell_offset = (2*j-1)/9.*binw
            ax.errorbar(bell+bell_offset, bCl[1], yerr=bCle[1], ls="none", 
                color=cal_color[j], marker=mkrs[j], markersize=4, 
                label=cal_label[j])



    ax.set_ylabel(r"$D_\ell^{{EE}} (\mu K^2)$")
    ax.set_xlabel(r"Multipole $\ell$")

ax9[0].set_xlim(2.,200.)
ax9[0].set_ylim(1e-9, 1)
ax9[0].set_yscale("log")
ax9[0].legend(frameon=False, ncol=2)

f9.suptitle("CMB residuals vs gaussian beam (ideal HWP)")
f9.tight_layout()
plt.savefig(opj("images","binned_cmb_spectra_difference_from_gauss.png"), dpi=180)

f10, ax10 = plt.subplots(1,3, sharex=True, sharey=True, figsize=(12,6))
gauss_spec = np.load(opj("spectra", "cmb_ideal_fp2_gauss_150_cl.npy"))
for i, bt in enumerate(beam_type[1:]):
    filename = "cmb_ideal_fp2_{}_150_cl.npy".format(bt)
    spec = np.load(opj("spectra", filename))
    ell = np.arange(spec.shape[1])
    dfac = 0.5*ell*(ell+1)/np.pi
    ax10[i].plot(ell, spec[1]/gauss_spec[1])
    ax10[i].set_title(beam_names_full[i+1])
f10.suptitle("Cl ratios")
ax10[0].set_xlim(2.,200.)
ax10[0].set_ylim(0., 2.)
#ax10[0].set_yscale("log")


#plt.savefig(opj("images","binned_spectra_hwp_difference_from_gauss.png"), dpi=180)

"""
plt.figure(5, figsize=(12,6))
for i, hw in enumerate(hwp_n):
    for j, bt in enumerate(beam_type):
        filename = "cmb_BR{}_fp2_".format(hw)+bt+"_150avg_diffideal_cl_calEB.npy"
        label = str(hw)+" layer, "+beam_names_full[j]
        spec = np.load(opj("cmb_sims", "spectra", filename))
        ell = np.arange(spec.shape[1])
        dfac = 0.5*ell*(ell+1)/np.pi
        plt.plot(ell[2:], dfac[2:]*spec[1,2:], ls=ls[j%2], label=label,  
                c=cm["tab20"](i/10 + 0.05*(j>1)))

plt.plot(ell[2:], dfac[2:]*sigma_tau, c="tab:purple", 
    label=r"$\sigma(\tau)=0.003$")
plt.plot(ell[2:], -dfac[2:]*sigma_tau, c="tab:purple")
plt.legend(ncol=3, frameon=False)
plt.xlim(0.,40)
plt.ylim(-0.05,0.15)
plt.xlabel(r"Multipole $\ell$")
plt.ylabel(r"$D_\ell^{{EE}} (\mu K^2)$")
plt.title("Power spectrum of difference maps, EB gain calibration")
plt.savefig(opj("images","cmbdiffEB_spectra.png"), dpi=180)
"""

plt.show()