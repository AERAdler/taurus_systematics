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
beam_type = ["33gauss", "wingfit", "po", "poside"]
beam_names_full = ["Gaussian", "Gaussian+sidelobe", "PO", "PO+sidelobe"]
ls = ["-", "--"]

plt.figure(1, figsize=(12,6))
for i, hw in enumerate(hwp_n):
    for j, bt in enumerate(beam_type):
        if "po" in bt:
            filename = "cmb_BR{}_fp2_{}_150avg_cl.npy".format(hw, bt)
        else:
            filename = "cmb_BR{}_{}_150avg_cl.npy".format(hw, bt)
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
plt.xlim(2.,50)
plt.ylim(2e-6,2e-2)

plt.xlabel(r"Multipole $\ell$")
plt.ylabel(r"$D_\ell^{{EE}} (\mu K^2)$")
plt.savefig(opj("images","cmb_spectra.pdf"), dpi=180, bbox_inches="tight")

sigma_tau = np.load(opj("..","sigma_tau_cl_target.npy"))
ll, cvEE = np.load(opj("..", "cv_EE_fullsky.npy"))
lldfac = 0.5*ll*(ll+1)/np.pi

plt.figure(2, figsize=(12,6))
for i, hw in enumerate(hwp_n):
    for j, bt in enumerate(beam_type):
        if "po" in bt:
            filename = "cmb_BR{}_fp2_{}_150avg_diffideal_cl_calno.npy".format(hw, bt)
        else:
            filename = "cmb_BR{}_{}_150avg_diffideal_cl_calno.npy".format(hw, bt)

        label = str(hw)+" layer, "+beam_names_full[j]
        spec = np.load(opj("cmb_sims", "spectra", filename))
        ell = np.arange(spec.shape[1])
        dfac = 0.5*ell*(ell+1)/np.pi
        plt.plot(ell[2:], dfac[2:]*spec[1,2:], ls=ls[j%2], label=label,
                c=cm["tab20"](i/10 + 0.05*(j>1)))
plt.plot(ll, lldfac*cvEE/0.7, c="tab:purple", label=r"c.v., $f_{sky}=0.7$")
#plt.plot(ell[2:], dfac[2:]*sigma_tau, c="tab:purple", 
#    label=r"$\sigma(\tau)=0.003$")
#plt.plot(ell[2:], -dfac[2:]*sigma_tau, c="tab:purple")
plt.legend(ncol=2, frameon=False)
plt.xlim(2.,50)
plt.ylim(-1e-3,1e-1)
plt.xlabel(r"Multipole $\ell$")
plt.ylabel(r"$D_\ell^{{EE}} (\mu K^2)$")
plt.title("Power spectrum of difference maps, no gain calibration")
plt.savefig(opj("images","cmbdiff_spectra.pdf"), dpi=180, bbox_inches="tight")


plt.figure(3, figsize=(12,6))
for i, hw in enumerate(hwp_n):
    for j, bt in enumerate(beam_type):
        if "po" in bt:
            filename = "cmb_BR{}_fp2_{}_150avg_diffideal_cl_calTT.npy".format(hw, bt)
        else:
            filename = "cmb_BR{}_{}_150avg_diffideal_cl_calTT.npy".format(hw, bt)
        label = str(hw)+" layer, "+beam_names_full[j]
        spec = np.load(opj("cmb_sims", "spectra", filename))
        ell = np.arange(spec.shape[1])
        dfac = 0.5*ell*(ell+1)/np.pi
        plt.plot(ell[2:], dfac[2:]*spec[1,2:], ls=ls[j%2], label=label, 
                c=cm["tab20"](i/10 + 0.05*(j>1)))

plt.plot(ll, lldfac*cvEE/0.7, c="tab:purple", label=r"c.v., $f_{sky}=0.7$")
plt.legend(ncol=3, frameon=False)
plt.xlim(2.,50)
plt.ylim(2e-6,2e-2)
plt.xlabel(r"Multipole $\ell$")
plt.ylabel(r"$D_\ell^{{EE}} (\mu K^2)$")
plt.title("Power spectrum of difference maps, TT gain calibration")
plt.savefig(opj("images","cmbdiffTT_spectra.pdf"), dpi=180, bbox_inches="tight")


plt.figure(4, figsize=(12,6))
for i, hw in enumerate(hwp_n):
    for j, bt in enumerate(beam_type):
        if "po" in bt:
            filename = "cmb_BR{}_fp2_{}_150avg_diffideal_cl_calEE.npy".format(hw, bt)
        else:
            filename = "cmb_BR{}_{}_150avg_diffideal_cl_calEE.npy".format(hw, bt)
        label = str(hw)+" layer, "+beam_names_full[j]
        spec = np.load(opj("cmb_sims", "spectra", filename))
        ell = np.arange(spec.shape[1])
        dfac = 0.5*ell*(ell+1)/np.pi
        plt.plot(ell[2:], dfac[2:]*spec[1,2:], ls=ls[j%2], label=label, 
                c=cm["tab20"](i/10 + 0.05*(j>1)))

plt.plot(ll, lldfac*cvEE/0.7, c="tab:purple", label=r"c.v., $f_{sky}=0.7$")
plt.legend(ncol=3, frameon=False)
plt.xlim(2.,50)
plt.ylim(2e-6,2e-2)
plt.xlabel(r"Multipole $\ell$")
plt.ylabel(r"$D_\ell^{{EE}} (\mu K^2)$")
plt.title("Power spectrum of difference maps, EE gain calibration")
plt.savefig(opj("images","cmbdiffEE_spectra.pdf"), dpi=180, bbox_inches="tight")

####Binned

binw = 5
lmax = 767
Bl = hp.gauss_beam(np.radians(.55), lmax=lmax)
sigma_tau = np.load(opj("..","sigma_tau_cl_target.npy"))
ll, cvEE = np.load(opj("..", "cv_EE_fullsky.npy"))
lldfac = 0.5*ll*(ll+1)/np.pi
f6, ax6 = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(12,8))

cal_type = ["no", "TT", "EE"]
cal_label = ["None", "TT", "EE"]
cal_color = ["tab:blue", "tab:orange", "tab:green"]
mkrs = ["o", "s", "v"]
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
            ax6[j,i].errorbar(bell+bell_offset, bCl[1], yerr=bCle[1], ls="none", 
                color=cal_color[k], marker=mkrs[k], markersize=4, 
                label=cal_label[k])
        #ax6[j,i].plot(ell[2:], dfac[2:]*sigma_tau, c="tab:purple", 
        #    label=r"$\sigma(\tau)=0.003$")
        ax6[j,i].plot(ll, lldfac*cvEE/0.7, c="tab:purple", 
            label=r"c.v., $f_{sky}=0.7$")
        if i==0:
            ax6[j,0].set_ylabel(r"$D_\ell^{{EE}} (\mu K^2)$")
        if i==2:
            ax6[j,2].set_ylabel(beam_names_full[j])
            ax6[j,2].yaxis.set_label_position("right")

ax6[0,0].set_xlim(2.,102.)
ax6[0,0].set_ylim(5e-7, 5e-2)
ax6[0,0].set_yscale("log")
ax6[0,1].legend(frameon=False, ncol=2)
ax6[0,0].set_title("1BR")
ax6[0,1].set_title("3BR")
ax6[0,2].set_title("5BR")
ax6[3,0].set_xlabel(r"Multipole $\ell$")
ax6[3,1].set_xlabel(r"Multipole $\ell$")
ax6[3,2].set_xlabel(r"Multipole $\ell$")
#f6.suptitle("Residual vs ideal HWP for CMB maps")
f6.tight_layout()
plt.savefig(opj("images","binned_cmb_spectra_hwp_difference.pdf"), dpi=180, bbox_inches="tight")
"""
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
"""

#Difference from ideal HWP with Gaussian beam
f8, ax8 = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(12,8))
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
            ax8[j,i].errorbar(bell+bell_offset, bCl[1], yerr=bCle[1], ls="none", 
                color=cal_color[k], marker=mkrs[k], markersize=4, 
                label=cal_label[k])
        ax8[j,i].plot(ll, lldfac*cvEE/0.7, c="tab:purple", 
            label=r"c.v., $f_{sky}=0.7$")
        if i==0:
            ax8[j,0].set_ylabel(r"$D_\ell^{{EE}} (\mu K^2)$")
        if i==2:
            ax8[j,2].set_ylabel(beam_names_full[j])
            ax8[j,2].yaxis.set_label_position("right")

ax8[0,0].set_xlim(2.,100.)
ax8[0,0].set_ylim(5e-7, 5e-2)
ax8[0,0].set_yscale("log")
ax8[0,1].legend(frameon=False, ncol=2)
ax8[0,0].set_title("1BR")
ax8[0,1].set_title("3BR")
ax8[0,2].set_title("5BR")
ax8[3,0].set_xlabel(r"Multipole $\ell$")
ax8[3,1].set_xlabel(r"Multipole $\ell$")
ax8[3,2].set_xlabel(r"Multipole $\ell$")
#f8.suptitle("Residual vs ideal HWP and gaussian beam for CMB maps")
f8.tight_layout()
plt.savefig(opj("images","binned_cmb_spectra_hwp_difference_from_gauss.pdf"), 
    dpi=180, bbox_inches="tight")


###Ideal HWP everytime, just beam shapes

f9, ax9 = plt.subplots(3, sharex=True, sharey=True, figsize=(8,8))
for i, ax in enumerate(ax9.flat):
    bt = beam_type[i+1]
    ax.text(10, 1e-2, beam_names_full[i+1], fontsize=18)
    ax.tick_params(labelsize=18)
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
                color=cal_color[j], marker=mkrs[j], markersize=4, 
                label=cal_label[j])
    ax.plot(ll, lldfac*cvEE/0.7, c="tab:purple", label=r"c.v., $f_{sky}=0.7$")

    ax.set_ylabel(r"$D_\ell^{{EE}} (\mu K^2)$", fontsize=18)
ax9[2].set_xlabel(r"Multipole $\ell$", fontsize=18)
ax9[0].set_xlim(2.,100.)
ax9[0].set_ylim(5e-7, 5e-2)
ax9[0].set_yscale("log")
ax9[2].legend(frameon=False, ncol=2, loc=4, fontsize=18)
#f9.suptitle("CMB residuals vs gaussian beam (ideal HWP)")

f9.tight_layout()
plt.savefig(opj("images","binned_cmb_spectra_difference_from_gauss.pdf"), dpi=180, bbox_inches="tight")
"""
f10, ax10 = plt.subplots(1,3, sharex=True, sharey=True, figsize=(12,6))
gauss_spec = np.load(opj("cmb_sims", "spectra", "cmb_ideal_33gauss_150_cl.npy"))
for i, bt in enumerate(beam_type[1:]):
    filename = "cmb_ideal_fp2_{}_150_cl.npy".format(bt)
    spec = np.load(opj("cmb_sims", "spectra", filename))
    ell = np.arange(spec.shape[1])
    dfac = 0.5*ell*(ell+1)/np.pi
    ax10[i].plot(ell, spec[1]/gauss_spec[1])
    ax10[i].set_title(beam_names_full[i+1])
f10.suptitle("Cl ratios")
ax10[0].set_xlim(2.,200.)
ax10[0].set_ylim(0., 2.)
#ax10[0].set_yscale("log")
"""

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