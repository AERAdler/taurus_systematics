import healpy as hp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pipeline_tools
import os
opj = os.path.join
plt.switch_backend("agg")

#xqml
import xqml
from xqml.xqml_utils import getstokes
from xqml.simulation import extrapolpixwin, Karcmin2var

def cv(ell, cl, fsky):
    return 2./(2*ell+1)/fsky*cl


#Spice options
def get_default_spice_opts(lmax=700, fsky=None):

    if fsky is None:
        fsky = 1.0

    spice_opts = dict(nlmax=lmax,
        apodizetype=1,
        apodizesigma=180*fsky*0.8,
        thetamax=180*fsky,
        decouple=False,
        symmetric_cl=True,
        outroot=os.path.realpath(__file__),
        verbose=0,
        subav=False,
        subdipole=True,
        return_kernel=True)

    return spice_opts


nside = 8
lmax = 3 * nside - 1

#ell, TT, TE, EE, BB, PP
dl_th = np.loadtxt("planckCOMr3_spectra.txt", unpack=True)
ell = np.arange(dl_th.shape[1], dtype=int)
dfac = ell*(ell+1)/(2*np.pi)
cl = np.zeros((4,lmax))
#Change order to match spice output, let mono and dipole be zero
cl[0,2:] = dl_th[1,:lmax-2]/dfac[2:lmax]#TT
cl[1,2:] = dl_th[3,:lmax-2]/dfac[2:lmax]#EE
cl[2,2:] = dl_th[4,:lmax-2]/dfac[2:lmax]#BB
cl[3,2:] = dl_th[2,:lmax-2]/dfac[2:lmax]#TE


#Create a mask we can use for a spectrum estimation
#It is the union of:
#1 A Planck-like 70% mask 
gal_mask = hp.ud_grade(hp.read_map("gal070_smoothed2.fits"), nside)

#2 A mask based on the scan pattern
condition = hp.ud_grade(hp.read_map("../output/cond_groundnoside.fits"), nside)
hits = hp.ud_grade(hp.read_map("../output/hits_groundnoside.fits"), nside)
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
spice_opts= get_default_spice_opts(lmax, fsky)

#Skip spice steps if you already have your array
#Components of interest
comp=["TT", "EE", "BB", "TE"]


#xQML options -------
comp_xqml = ["TT", "TE", "EE", "BB"]
cl_full = np.zeros((6,dl_th.shape[1]))
cl_full[0,2:] = dl_th[1,:-2]/dfac[2:]#TT
cl_full[1,2:] = dl_th[3,:-2]/dfac[2:]#EE
cl_full[2,2:] = dl_th[4,:-2]/dfac[2:]#BB
cl_full[3,2:] = dl_th[2,:-2]/dfac[2:]#TE

dell=1
lmin=2
muKarcmin = 0.1
fwhm=0.
stokes, spec, istokes, ispecs = getstokes(spec=comp_xqml)
nspec = len(spec)
nstoke = len(stokes)

#xQMl white noise ---
pixvar = Karcmin2var(muKarcmin*1e-6, nside)
xqml_mask = np.ones(12*nside*nside, dtype=bool)
xqml_mask[mask<0.1] = False
xqml_pix = np.sum(xqml_mask)#I don't get it 
xqml_fsky = np.mean(xqml_mask)
varmap = np.ones((nstoke * xqml_pix)) * pixvar *0.
NoiseVar = np.diag(varmap)
#xQML objects -------
bins = xqml.Bins.fromdeltal(lmin, lmax, dell)
esti = xqml.xQML(xqml_mask, bins, cl_full, NA=NoiseVar, NB=NoiseVar, lmax=lmax,
    fwhm=fwhm, spec=spec)
lb = bins.lbin
#Output Cl arrays
spice_cl = np.zeros((100, 6, lmax+1))
xqml_cl = np.zeros((100, 4, lmax+1))

for i in range(100):
    np.random.seed(i)
    inmap = hp.synfast(cl, nside=nside, new=True)
    spice_cl[i], kernel = pipeline_tools.spice(inmap, mask=gal_mask, **spice_opts)

    xqml_map = inmap[:, xqml_mask]
    xqml_map_a = xqml_map + np.random.randn(nstoke, xqml_pix) * np.sqrt(pixvar)
    xqml_map_b = xqml_map + np.random.randn(nstoke, xqml_pix) * np.sqrt(pixvar)
    xqml_cl[i,:,2:] = np.array(esti.get_spectra(xqml_map_a))
np.save("spice_taumask_kernel_couple.npy", kernel)
np.save("spice_cl_couple.npy", spice_cl)
np.save("xqml_cl_couple.npy", xqml_cl)

spice_cl = np.load("spice_cl_couple.npy")
#kernel = np.load("spice_taumask_kernel_couple.npy")
#square_kernel = kernel[:,2:lmax+1,2:lmax+1]
#for i in range(100):
#    spice_cl[i,0,2:] = np.linalg.inv(square_kernel[0])@spice_cl[i,0,2:]
#    spice_cl[i,1,2:] = np.linalg.inv(square_kernel[2])@spice_cl[i,1,2:]
#    spice_cl[i,2,2:] = np.linalg.inv(square_kernel[2])@spice_cl[i,2,2:]
#    spice_cl[i,3,2:] = np.linalg.inv(square_kernel[3])@spice_cl[i,3,2:]

spice_cl.sort(axis=0)
spice_cl1s = np.abs([spice_cl[49] - spice_cl[15], spice_cl[83] - spice_cl[49]])

spice_fl = np.zeros((100, 4, lmax))
for i in range(100):
    spice_fl[i,:,2:]=spice_cl[i,:4,2:lmax]/cl[:,2:lmax]
spice_fl1s = np.abs([spice_fl[49] - spice_fl[15], spice_fl[83] - spice_fl[49]])


xqml_cl = np.load("xqml_cl_couple.npy")
xqml_cl.sort(axis=0)
xqml_cl1s = np.abs([xqml_cl[49] - xqml_cl[15], xqml_cl[83] - xqml_cl[49]])

xqml_fl = np.zeros((100, 4, lmax-1))#Start at ell=2
for i in range(100):
    xqml_fl[i]=xqml_cl[i,:4,2:]/cl_full[:4,2:lmax+1]
xqml_fl1s = np.abs([xqml_fl[49] - xqml_fl[15], xqml_fl[83] - xqml_fl[49]])


polspicedict = dict(linestyle="none", color="tab:blue", label="Polspice")
xqmldict = dict(linestyle="none", color="tab:orange", label="xQML")
plot_order = [0,3,1,2]

fig1, axs1 = plt.subplots(2,2, sharex=True, figsize=(11,5))
fig1.text(0.45, 0.02, r"Multipole $\ell$", fontsize=12)

for k, ax in enumerate(axs1.flat):
    j = plot_order[k]#I want TT,TE,EE,BB as plot order
    ax.set_xlim(0.,23)
    ax.set_ylabel(r"$D_\ell^{{{}}}$".format(comp[j]))
    ax.plot(ell, dfac*cl_full[j], "k--", label="Planck best-fit spectrum")
    cstd = cv(ell, cl_full[j], fsky)
    ax.plot(ell, dfac*(cl_full[j]+cstd), ls=":", color="tab:purple", label="Cosmic variance")
    ax.plot(ell, dfac*(cl_full[j]-cstd), ls=":", color="tab:purple")
    ax.errorbar(ell[:lmax+1]+0.05, dfac[:lmax+1]*spice_cl[49, j], 
        yerr=dfac[:lmax+1]*spice_cl1s[:,j], marker=".", **polspicedict)
    ax.errorbar(lb-0.05, lb*(lb+1)/(2*np.pi)*xqml_cl[49, j, 2:], 
        yerr = lb*(lb+1)/(2*np.pi)*xqml_cl1s[:,j, 2:], marker=".", **xqmldict)


axs1[0,0].set_ylim(0.,1500)
axs1[0,1].set_ylim(0,5)
axs1[1,0].set_ylim(0.,0.05)
axs1[1,1].set_ylim(-1e-3,1e-3)
axs1[1,0].legend(frameon=False)
fig1.suptitle(r"$D_\ell$ comparaison, coupled")
#fig.tight_layout()
plt.savefig("dl_coupled.png", dpi=200)


fig2, axs2 = plt.subplots(2,2, sharex=True, figsize=(11,5))
fig2.text(0.45, 0.02, r"Multipole $\ell$", fontsize=12)

for k, ax in enumerate(axs2.flat):
    j = plot_order[k]#I want TT,TE,EE,BB as plot order
    ax.set_xlim(1.2,23.8)
    ax.set_ylim(0,4)
    if j==2:
        ax.set_ylim(-149,151)
    ax.set_ylabel(r"$F_\ell^{{{}}}$".format(comp[j]))
    cstd = cv(ell, cl_full[j], fsky)
    ax.plot(np.arange(0, 50), np.ones(50), "k--")
    ax.plot(1+ 2./(2*ell+1)/fsky, c="tab:purple", ls=":", label="Cosmic variance")
    ax.plot(1- 2./(2*ell+1)/fsky, c="tab:purple", ls=":")
    ax.errorbar(ell[:23]+0.05, spice_fl[49, j, :23], yerr=spice_fl1s[:,j,:23], 
        marker=".", **polspicedict)
    ax.errorbar(lb-0.05, xqml_fl[49, j], yerr=xqml_fl1s[:,j], 
        marker=".", **xqmldict)
    """
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
    """

axs2[0,0].legend(frameon=False)
fig2.suptitle(r"Transfer function comparaison, coupled")
#fig.tight_layout()
plt.savefig("fl_coupled.png", dpi=200)


