#spice problems

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pipeline_tools as tools
import os

nside_list = [256, 512, 1024]
lmax_analysis = 600
fwhm_list = np.linspace(10.,60., num=6)
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

def mask_and_spice(ell, nside=512, fwhm=45., lmax_analysis=700):

    mask = np.ones(12*nside**2)#hp.ud_grade(hp.read_map("smoothed_gal040.fits"), nside_out=nside)
    fsky = np.sum(mask)/(12*nside**2)
    maps = hp.alm2map(sky_alm, nside=nside, fwhm=np.radians(fwhm/60.))
    spice_opts2use = get_default_spice_opts(lmax=lmax_analysis, fsky=fsky)
    bl = hp.gauss_beam(fwhm=np.radians(fwhm/60.), lmax=lmax_analysis)
    ell_t = ell[:lmax_analysis+1]
    cl = tools.spice(maps, mask=mask, **spice_opts2use)
    dl = ell_t*(ell_t+1)/(2*np.pi*bl*bl)*cl
    return ell_t, dl


#Input sky
cls = np.loadtxt("wmap7_r0p03_lensed_uK_ext.txt" ,unpack=True)
ell, cls = cls[0], cls[1:]
np.random.seed(25) 
sky_alm = hp.synalm(cls, lmax=1000, new=True, verbose=True)



plt.figure(1)
for fwhm in fwhm_list:
    ell_t, dl = mask_and_spice(ell, nside=512, fwhm=fwhm, lmax_analysis=700)
    plt.plot(ell_t, dl[1], label="FWHM = {:d}".format(int(fwhm)))
plt.plot(ell, ell*(ell+1)/(2*np.pi)*cls[1], 'k--', label="Input CMB spectrum")
plt.xlim(0, 700)
plt.ylim(0, 60)
plt.legend()
plt.xlabel(r"Multipole $\ell$")
plt.ylabel(r"$D_\ell^{EE}$")
plt.title("Unmasked, NSIDE = 512")

plt.figure(2)
for nside in nside_list:
    ell_t, dl = mask_and_spice(ell, nside=nside, fwhm=50., lmax_analysis=700)
    plt.plot(ell_t, dl[1], label="NSIDE = {:d}".format(nside))

plt.plot(ell, ell*(ell+1)/(2*np.pi)*cls[1], 'k--', label="Input CMB spectrum")
plt.xlim(0, 700)
plt.ylim(0, 60)
plt.legend()
plt.xlabel(r"Multipole $\ell$")
plt.ylabel(r"$D_\ell^{EE}$")
plt.title("Unmasked, fwhm=50\'")
plt.show()