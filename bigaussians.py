#make beams with sidelobes

import numpy as np
import healpy as hp
from beamconv import tools
import matplotlib.pyplot as plt
import matplotlib.cm 
from matplotlib.lines import Line2D
import os
opj = os.path.join
fwhm = 30
nside = 512
lmax = 700
pixels = np.arange(hp.nside2npix(nside))
theta, phi = hp.pix2ang(nside, pixels)
#off-center secondary
amplitudes = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4]
spreads = [5., 10., 15.]
colors = matplotlib.cm.get_cmap('tab20b')
shades = matplotlib.cm.get_cmap('tab20c')
plt.figure(1, figsize=(9,5), dpi=100)
for i, side_amp in enumerate(amplitudes):
    for j, side_fwhm in enumerate(spreads):
        blm_main = tools.gauss_blm(fwhm, lmax, pol=False)
        blm_main *= 1-side_amp
        blm_main = tools.get_copol_blm(blm_main, c2_fwhm=fwhm)
        blm_side = tools.gauss_blm(side_fwhm*60, lmax, pol=False)
        blm_side *= side_amp
        blm_side = tools.get_copol_blm(blm_side, c2_fwhm=side_fwhm*60)
        blm = (np.array(blm_side)+np.array(blm_main)).tolist()
        plt.plot(np.real(blm[0][:lmax+1]/blm_main[0][:lmax+1]*(1-side_amp)), 
            c=colors(.2*i+.05*j))
        filename = 'direct_blm_{:.0e}sideamp_{:0>2d}sidefwhm'.format(
            side_amp, int(side_fwhm))
        np.save(opj("./", "beams",filename), blm)
plt.plot(np.arange(lmax), np.ones(lmax), 'k--')
custom_lines = [Line2D([0], [0], c=colors(0.)),
                Line2D([0], [0], c=colors(0.3)),
                Line2D([0], [0], c=colors(0.5)),
                Line2D([0], [0], c=colors(0.7)),
                Line2D([0], [0], c=colors(0.9)),
                Line2D([0], [0], ls = '--', c = 'k'),
                Line2D([0], [0], c=shades(0.8)),
                Line2D([0], [0], c=shades(0.85)),
                Line2D([0], [0], c=shades(0.9)),
               ]
legend = [r'$\varepsilon = 1e-5$', r'$\varepsilon = 2e-5$',r'$\varepsilon = 5e-5$', 
          r'$\varepsilon = 1e-4$', r'$\varepsilon = 2e-4$', r'$\varepsilon = 0.$',
          r'$FWHM=5^\circ$', r'$FWHM=10^\circ$', r'$FWHM=15^\circ$']

plt.legend(custom_lines, legend, ncol=2, frameon=False, loc=(0.6, 0.15))
plt.xlim(0,100)
plt.xlabel(r'$\ell$')
plt.ylabel(r'$b_{\ell0}/b_{\ell0}^{gaussian}$')
plt.tight_layout()
plt.show()

"""
side_amp=0
side_fwhm = 1.
beam = (1.-side_amp)*np.exp(-theta**2/(2*(fwhm/2.355)**2))
beam += side_amp*np.exp(-theta**2/(2*(np.radians(side_fwhm)/2.355)**2))

blm = hp.map2alm([beam, beam, np.zeros(hp.nside2npix(nside))], lmax=lmax, mmax=lmax)

filename = 'gaus_beam'

np.save(filename, blm, allow_pickle=True)
fwhm = 46.5
side_amp = 3e-5
side_fwhm = 20
blm_main = tools.gauss_blm(fwhm, lmax, pol=False)
blm_main *= 1-side_amp
blm_main = tools.get_copol_blm(blm_main, c2_fwhm=fwhm)
blm_side = tools.gauss_blm(side_fwhm*60, lmax, pol=False)
blm_side *= side_amp
blm_side = tools.get_copol_blm(blm_side, c2_fwhm=side_fwhm*60)
blm = (np.array(blm_side)+np.array(blm_main)).tolist()
filename = "bigauss_PO_mock"
np.save(filename, blm, allow_pickle=True)
#plt.plot(np.real(blm[0][:lmax+1]))
#plt.plot(np.real(blm[0][:lmax+1] - np.array(blm_main)[0,:lmax+1]/(1-side_amp)))

#plt.show()
"""