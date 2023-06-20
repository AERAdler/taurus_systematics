import healpy as hp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import argparse as ap
opj = os.path.join
plt.switch_backend("agg")

#Polspice
import pipeline_tools

#xqml
import xqml
from xqml.xqml_utils import getstokes
from xqml.simulation import extrapolpixwin, Karcmin2var

#NaMaster
from pymaster import *

def cv(ell, cl, fsky):
    #cosmic variance
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
        return_kernel=False)

    return spice_opts

def estimator_study(nsims=100, nside=16, mask="dec_cut", dec_cut=0., 
                    sim_tag="",polspice=False, namaster=False, qml=False, 
                    lmax_est=30, binw=1):
    """
    Generate simulated skies with the planck spectrum and optionnaly estimate 
    their power spectra on a cut sky with some of the three estimators
    --------- 
    Keyword arguments
    nsims : int
        Number of sky realisations to simulate (default: 100)
    nside : int
        Resolution of sky maps (default: 16)
    mask : string
        Path to mask file or instruction to cut a band (default: dec_cut)
    dec_cut : float
        Latitudes to be cut (default: 0.)
    polspice : bool
        Estimate spectra with polspice (default: False)
    namaster : bool
        Estimate spectra with NaMaster (default: False)
    qml : bool
        Estimate spectra with xQML (default: False)
    lmax_est : int
        Maximum multipole of estimation
    binw : int

    """

    lmax_sim = 3*nside - 1

    #ell, TT, TE, EE, BB, PP
    dl_th = np.loadtxt("planckCOMr3_spectra.txt", unpack=True)
    ell = np.arange(dl_th.shape[1], dtype=int)
    dfac = ell*(ell+1)/(2*np.pi)
    cl = np.zeros((4,lmax_sim+1))
    comp=["TT", "EE", "BB", "TE"]
    #Change order to match spice output, let mono and dipole be zero
    cl[0,2:] = dl_th[1,:lmax_sim-1]/dfac[2:lmax_sim+1]#TT
    cl[1,2:] = dl_th[3,:lmax_sim-1]/dfac[2:lmax_sim+1]#EE
    cl[2,2:] = dl_th[4,:lmax_sim-1]/dfac[2:lmax_sim+1]#BB
    cl[3,2:] = dl_th[2,:lmax_sim-1]/dfac[2:lmax_sim+1]#TE

    if mask !="dec_cut":
        pseud_mask = hp.ud_grade(hp.read_map(mask), nside)
    else:
        pseud_mask = np.ones(12*nside*nside)
        tht, _ = hp.pix2ang(nside, np.arange(12*nside*nside))
        pseud_mask[np.abs(np.pi/2.-tht)<np.radians(dec_cut)]=0.

    fsky = np.sum(pseud_mask)/(12*nside*nside)
    print("fsky is: ", fsky)

    if polspice:
        spice_opts = get_default_spice_opts(lmax_est, fsky)
        spice_cl = np.zeros((nsims, 6, lmax_est+1))
        
    if namaster:
        nama_mask = mask_apodization(pseud_mask, 10.0, apotype='C1')
        nama_cl = np.zeros((nsims, 6, lmax_sim+1))
        nama_bin = NmtBin(nside, nlb = binw)

    if qml:
        comp_xqml = ["TT", "TE", "EE", "BB"]
        cl_full = np.zeros((6,dl_th.shape[1]))
        cl_full[0,2:] = dl_th[1,:-2]/dfac[2:]#TT
        cl_full[1,2:] = dl_th[3,:-2]/dfac[2:]#EE
        cl_full[2,2:] = dl_th[4,:-2]/dfac[2:]#BB
        cl_full[3,2:] = dl_th[2,:-2]/dfac[2:]#TE

        
        _, spec, _, _ = getstokes(spec=comp_xqml)

        #xQMl white noise 
        muKarcmin = 15.
        pixvar = Karcmin2var(muKarcmin*1e-6, nside)
        xqml_mask = np.ones(12*nside*nside, dtype=bool)
        xqml_mask[pseud_mask<0.5] = False
        xqml_pix = np.sum(xqml_mask)#I don't get it 
        xqml_fsky = np.mean(xqml_mask)
        varmap = np.ones((3 * xqml_pix)) * pixvar
        NoiseVar = np.diag(varmap)

        #xQML objects 
        bins = xqml.Bins.fromdeltal(2, lmax_est, binw)
        esti = xqml.xQML(xqml_mask, bins, cl_full, NA=NoiseVar, NB=NoiseVar, 
            lmax=lmax_est, fwhm=0, spec=spec)
        lb = bins.lbin
        xqml_cl = np.zeros((nsims, 4, len(lb)))

    for i in range(nsims):
        np.random.seed(i)
        simmap = hp.synfast(cl, nside=nside, new=True)
        if polspice:
            spice_cl[i] = pipeline_tools.spice(simmap, mask=pseud_mask, 
                **spice_opts)
        if namaster:
            f2 = NmtField(nama_mask, [simmap[1], simmap[2]], purify_b=True)#purify option
            f0 = NmtField(nama_mask, [simmap[0]])
            wsp = NmtWorkspace()
            wsp.compute_coupling_matrix(f0, f0, nama_bin)
            cl_tt_coupled = compute_coupled_cell(f0, f0)
            nama_cl[i,0,2:] = wsp.decouple_cell(cl_tt_coupled)[0]
            #cl_bias = deprojection_bias(f0, f2, cl)
            wsp.compute_coupling_matrix(f0, f2, nama_bin)
            cl_tpol_coupled = compute_coupled_cell(f0, f2)
            nama_cl[i,3,2:] = wsp.decouple_cell(cl_tpol_coupled)[0]
            nama_cl[i,4,2:] = wsp.decouple_cell(cl_tpol_coupled)[1]
            wsp.compute_coupling_matrix(f2, f2, nama_bin)
            cl_pol_coupled = compute_coupled_cell(f2, f2)
            nama_cl[i,1,2:] = wsp.decouple_cell(cl_pol_coupled)[0]
            nama_cl[i,2,2:] = wsp.decouple_cell(cl_pol_coupled)[3]
            nama_cl[i,5,2:] = wsp.decouple_cell(cl_pol_coupled)[1]

        if qml:
            xqml_map = simmap[:, xqml_mask]
            xqml_map_a = xqml_map + np.random.randn(3, xqml_pix) * np.sqrt(pixvar)
            xqml_map_b = xqml_map + np.random.randn(3, xqml_pix) * np.sqrt(pixvar)
            xqml_cl[i] = np.array(esti.get_spectra(xqml_map_a))

    if polspice:
        np.save(sim_tag+"_spicecl.npy", spice_cl)
    if namaster:
        np.save(sim_tag+"_namacl.npy", nama_cl)
    if qml:
        np.save(sim_tag+"_xqmlcl.npy", xqml_cl)


    #Statistics

    if polspice:
        spice_cl = np.load(sim_tag+"_spicecl.npy")
        #Bin
        if binw!=1:
            band_edges = np.arange(2, lmax_est+1, binw)
            if lmax_est -1 %binw !=0:
                band_edges = np.append(band_edges, [lmax_est])
            bcen = 0.5 * (band_edges[1:] + band_edges[:-1])
            bwidth = band_edges[1:] - band_edges[:-1]
            ell_bin = np.digitize(np.arange(2, lmax_est+1), band_edges)
            spice_bp = np.zeros((nsims, 4, bcen.size))
            for i, bi in enumerate(ell_bin):
                spice_bp[:,:, bi] += spice_cl[:,:4,2+i]
        else:
            spice_bp = spice_cl
            bcen = ell[:lmax_est+1]
        dfac_bcen = bcen*(bcen+1)/(2*np.pi)
        spice_median = np.median(spice_bp, axis=0)[:4]
        spice_16 = np.percentile(spice_bp, 15.865, axis=0)[:4]
        spice_84 = np.percentile(spice_bp, 84.135, axis=0)[:4]
        spice_cl1s = np.abs([spice_median - spice_16, spice_84 - spice_median])
        spice_fl = spice_median /cl[:,:lmax_est+1]
        spice_fl1s = np.abs(spice_cl1s/cl[:,:lmax_est+1])

        polspicedict = dict(linestyle="none", color="tab:blue", 
            label="Polspice")

    if namaster:
        nama_cl = np.load(sim_tag+"_namacl.npy")
        namasterdict =  dict(linestyle="none", color="tab:green", 
            label="Namaster")
        nama_median = np.median(nama_cl, axis=0)[:4]
        nama_16 = np.percentile(nama_cl, 15.865, axis=0)[:4]
        nama_84 = np.percentile(nama_cl, 84.135, axis=0)[:4]
        nama_cl1s = np.abs([nama_median - nama_16, nama_84 - nama_median])
        nama_fl = nama_median /cl[:,:lmax_sim+1]
        nama_fl1s = np.abs(nama_cl1s/cl[:,:lmax_sim+1])
        nama_bcen = ell[:lmax_sim+1]
        nama_dfac = nama_bcen*(nama_bcen+1)/(2*np.pi)


    if qml:
        xqml_cl = np.load(sim_tag+"_xqmlcl.npy")
        xqml_median = np.median(xqml_cl, axis=0)
        xqml_16 = np.percentile(xqml_cl, 15.865, axis=0)
        xqml_84 = np.percentile(xqml_cl, 84.135, axis=0)
        xqml_cl1s = np.abs([xqml_median - xqml_16, xqml_84 - xqml_median])
        xqml_fl = xqml_median/cl[:,2:lmax_est+1]
        xqml_fl1s = np.abs(xqml_cl1s/cl[:,2:lmax_est+1])
        xqmldict = dict(linestyle="none", color="tab:orange", label="xQML")
        dfac_bin = lb*(lb+1)/(2*np.pi)

    #Plotting
    plot_order = [0,3,1,2]#I want TT,TE,EE,BB as plot order
    fig1, axs1 = plt.subplots(2,2, sharex=True, figsize=(11,5))
    fig1.text(0.45, 0.02, r"Multipole $\ell$", fontsize=12)

    for k, ax in enumerate(axs1.flat):
        j = plot_order[k]
        ax.set_xlim(0.,lmax_est+1)
        ax.set_ylabel(r"$D_\ell^{{{}}}$".format(comp[j]))
        ax.plot(ell[:lmax_sim+1], dfac[:lmax_sim+1]*cl[j], "k--", 
            label="Planck best-fit spectrum")
        cstd = cv(ell[:lmax_sim+1], cl[j], fsky)
        ax.plot(ell[:lmax_sim+1], dfac[:lmax_sim+1]*(cl[j]+cstd), ls=":", 
            color="tab:purple", label="Cosmic variance")
        ax.plot(ell[:lmax_sim+1], dfac[:lmax_sim+1]*(cl[j]-cstd), ls=":", 
            color="tab:purple")

        if polspice:
            ax.errorbar(bcen+0.1, dfac_bcen*spice_median[j], 
                        marker=".", yerr=dfac_bcen*spice_cl1s[:,j], 
                        **polspicedict)
        if namaster:
            ax.errorbar(nama_bcen, nama_dfac*nama_median[j], marker=".",
                        yerr=nama_dfac*nama_cl1s[:,j], **namasterdict)
        if qml:
            ax.errorbar(lb-0.1, dfac_bin*xqml_median[j], marker=".",
                        yerr = dfac_bin*xqml_cl1s[:,j], **xqmldict)

    axis_max = np.amax(dfac[:lmax_sim+1]*(cl+cv(ell[:lmax_sim+1], cl, fsky)),
                                                                   axis=1)
    axis_min = np.amin(dfac[:lmax_sim+1]*(cl-cv(ell[:lmax_sim+1], cl, fsky)), 
                                                                      axis=1)
    #axs1[0,0].set_ylim(0., 1.1*axis_max[0])
    #axs1[0,1].set_ylim(1.1*axis_min[3], 1.1*axis_max[3])
    #axs1[1,0].set_ylim(0., 1.1*axis_max[1])
    #axs1[1,1].set_ylim(1.1*axis_min[2], 1.1*axis_max[2])
    axs1[1,0].legend(frameon=False)
    fig1.suptitle(r"$D_\ell$ comparaison")
    plt.savefig(sim_tag+"_dl.png", dpi=200)


    fig2, axs2 = plt.subplots(2,2, sharex=True, figsize=(11,5))
    fig2.text(0.45, 0.02, r"Multipole $\ell$", fontsize=12)

    for k, ax in enumerate(axs2.flat):
        j = plot_order[k]#I want TT,TE,EE,BB as plot order
        ax.set_xlim(1, lmax_est+1)
        ax.set_ylim(0,4)
        if j==2:
            ax.set_ylim(-149,151)
        ax.set_ylabel(r"$F_\ell^{{{}}}$".format(comp[j]))
        ax.plot(np.arange(0, 50), np.ones(50), "k--")
        ax.plot(1+ 2./(2*ell+1)/fsky, c="tab:purple", ls=":", 
                label="Cosmic variance")
        ax.plot(1- 2./(2*ell+1)/fsky, c="tab:purple", ls=":")
        if polspice:
            ax.errorbar(bcen+0.1, spice_fl[j], yerr=spice_fl1s[:,j], 
                        marker=".", **polspicedict)
        if namaster:
            ax.errorbar(nama_bcen, nama_fl[j], marker=".", yerr=nama_fl1s[:,j],
                        **namasterdict)
        if qml:
            ax.errorbar(lb-0.1, xqml_fl[j], yerr=xqml_fl1s[:,j], 
                        marker=".", **xqmldict)


    axs2[0,0].legend(frameon=False, ncol=2)
    fig2.suptitle(r"Transfer function comparaison")
    plt.savefig(sim_tag+"_fl.png", dpi=200)

def main():
    # Parser
    parser = ap.ArgumentParser(formatter_class=ap.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--nsims", type=int, default=100, 
        help="Number of maps to simulate")
    parser.add_argument("--nside", type=int, default=16, 
        help="NSIDE of the simulated maps")
    parser.add_argument("--mask", type=str, default="dec_cut", 
        help="Mask, can be a cut through the middle or a path to a fits file")
    parser.add_argument("--dec_cut", type=float, default=0., 
        help="If using a middle cut, range of declinations excluded")
    parser.add_argument("--sim_tag", type=str, default="",
        help="Name of the current run")

    parser.add_argument("--polspice", action="store_true", 
        help="Estimate the spectra with polspice")
    parser.add_argument("--namaster", action="store_true", 
        help="Estimate the spectra with namaster")
    parser.add_argument("--qml", action="store_true", 
        help="Estimate the spectra with xqml")
    parser.add_argument("--lmax_est", type=int, default=30, 
        help="lmax for the estimators")
    parser.add_argument("--binw", type=int, default=1,
        help="Bin width for estimators")



    args = vars(parser.parse_args())
    estimator_study(**args)

if __name__ == "__main__":
    main()


