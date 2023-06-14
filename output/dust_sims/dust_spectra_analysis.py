#dust_spectra_analysis.py

import healpy as hp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import argparse as ap
opj = os.path.join
plt.switch_backend("agg")

#NaMaster
from pymaster import *

def cv(ell, cl, fsky):
    #cosmic variance
    return 2./(2*ell+1)/fsky*cl

def namaster_estimation(maps, mask):
    """
    Generate estimated power spectra with namaster
    --------- 
    Arguments
    maps : healpix sky maps (I,Q,U)
    mask : mask to apply


    """
    nside = int((len(mask)/12)**.5)
    lmax = 3*nside-1
    nama_mask = mask_apodization(mask, 10.0, apotype='C1')
    nama_cl = np.zeros((nsims, 6, lmax+1))
    nama_bin = NmtBin(nside, nlb = 1.)
    f2 = NmtField(nama_mask, [maps[1], maps[2]], purify_b=True)#purify option
    f0 = NmtField(nama_mask, [maps[0]])
    wsp = NmtWorkspace()
    wsp.compute_coupling_matrix(f0, f0, nama_bin)
    cl_tt_coupled = compute_coupled_cell(f0, f0)
    nama_cl[i,0,2:] = wsp.decouple_cell(cl_tt_coupled)[0]
    wsp.compute_coupling_matrix(f0, f2, nama_bin)
    cl_tpol_coupled = compute_coupled_cell(f0, f2)
    nama_cl[i,3,2:] = wsp.decouple_cell(cl_tpol_coupled)[0]
    nama_cl[i,4,2:] = wsp.decouple_cell(cl_tpol_coupled)[1]
    wsp.compute_coupling_matrix(f2, f2, nama_bin)
    cl_pol_coupled = compute_coupled_cell(f2, f2)
    nama_cl[i,1,2:] = wsp.decouple_cell(cl_pol_coupled)[0]
    nama_cl[i,2,2:] = wsp.decouple_cell(cl_pol_coupled)[3]
    nama_cl[i,5,2:] = wsp.decouple_cell(cl_pol_coupled)[1]

    return nama_cl



def analysis(analyzis_dir, sim_tag, ideal_map=None, input_map=None,
             cal=None, mask_file=None, nside_out=256, lmax=767, 
             l1=50, l2=200, fwhm=30., plot=False, label=None):
    """
    Function to analyze simulation output 
    Arguments
    ---------
    analyzis_dir : string
        Path to the directory in which input maps are located, and where 
        the output spectra and maps directories are.
    sim_tag : string
        Tag of the simulated map that is getting analyzed.

    Keyword arguments
    -----------------
    ideal_map : string
        Name of an ideal simulation map the sim_tag map gets 
        differentiated against. (default: None)
    input_map : string
        Name of an input map the sim_tag map gets 
        differentiated against. (default: None)
    cal : int
        Whether to recalibrate the scanned map vs ideal and/or input.
        A gain is computed as the average ratio between the ideal/input 
        spectrum and the sim_tag spectrum. The sim_tag map is scaled by 
        sqrt(gain). 0 for TT, 1 for EE etc... (default: None)
    mask : string
        Mask file to apply to the maps (default: None)
    nside_out : int
        Healpix NSIDE of the output map. (default: 256)
    lmax : int
        Maximum l when making power spectra. (default: 767)
    l1 : int
        Lower edge of the calibration window. (default: 50)
    l2 : int
        Higher edge of the calibration window. (default: 200)
    fwhm : float
        fwhm of the beam in arcmin. (default: 30.)
    plot : bool
        Whether to plot the spectra we have 
    label : string
        What the non-ideality is called on the legend
    """

    mapfilename = "maps_"+sim_tag+".fits"
    fstrs = ["TT", "EE", "BB", "TE", "TB", "EB"]
    spectra_dir = opj(analyzis_dir, "spectra")
    maps = tools.read_map(opj(analyzis_dir, mapfilename), field=None, fill=np.nan)
    hits = tools.read_map(opj(analyzis_dir, mapfilename.replace("maps_", "hits_")))
    cond = tools.read_map(opj(analyzis_dir, mapfilename.replace("maps_", "cond_")))
    maps = hp.ud_grade(maps, nside_out)
    hits = hp.ud_grade(hits, nside_out)
    cond = hp.ud_grade(cond, nside_out)

    #Condition-number based mask
    mask = np.ones(12*nside_out**2)
    mask[cond>3]=0.
    mask[hits==0]=0.

    if mask_file:
        custom_mask = hp.ud_grade(tools.read_map(opj(analyzis_dir, mask)), 
                                  nside_out)
        mask *=custom_mask

    fsky = np.sum(mask) / 12*nside_out**2
    print("fsky: {:.3f}".format(fsky))
    cl = namaster_estimation(maps, mask)
    bl = hp.gauss_beam(fwhm=np.radians(fwhm/60.), lmax=len(cl[1])-1)
    cl = cl/bl**2
    np.save(opj(spectra_dir, "{}_cl.npy".format(sim_tag)), cl)
    gain = 1.
    
    #Versus ideal
    if ideal_map:

        ideal_maps = tools.read_map(opj(analyzis_dir, ideal_map),
            field=None, fill=np.nan)
        ideal_maps = hp.ud_grade(ideal_maps, nside_out)
        cl_ideal = namaster_estimation(ideal_maps, mask)
        cl_ideal = cl_ideal/bl**2
        np.save(opj(analyzis_dir, "spectra", 
                   "{}_cl.npy".format(sim_tag)), cl_ideal)
            #Compute Cls for the smoothed map, deconvolve
        if cal is not None:
            gain = np.average(cl_ideal[cal, l1:l2]/cl[cal, l1:l2])
            print("{} gain for map {} versus ideal is: {:.3f}".format(
                fstrs[cal], sim_tag, gain)) 

        #Should difference maps be gain_corrected?
        diff_ideal = maps*np.sqrt(gain) - ideal_maps
        diff_ideal_cl = namaster_estimation(diff_ideal, mask)
        diff_ideal_cl = diff_ideal_cl/bl**2
        if cal is not None:
            np.save(opj(spectra_dir, "{}_diffideal_cl_cal{}.npy".format(
                 sim_tag, fstrs[cal])), diff_ideal_cl)
        else:
            np.save(opj(spectra_dir, "{}_diffideal_cl_calno.npy".format(
                 sim_tag)), diff_ideal_cl)
    gain = 1.
    #Versus input
    if input_map:
        input_maps = tools.read_map(opj(analyzis_dir, input_map),
            field=None, fill=np.nan)
        input_maps = hp.ud_grade(input_maps, nside_out)
        #Calibration
        cl_input= namaster_estimation(input_maps, mask)
        cl_input = cl_input/bl**2
        #Compute Cls for the smoothed map, deconvolve
        if cal is not None:
            gain = np.average(cl_input[cal, l1:l2]/cl[cal, l1:l2])
            print("{} gain for map {} versus input is: {:.3f}".format(
                  fstrs[cal], sim_tag, gain))

        diff_input = maps*np.sqrt(gain) - input_maps
        for diffi in diff_input:
            diffi[~mask] = np.nan
        diff_input_cl = tools.spice(diff_input, mask=mask, **spice_opts2use)
        diff_input_cl = diff_input_cl/bl**2
        if cal is not None:
            np.save(opj(spectra_dir, "{}_diff_input_cl_cal{}.npy".format(
                 sim_tag, fstrs[cal])), diff_input_cl)
        else:
            np.save(opj(spectra_dir, "{}_diff_input_cl_nocal.npy".format(
                 sim_tag)), diff_input_cl)

    if not plot:
        return 

    img_dir = opj(analyzis_dir, "images")
    cmap4maps = cm.RdBu_r
    cmap4maps.set_under("w")
    cmap4maps.set_bad("black", 0.5)
    cmap = plt.get_cmap("tab10")
    xlmax = 300

    for f in range(6):
        plt.figure(f)
        ell = np.arange(len(cl[f]))
        #Truncate to start in l=4
        ell = ell[5:]
        plt.plot(ell, ell*(ell+1)/(2*np.pi)*gain*cl[f][5:], label=label)
        #Plot ideal spectrum
        if ideal_map:
            plt.plot(ell, ell*(ell+1)/(2*np.pi)*cl_ideal[f][5:], label="Ideal", 
                ls="-.")

            #Plot the difference spectra on extra plots
            plt.figure(f+6)
            plt.plot(ell, ell*(ell+1)/(2*np.pi)*diff_ideal_cl[f][5:], 
                label="Residuals of "+str(label)+" vs ideal")
            plt.figure(f)

        if input_map:
        #plotting input spectra
            plt.plot(ell, ell*(ell+1)/(2*np.pi)*cl_input[f][5:], label="Input",
                ls="--", color="k")
            plt.figure(f+6)
            plt.plot(ell, ell*(ell+1)/(2*np.pi)*diff_input_cl[f][5:], 
                label="Residuals of "+str(label)+" vs input", ls="--")
            plt.figure(f)

        #Labeling plots
        plt.legend(loc=2, frameon=False)
        plt.xlabel(r"Multipole, $\ell$")
        plt.ylabel(r"$D_\ell^{{{}}}$".format(fstrs[f]))
        plt.xlim([0, xlmax])

        if f !=12:#Let be for now...
            autoscale_y(plt.gca())
            plt.xlim([1,xlmax])

        img_name = sim_tag+"_"+"spec{}.png".format(fstrs[f])
        plt.savefig(opj(analyzis_dir, img_dir, img_name),
                bbox_inches="tight", dpi=300)
        plt.close()

        if input_map or ideal_map:
            #Add BB contours
            plt.figure(f+6)

            if f == 2:
                #plot_bb(outdir)
                plt.gca().set_yscale("log")
                plt.gca().set_xscale("log")
                plt.xlim([1,xlmax])
                plt.ylim([1e-5, 1e0])

            plt.legend(loc=2, frameon=False)
            plt.xlabel(r"Multipole, $\ell$")
            plt.ylabel(r"$D_\ell^{{{}}}$".format(fstrs[f]))
            plt.xlim([1, xlmax])
            if f != 2:
                autoscale_y(plt.gca())
            img_name = sim_tag+"_"+"dspec{}.png".format(fstrs[f])
            plt.savefig(opj(analyzis_dir, img_dir, img_name),
                bbox_inches="tight", dpi=300)
            plt.close()

    return


def main():

    parser = ap.ArgumentParser(formatter_class=ap.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ideal_map", type=str, action="store", default=None, 
        help="ideal map for comparaison", dest="ideal_map") 
    parser.add_argument("--input_map", type=str, action="store", default=None, 
        help="input map for comparaison", dest="input_map") 
    parser.add_argument("--mask", type=str, default=None, action="store",
        help="Mask for analysis", dest="mask")
    parser.add_argument("--calibrate", type=int, default=None, action="store", 
        dest="calibrate", help="Calibrate vs ideal or input") 
    parser.add_argument("--plot", default=False, action="store_true", 
        dest="plot", help="Plot spectra") 
    parser.add_argument("--label", type=str, action="store", default=None, 
    help="Label of non-ideality on plots", dest="label") 
    args = parser.parse_args()

    analyzis_dir = "./"
    analysis(
            analyzis_dir = analyzis_dir, 
            sim_tag = args.sim_tag, 
            ideal_map = args.ideal_map, 
            input_map = args.input_map,
            cal = args.calibrate, 
            mask_file = args.mask,
            nside_out = 256, 
            lmax = 767,
            fwhm = args.fwhm, 
            l1 = 50, 
            l2 = 100,
            plot = args.plot,
            label = args.label)

    return

if __name__ == "__main__":
    main()
