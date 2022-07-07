#SAT-like scan with ground

import numpy as np
import healpy as hp
import argparse as ap
from beamconv import ScanStrategy, Beam
from beamconv import tools as beam_tools
import pipeline_tools as tools
import qpoint as qp
import matplotlib.pyplot as plt
import pickle
import time
import datetime
import os 
opj = os.path.join
plt.switch_backend("agg")
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
from ground_tools import template_from_position

def run_sim(simname, sky_alm,
            basedir = opj("/","mn", "stornext", "u3", "aeadler", "ssn"),
            beamdir = "beams", outdir = opj("output", "maps"),
            mlen= 24*60*60,  sample_rate = 119.1, t0 = 1546300800, 
            npairs=None, create_fpu=False, fov=2.0, beam_files=None,
            no_pairs=False, btype="Gaussian", fwhm=43., deconv_q=True, 
            lmax=1000, mmax=4, pol_only=False, no_pol=False, add_ghosts=False, 
            ghost_amp=0.01, scan_type="atacama", el0=35., az0=0., freq=150., 
            ground = None, filter_highpass=False, w_c=None, filter_m=1,
            hwp_mode=None, hwp_model="HWP_only", load_mueller=False, varphi=0.0, 
            hfreq=1.0, hstepf=1/(3*60*60), filter_4fhwp=False, nside_spin=1024, 
            nside_out=512, verbose=1, balloon_track = None, killfrac=0., 
            seed=25, preview_pointing=False, comm=None, **kwargs):
    """
    Run beamconv simulations, either ground-based or balloon-borne. 
    Creates maps.

    Arguments
    ---------
    simname : string
        Name of the simulation and of the output maps
    sky_alm : array-like
        alm for the input sky

    Keyword Arguments
    -----------------
    basedir : string
        Path to directory where beamdir and outdir are located
    beamdir : string
        Directory in basedir containing the listed beams
    outdir : string
        Output directory for maps
    mlen : float
        Mission duration in seconds
    sample_rate : float
        Number of samples per second (default: 50.01)
    t0 : int 
        Mission start-time, Unix time (default: 1546300800 for Jan 1, 2019)
    npairs : int
        Number of detector pairs
    create_fpu : bool
        Create square focal plane (default : False)
    fov : float
        Field of view on square focal plane, in degrees (default: 2.0)
    beam_files : list
        List containing the name of the beams/the one beam to be loaded 
        (default : None)
    no_pairs : bool
        Only have A detectors (default : False)
    btype : string
        Beam type for the autogenerated FP: Gaussian/EG/PO
        (default: Gaussian)
    fwhm : float 
        Beam FWHM for the autogenerated Gaussian FP in arcmin (default: 43)
    deconv_q : bool
        Apply the 2*sqrt(pi/2l+1) factor to the loaded blms (default: True)
    lmax : int
        Maximum l mode of beam and sky decomposition
    mmax : int
        Maximum m mode of beam decomposition
    pol_only : bool
        Measure only polarised component (default : False)
    no_pol : bool
        Measure only Stokes I (default : False)
    add_ghosts : bool
        Add ghost reflections to the main beam (default : False)  
    ghost_amp : float
        Amplitude of the ghosts (default : 0.01)
    el0 : float
        Boresight starting elevation in degrees (default: 35.)
    az0 : float
        Boresight starting azimuth in degrees (default: 0.)
    freq : float
        Detector frequency, in Hz (default: 1.0e11)
    ground_alm : array-like
        alm for the ground template
    filter_highpass : bool
        High-pass filter the data. In the absence of a w_c parameter, 
        defaults to removing chunk average (default: False)
    w_c : float
        Filter frequency for the high-pass filter (default: None)
    filter_m : int
        Order of the highpass Butterworh filter (default: 1)
    hwp_mode : string
        Type of HWP motion (None, stepped, continuous)
    hwp_model : string
        Pre-included HWP model selected (default: HWP_only)
    varphi : float
        HWP angle correction to apply due to multi-layer phase offset 
        (default: 0.0)
    hfreq : float
        HWP rotation frequency, cycles per second (default: 1.0)
    hstepf : float
        HWP stepping frequency, Hz^-1 (default: 3hrs^-1)
    filter_4fhwp : bool
        Only use TOD modes modulated at 4 x the HWP frequency.
        Only allowed with spinning HWP. (default: False)
    nside_spin : int
        Healpix NSIDE of spin maps (default: 1024)
    nside_out : int
        Healpix NSIDE of resulting sky map (default: 512)
    verbose : int
        How much to print while running beamconv (default: 1)
    balloon_track : string
        Path to file with balloon position over time (default: None)
    killfrac : float
        Fraction of detectors to randomly kill (default: 0.)
    seed : int
        Seed for random number generator (default: 25)
    preview_pointing : bool
        (default: False)
    comm : MPI communicator
    """
    np.random.seed(seed)

    ndays = int(mlen/(24*60*60))
    track = np.loadtxt(opj(basedir, balloon_track))
    co_added_maps = np.zeros((3,hp.nside2npix(nside_out)))
    co_added_cond = np.zeros(hp.nside2npix(nside_out))
    co_added_hits = np.zeros((hp.nside2npix(nside_out)))
    days_visited = np.zeros(hp.nside2npix(nside_out))
    for day in range(ndays):
        ctime0 = t0+day*24*60*60
        track_idx = np.argmin(np.absolute(track[:,0]-ctime0))
        lat = track[track_idx,1]
        lon = track[track_idx,2]
        if rank==0:
            print(track[track_idx,0],lat, lon)
        h = 35000+200*np.random.normal()
        ymd = datetime.date.fromtimestamp(ctime0).strftime("%Y%m%d")

        scan = ScanStrategy(24*60*60, sample_rate=sample_rate, 
                            lat=lat, lon=lon, ctime0=ctime0)
        #reverse scan direction every day
        scan_opts = dict(scan_speed=30.*int(2*(day%2-.5)), 
                         use_strictly_az=True,
                         q_bore_func=scan.strictly_az, 
                         ctime_kwargs=dict(),
                         q_bore_kwargs=dict(el0=el0, az0=az0),
                         max_spin=2,
                         nside_spin=256,
                         preview_pointing=False,
                         interp = True, 
                         filter_highpass = filter_highpass)
            

        if create_fpu:#A square focal plane
            beam_opts = dict(lmax=lmax, fwhm=fwhm, btype=btype, 
                             sensitive_freq=freq, deconv_q=deconv_q)
            nfloor = int(np.floor(np.sqrt(npairs)))
            if btype=="PO":
                beam_opts["po_file"] = beam_files#It's just the one file, actually
            scan.create_focal_plane(nrow=nfloor, ncol=nfloor, fov=fov,
                    **beam_opts)

        else:
            scan.load_focal_plane(beamdir, btype=btype, no_pairs=no_pairs, 
                                  sensitive_freq=freq, file_names=beam_files)


        if hwp_model != "HWP_only":
            for beami in scan.beams:
                beami[0].set_hwp_mueller(model_name=hwp_model)
                beami[1].set_hwp_mueller(model_name=hwp_model) 
        scan.partition_mission(chunksize=int(sample_rate*3600))
        scan.allocate_maps(nside=512)
        scan.set_hwp_mod(mode=hwp_mode, freq=1.)
        if filter_highpass and (w_c is not None):
            scan.set_filter_dict(w_c, m=filter_m)

        if ground:
            if rank==0:
                world_map = hp.read_map(opj(basedir,"ground_input",
                            "SSMIS","SSMIS-{}-91H.fits".format(ymd)))
                ground_template = template_from_position(world_map, 
                    lat, lon, h, nside_out=4096, cmb=False, freq=95.,
                                                     frac_bwidth=.2)
                ground_alm = hp.map2alm([ground_template, 
                                    np.zeros_like(ground_template), 
                                    np.zeros_like(ground_template)], 
                                    lmax = lmax)
                ground_alm = hp.smoothalm(ground_alm, fwhm = np.radians(1.))
            else:
                ground_alm = np.zeros((3,hp.Alm.getsize(lmax=lmax)), dtype=complex)
            comm.Bcast(ground_alm, root=0)

            scan.scan_instrument_mpi(sky_alm, ground_alm=ground_alm, **scan_opts)
        
        else:
            scan.scan_instrument_mpi(sky_alm, **scan_opts)

        maps, cond, proj = scan.solve_for_map(return_proj=True)
        if scan.mpi_rank == 0:
            hp.write_map(opj(basedir, outdir,
                 "maps_"+simname+"_{}.fits".format(ymd)), maps)
            hp.write_map(opj(basedir, outdir, 
                "cond_"+simname+"_{}.fits".format(ymd)), cond)
            hp.write_map(opj(basedir, outdir, 
                "hits_"+simname+"_{}.fits".format(ymd)), proj[0])
            co_added_maps[:, maps[0]!=hp.UNSEEN] += maps[:, maps[0]!=hp.UNSEEN]
            co_added_hits += proj[0]
            co_added_cond[maps[0]!=hp.UNSEEN] = np.minimum(
                cond[maps[0]!=hp.UNSEEN], co_added_cond[maps[0]!=hp.UNSEEN])
            days_visited[maps[0]!=hp.UNSEEN] += 1
    if scan.mpi_rank==0:

        co_added_maps[:,days_visited!=0] /= days_visited[days_visited!=0]
        hp.write_map(opj(basedir, outdir, "maps_"+simname+"_coadd.fits"),
                 co_added_maps)
        hp.write_map(opj(basedir, outdir, "hits_"+simname+"_coadd.fits"),
                 co_added_hits)
        hp.write_map(opj(basedir, outdir, "cond_"+simname+"_coadd.fits"),
                 co_added_cond)
    return


def parse_beams(beam_files, beamdir, ss_obj=None, lmax=2000, 
                stitch_wide=False, plot=False):
    """
    Load GRASP output, convert into blms and save.

    Arguments
    ---------
    beam_files : array-like
        List of files containing GRASP output, absolute path. Format should
        be such that <filename>_<x>.pkl files exist (x = prop,
        fields, (optionally: eg). If shaped as (...,2) second beam
        file is intepreted as sidelobe that will be stitched to
        main beam.
    beamdir: str
        Path to beam directory where .npy and .pkl beam files
        will be stored, and where grasp files are located.
        Filenames will be equal to input.

    Keyword Arguments
    -----------------
    ss_obj : ScanStrategy object
        Used for MPI. If None, create new instance
        (default: None)
    lmax : int
        lmax for blms (default : 2000)
    stitch_wide : bool
        Whether to add a sidelobe (from the pickle file) to the main beam
        (default: False)
    plot : bool
        Produce plots of the Stokes beams (default: False)
    Notes
    -----
    GRASP output e fields are converted to HEALPix pixels, apodized,
    then spin-0 blm"s are synthesized, which are converted
    to spin-0 and spin-\pm2 by convolution in harmonic space.

    Will create A and B orthogonal pairs
    """

    # hardcoded
    nside_blm = 2048
    apodize = True # Note, apodize doesnt apply in case of sidelobes
    polang_a = 0
    polang_b = 90
    lmax_big = 2000

    # scatter files
    if isinstance(ss_obj, ScanStrategy):
        sat = ss_obj
    else:
        sat = ScanStrategy(sample_rate=10., num_samples=10)

    # perhaps pair all beam files with the appropriate wide beam?

    beam_files = np.asarray(beam_files)
    beam_files_loc = beam_files[sat.mpi_rank::sat.mpi_size]

    num_beams = beam_files_loc.shape[0]
    for bidx in xrange(num_beams):

        beam_file = beam_files_loc[bidx]

        print("Rank = {:03d} | bidx = {:03d} | filename = {}".\
            format(sat.mpi_rank, bidx, beam_file))

        # fields
        pkl_file = open(opj(beamdir, beam_file +"_fields.pkl"), "rb")
        fields = pickle.load(pkl_file)
        pkl_file.close()
        e_co = fields["e_co"]
        e_cx = fields["e_cx"]
        invert_flag=False
        #Sanity check that we have the main field in the right order
        if np.sum(np.absolute(e_co))<np.sum(np.absolute(e_cx)):
            e_co = fields["e_cx"]
            e_cx = fields["e_co"]
            print("Inverted eco and ecx!")
            invert_flag=True

        cr = fields["cr"] # [azmin, elmin, azmax, elmax]
        d_az = cr[2] - cr[0]
        d_el = cr[3] - cr[1]

        if stitch_wide:
            e_co_wide = fields["e_co_sl"]
            e_cx_wide = fields["e_cx_sl"]
            if invert_flag:
                e_co_wide = fields["e_cx_sl"]
                e_cx_wide = fields["e_co_sl"]

            cr_wide = fields["cr_sl"] # [azmin, elmin, azmax, elmax]
            d_az_wide = cr_wide[2] - cr_wide[0]
            d_el_wide = cr_wide[3] - cr_wide[1]

        else:
            e_co_wide = None
            e_cx_wide = None
            d_az_wide = None
            d_el_wide = None

        stokes = tools.e2iqu(e_co, e_cx, d_az, d_el, vpol=False,
            basis="grasp",
            nside_out=nside_blm,
            e_co_wide=e_co_wide,
            e_cross_wide=e_cx_wide,
            delta_az_wide=d_az_wide,
            delta_el_wide=d_el_wide,
            apodize=apodize)
        if plot:
            maxI = 10*np.log10(np.amax(stokes[0]))
            maxQ = np.amax(stokes[1])
            maxU = np.amax(stokes[2])
            hp.orthview(10*np.log10(stokes[0]), rot=[0,90,0], half_sky=True,
                       max=maxI, min=maxI-80.)
            plt.savefig(opj(beamdir, beam_file+"_I.png") ,dpi=1000)
            hp.orthview(stokes[1], rot=[0,90,0], half_sky=True)
            plt.savefig(opj(beamdir, beam_file+"_Q.png") ,dpi=1000)
            hp.orthview(stokes[2], rot=[0,90,0], half_sky=True)
            plt.savefig(opj(beamdir, beam_file+"_U.png") ,dpi=1000)
            hp.write_map(opj(beamdir, beam_file+"_Stokes.fits"), stokes)

        blm_stokes = hp.map2alm(stokes, lmax_big, pol=False)
        blm_stokes = beam_tools.trunc_alm(blm_stokes, lmax_new=lmax)
        blmm2, blmp2 = beam_tools.get_pol_beam(blm_stokes[1], blm_stokes[2])
        blm  = blm_stokes[0]
        blm = np.array([blm, blmm2, blmp2], dtype=np.complex128)

        # save npy file
        po_file = opj(beamdir, beam_file+".npy")
        np.save(po_file, blm)

        # set common opts
        #fwhm = prop["fwhm"] * 60 # arcmin
        az_off = fields["az_off"]
        el_off = fields["el_off"]
        beam_opts = dict(lmax=lmax, deconv_q=True, normalize=True,
                         po_file=po_file, cross_pol=True, btype="PO",
                         az=az_off, el=el_off)#fwhm=fwhm)
        #beam_name = 
        # set A and B specifics
        a_opts = dict(polang=polang_a, pol="A",
                      name=beam_file+"_A")
        b_opts = dict(polang=polang_b, pol="B",
                      name=beam_file+"_B")

        a_opts.update(beam_opts)
        b_opts.update(beam_opts)

        # store in pickle file
        with open(opj(beamdir, beam_file+".pkl"), "wb") as handle:
            pickle.dump([a_opts, b_opts], handle, 
                        protocol=pickle.HIGHEST_PROTOCOL)

    sat.barrier()


def main():


    #**************************
    # Process commandline input
    #**************************
    parser = ap.ArgumentParser(formatter_class=ap.ArgumentDefaultsHelpFormatter)
    # I/O
    parser.add_argument("--system", action="store", dest="system", type=str,
        default="OwlAEA")

    
    #Beams
    parser.add_argument("--beamdir", type=str, dest="beamdir", default=None,
        help="Path to beamdir, overrides system beam dir")
    parser.add_argument("--beam_file", action="store", dest="beam_file", type=str,
        default=None, help="Text file listing beams to use")
    parser.add_argument("--beam_lmax", action="store", dest="beam_lmax",
        default=2000, type=int, help="Maximum lmax for beam decomposition")
    parser.add_argument("--beam_type", type=str, dest="btype", default="Gaussian",
        help="Input beam type: [Gaussian, PO]")
    parser.add_argument("--grasp", action="store_true", dest="grasp", 
        default=False, help="The beams come as pickles of grasp grids and cuts")
    parser.add_argument("--stitch_wide", action="store_true", dest="stitch_wide",
        default=False, help="stitch wide GRASP cuts to main beam files")
    parser.add_argument("--plot_beam", action="store_true", dest="plot_beam",
        default=False, help="Plot the Stokes beams before converting to blm")

    #Beams, beamconv side
    parser.add_argument("--fwhm", action="store", dest="fwhm",
        default=30.0, type=float, help="Beam FWHM in arcmin")
    parser.add_argument("--freq", action="store", dest="freq",
        default=150, type=float, help="Beam frequency")
    parser.add_argument("--add_ghosts", action="store_true", dest="add_ghosts",
        default=False)
    parser.add_argument("--ghost_amp", action="store_true", dest="ghost_amp",
        default=False)
    parser.add_argument("--lmax", action="store", dest="lmax",
        default=1000, type=int)
    parser.add_argument("--mmax", action="store", dest="mmax",
        default=4, type=int)
    parser.add_argument("--deconv_q", action="store_false", dest="deconv_q")

    #Beamconv
    parser.add_argument("--sim_tag", action="store", dest="sim_tag", type=str,
        default="test", help="Identifier for simulation name")
    parser.add_argument("--quiet", action="store_const", dest="verbose",
        const=0, default=1, help="less print statements")
    parser.add_argument("--operation", action="store", dest="operation",
        default="run", type=str, help="run, analysis")
    parser.add_argument("--seed", action="store", dest="seed", type=float,
        default=25)
    parser.add_argument("--el0", action="store", dest="el0", type=float,
        default=50., help="Starting elevation (ballon scan)")
    parser.add_argument("--az0", action="store", dest="az0", type=float,
        default=0., help="Starting azimuth, balloon scan")
 
    #Beamconv timing
    parser.add_argument("--t0", action="store", dest="t0",
        default=1546300800, type=float)
    parser.add_argument("--days", action="store", dest="days",
        default=1, type=int)
    parser.add_argument("--sample_rate", action="store", dest="sample_rate",
        default=50.01, type=float)

    #Beamconv fpu
    parser.add_argument("--fov", action="store", dest="fov",
        default=2.0, type=float, help="Field of view in degrees")
    parser.add_argument("--create_fpu", action="store_true", dest="create_fpu",
        default=False)
    parser.add_argument("--npairs", action="store", dest="npairs",
        default=1, type=int, help="Number of beams in fpu")
    parser.add_argument("--no_pairs", action="store_true", dest="no_pairs",
        default=False)
    parser.add_argument("--killfrac", action="store", dest="killfrac",
        default=0., type=float, help="Randomly kill fraction of beams")

    #Map parameters
    parser.add_argument("--nside_spin", action="store", dest="nside_spin",
        default=1024, type=int)
    parser.add_argument("--nside_out", action="store", dest="nside_out",
        default=512, type=int)

    # HWP-related
    parser.add_argument("--hwp_mode", action="store", dest="hwp_mode", type=str,
        default=None)
    parser.add_argument("--hwp_model", action="store", dest="hwp_model", type=str,
        default="ideal")
    parser.add_argument("--hwp_phase", action="store", dest="varphi",
        type=float, default=0.)
    parser.add_argument("--hfreq", action="store", dest="hfreq", type=float,
        default=1.0)
    parser.add_argument("--hstepf", action="store", dest="hstepf", type=float,
        default=1./(3*3600))
    parser.add_argument("--filter_4fhwp", action="store_true", dest="filter_4fhwp",
        default=False)

    # Map arguments
    parser.add_argument("--alm_type", action="store", dest="alm_type", type=str,
        default="synfast")
    parser.add_argument("--sky_map", type=str, default="",
        help="Input sky map", dest="sky_map")
    parser.add_argument("--ground", action="store_true",
        default=False, help="include pickup", dest="ground")
    parser.add_argument("--balloon_track", type=str, default=None,
        help="Balloon path file", dest="balloon_track")
    #TOD filters
    parser.add_argument("--filter_highpass", dest="filter_highpass", default=False,
        help="Substract mean of ground tod chunk per chunk", action="store_true")
    parser.add_argument("--w_c", type=float, default=None, action="store", 
        dest="w_c")
    parser.add_argument("--filter_m", type=int, default=1, action="store", 
        dest="filter_m")
    


    args = parser.parse_args()

    if args.system=="OwlAEA":
        #Alex on Owl
        basedir = opj("/","mn","stornext", "u3","aeadler","ssn")
    else:
        basedir = "./"

    outdir = opj(basedir, "output")
    beamdir = opj(basedir, "beams")

    if args.beamdir:
        beamdir = opj(basedir, args.beamdir)

    #If we have an automated FP layout   
    if args.create_fpu:
        npairs = args.npairs
        beam_files = args.beam_file

    else: 
        bfi = open(opj(beamdir, args.beam_file))
        beam_files = bfi.read().splitlines()
        bfi.close()
        npairs = len(beam_files)

    if args.grasp:
        parse_beams(beam_files, beamdir, ss_obj=None, lmax=args.beam_lmax, 
                    stitch_wide=args.stitch_wide, plot=args.plot_beam)

    if args.operation=="run":
        if args.alm_type=="synfast":
            cls = np.loadtxt(opj(basedir,"wmap7_r0p03_lensed_uK_ext.txt"), unpack=True)
            ell, cls = cls[0], cls[1:]
            np.random.seed(25) 
            sky_alm = hp.synalm(cls, lmax=args.lmax, new=True, verbose=True)
        else:
            map_path = opj(basedir, args.sky_map)
            sky_alm = hp.map2alm(hp.read_map(map_path, field=None), 
                lmax=args.lmax)
        ground_alm = None

        run_opts = dict(
                        basedir = basedir,
                        beamdir = beamdir,
                        outdir = outdir,

                        mlen= args.days*24*60*60,  
                        t0 = args.t0,

                        btype = args.btype,
                        create_fpu=args.create_fpu,
                        beam_files = beam_files,
                        no_pairs=args.no_pairs,
                        npairs=npairs,
                        fov=args.fov,
                        killfrac=args.killfrac,
                        seed=args.seed,
                        preview_pointing=False,#args.preview_pointing,
                        add_ghosts=args.add_ghosts,
                        ghost_amp=args.ghost_amp,
                        fwhm=args.fwhm, 
                        lmax=args.lmax,
                        mmax=args.mmax,
                        deconv_q = args.deconv_q,
                        ground = args.ground, 
                        filter_highpass = args.filter_highpass,  
                        w_c = args.w_c,
                        filter_m = args.filter_m,                     
                        balloon_track = args.balloon_track,
                        el0 = args.el0,
                        az0 = args.az0,
                        verbose=args.verbose,
                        nside_spin=args.nside_spin,
                        nside_out=args.nside_out,
                        sample_rate=args.sample_rate,
                        freq=args.freq,
                        hwp_mode=args.hwp_mode,
                        hwp_model=args.hwp_model,
                        varphi=args.varphi,
                        filter_4fhwp=args.filter_4fhwp,
#                        pol_only=args.pol_only,
#                        no_pol=args.no_pol,
#                        glob_tag=args.glob_tag,

                        hfreq=args.hfreq,
                        hstepf=args.hstepf,
                        comm=comm)
        

        run_sim(args.sim_tag, sky_alm, **run_opts)

    return

if __name__ == "__main__":
    main()

