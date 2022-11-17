#SAT-like scan with ground

import numpy as np
import healpy as hp
import argparse as ap
from beamconv import ScanStrategy, Beam
from beamconv import tools as beam_tools
import pipeline_tools as tools
import qpoint as qp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import time
from datetime import date, datetime
import os 
import ephem
opj = os.path.join
plt.switch_backend("agg")
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
from ground_tools import template_from_position
import transfer_matrix as tm

def hwp_band5(center_nu):
    saph_w = 300./(2.0*0.317*center_nu)
    ratio = saph_w/3.75#Rescale AR thickness from the 95/150 AHWP band

    #Thickness
    thicks = np.array([0.5*ratio, 0.31*ratio, 0.257*ratio, 
                       saph_w, saph_w, saph_w, saph_w, saph_w,
                       0.257*ratio, 0.31*ratio, 0.5*ratio])
    #Indices: five saph layers sandwiched between 3 AR layers
    idxs = np.zeros((11,2))
    idxs[0] = [1.268,1.268]
    idxs[1] = [1.979, 1.979]
    idxs[2] = [2.855, 2.855]
    idxs[3:8] = [3.019, 3.336]
    idxs[8] = idxs[2]
    idxs[9] = idxs[1]
    idxs[10] = idxs[0]
    #Losses: dielectric constant along two axes
    losses = np.ones((11,2))*1.2e-3
    losses[3:8] = [2.3e-4, 1.25e-4]
    #Angles: rotation of the saph layers for birefringence
    angles = np.array([0.,0.,0., 22.9,-50.,0.,50.,-22.9 ,0.,0.,0.])*np.pi/180.0

    return [thicks, idxs, losses, angles]

def hwp_band3(center_nu): 
    saph_w = 300./(2.0*0.317*center_nu)
    ratio = saph_w/3.75#Rescale AR thickness from the 95/150 AHWP band

    #Thickness
    thicks = np.array([0.5*ratio, 0.31*ratio, 0.257*ratio, 
                        saph_w, saph_w, saph_w,
                       0.257*ratio, 0.31*ratio, 0.5*ratio])
    #Indices: five saph layers sandwiched between 3 AR layers
    idxs = np.zeros((9,2))
    idxs[0] = [1.268,1.268]
    idxs[1] = [1.979, 1.979]
    idxs[2] = [2.855, 2.855]
    idxs[3:6] = [3.019, 3.336]
    idxs[6] = idxs[2]
    idxs[7] = idxs[1]
    idxs[8] = idxs[0]
    #Losses: dielectric constant along two axes
    losses = np.ones((9,2))*1.2e-3
    losses[3:6] = [2.3e-4, 1.25e-4]
    #Angles: rotation of the saph layers for birefringence
    angles = np.array([0.,0.,0., 0.,54.0, 0., 0.,0.,0.])*np.pi/180.0

    return [thicks, idxs, losses, angles]

def hwp_band(center_nu):
    saph_w = 300./(2.0*0.317*center_nu)
    ratio = saph_w/3.75#Rescale AR thickness from the 95/150 AHWP band

    #Thickness
    thicks = np.array([0.5*ratio, 0.31*ratio, 0.257*ratio, saph_w,
                               0.257*ratio, 0.31*ratio, 0.5*ratio])
    #Indices: five saph layers sandwiched between 3 AR layers
    idxs = np.zeros((7,2))
    idxs[0] = [1.268,1.268]
    idxs[1] = [1.979, 1.979]
    idxs[2] = [2.855, 2.855]
    idxs[3] = [3.019, 3.336]
    idxs[4] = idxs[2]
    idxs[5] = idxs[1]
    idxs[6] = idxs[0]
    #Losses: dielectric constant along two axes
    losses = np.ones((7,2))*1.2e-3
    losses[3] = [2.3e-4, 1.25e-4]
    #Angles: rotation of the saph layers for birefringence
    angles = np.zeros(7)

    return [thicks, idxs, losses, angles]

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

def djd_to_unix_t(djd):
    return int((djd-25567.5)*86400)

def autoscale_y(ax, margin=0.1):
    '''
    This function rescales the y-axis based on the data that is visible given 
    the current xlim of the axis.
    ax : a matplotlib axes object
    margin : the fraction of the total height of the y-data to pad
    the upper and lower ylims
    '''

    def get_bottom_top(line):
        xd = line.get_xdata()
        yd = line.get_ydata()
        lo,hi = ax.get_xlim()
        y_displayed = yd[((xd>lo) & (xd<hi))]
        h = np.max(y_displayed) - np.min(y_displayed)
        bot = np.min(y_displayed)-margin*h
        top = np.max(y_displayed)+margin*h
        return bot,top

    lines = ax.get_lines()
    bot,top = np.inf, -np.inf

    for line in lines:
        new_bot, new_top = get_bottom_top(line)
        if new_bot < bot: bot = new_bot
        if new_top > top: top = new_top

    ax.set_ylim(bot,top)

def run_sim(simname, sky_alm,
            basedir = opj("/","mn", "stornext", "u3", "aeadler", "ssn"),
            beamdir = "beams", outdir = opj("output", "maps"),
            mlen=24*60*60,  sample_rate=119.1, t0=1546300800, 
            npairs=None, create_fpu=False, fov=2.0, beam_files=None,
            no_pairs=False, btype="Gaussian", fwhm=43., deconv_q=True, 
            lmax=1000, mmax=4, pol_only=False, no_pol=False, add_ghosts=False, 
            ghost_amp=0.01, el0=35., az0=0., sun_angle=6., freq=150., 
            ground = None, filter_highpass=False, w_c=None, filter_m=1,
            hwp_mode=None, hwp_model="ideal", load_mueller=False, varphi=0.0, 
            hfreq=1.0, hstepf=1/(12.*60*60), filter_4fhwp=False, 
            nside_spin=1024, nside_out=512, balloon_track = None, killfrac=0., 
            seed=25, preview_pointing=False, comm=None, verbose=1, **kwargs):
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
    sun_angle : float
        How far under the horizon the sun needs to be in degrees (default : 6.)
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
        Pre-included HWP model selected (default: ideal)
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
    balloon_track : string
        Path to file with balloon position over time (default: None)
    killfrac : float
        Fraction of detectors to randomly kill (default: 0.)
    seed : int
        Seed for random number generator (default: 25)
    preview_pointing : bool
        (default: False)
    comm : MPI communicator
    verbose : int
        How much to print while running beamconv (default: 1)
    """

    np.random.seed(seed)
    ndays = int(mlen/(24*60*60))
    track = np.loadtxt(opj(basedir, balloon_track))
    co_added_maps = np.zeros((3,hp.nside2npix(nside_out)))
    co_added_cond = np.zeros(hp.nside2npix(nside_out))
    co_added_hits = np.zeros((hp.nside2npix(nside_out)))
    nights_visited = np.zeros(hp.nside2npix(nside_out))
    ctime0 = t0
    night_idx = 0
    while ctime0<t0+mlen:
        track_idx = np.argmin(np.absolute(track[:,0]-ctime0))
        lat = track[track_idx,1]
        lon = track[track_idx,2]
        if rank==0:
            print(track[track_idx,0],lat, lon)
        h = 35000+200*np.random.normal()

        ymd = date.fromtimestamp(ctime0).strftime("%Y%m%d")
        #PyEphem calculation of sunset and sunrise
        sun = ephem.Sun()
        gondola = ephem.Observer()
        gondola.lon = ephem.degrees(np.radians(lon))
        gondola.lat = ephem.degrees(np.radians(lat))
        gondola.date = datetime.utcfromtimestamp(ctime0)
        gondola.elevation = h
        gondola.horizon = np.radians(-sun_angle)
        next_set = djd_to_unix_t(gondola.next_setting(sun))
        next_rise = djd_to_unix_t(gondola.next_rising(sun))
        #Key assumption: we reach float altitude at daytime. So the first 
        #night might be lost by the following two lines
        if next_rise<next_set:#i.e. we are at night
            ctime0 = next_rise + 60
            continue
        scan_duration = next_rise - next_set
        if rank==0:
            print("{:d}GHz, night {:d}, ctime0 {:d}, duration: {:.2f}h".format(
                      int(freq), night_idx, ctime0, scan_duration/3600.))
        scan = ScanStrategy(scan_duration, sample_rate=sample_rate, 
                            lat=lat, lon=lon, ctime0=next_set)
        #reverse scan direction every night
        scan_opts = dict(scan_speed=30.*int(2*(night_idx%2-.5)), 
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
                beam_opts["po_file"] = beam_files#It"s just one file here
            scan.create_focal_plane(nrow=nfloor, ncol=nfloor, fov=fov, 
                                    ab_diff=45., **beam_opts)

        else:
            scan.load_focal_plane(beamdir, btype=btype, no_pairs=no_pairs, 
                                  sensitive_freq=freq, file_names=beam_files)


        if hwp_model == "ideal":
            pass
        elif "band" in hwp_model:
            center_nu = 185.
            if hwp_model=="band5":
                stack = hwp_band5(center_nu)
            elif hwp_model=="band3":
                stack = hwp_band3(center_nu)
            else:
                stack = hwp_band(center_nu)
            for beami in scan.beams:
                beami[0].set_hwp_mueller(thicknesses=stack[0], indices=stack[1],
                    losses=stack[2], angles=stack[3])
                beami[1].set_hwp_mueller(thicknesses=stack[0], indices=stack[1],
                    losses=stack[2], angles=stack[3]) 
        else:
            for beami in scan.beams:
                beami[0].set_hwp_mueller(model_name=hwp_model)
                beami[1].set_hwp_mueller(model_name=hwp_model) 
            
        scan.partition_mission(chunksize=int(sample_rate*3600))
        scan.allocate_maps(nside=512)
        
        hwpf = 0.
        if hwp_mode == "stepped":
            hwpf = hstepf
        elif hwp_mode == "continuous":
            hwpf = hfreq   
        scan.set_hwp_mod(mode=hwp_mode, freq=hwpf, varphi=varphi)
        
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
                ground_alm = np.zeros((3,hp.Alm.getsize(lmax)), dtype=complex)
            comm.Bcast(ground_alm, root=0)

            scan.scan_instrument_mpi(sky_alm, ground_alm=ground_alm, 
                                     **scan_opts)
        
        else:
            scan.scan_instrument_mpi(sky_alm, **scan_opts)
        ctime0 = next_rise+60
        maps, cond, proj = scan.solve_for_map(return_proj=True)
        if scan.mpi_rank == 0:
            hp.write_map(opj(basedir, outdir,
                 "maps_"+simname+"_{:02d}.fits".format(night_idx)), maps)
            hp.write_map(opj(basedir, outdir, 
                "cond_"+simname+"_{:02d}.fits".format(night_idx)), cond)
            hp.write_map(opj(basedir, outdir, 
                "hits_"+simname+"_{:02d}.fits".format(night_idx)), proj[0])
            co_added_maps[:, maps[0]!=hp.UNSEEN] += maps[:, maps[0]!=hp.UNSEEN]
            co_added_hits += proj[0]
            co_added_cond[maps[0]!=hp.UNSEEN] = np.minimum(
                cond[maps[0]!=hp.UNSEEN], co_added_cond[maps[0]!=hp.UNSEEN])
            nights_visited[maps[0]!=hp.UNSEEN] += 1
            night_idx +=1
    
    if scan.mpi_rank==0:

        co_added_maps[:,nights_visited!=0] /= nights_visited[nights_visited!=0]
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
        for det in ["A","B"]:
            # fields
            pk_file = open(opj(beamdir, beam_file+"_{}_fields.pkl".format(det)
                              ), "rb")
            peak_fields = pickle.load(pk_file)
            pk_file.close()
            prop_file = open(opj(beamdir, beam_file+"_"+det+"_prop.pkl"), "rb")
            prop = pickle.load(prop_file)
            prop_file.close()
            e_co = peak_fields["e_co"]
            e_cx = peak_fields["e_cx"]
            invert_flag=False
            #Sanity check that we have the main field in the right order
            if np.sum(np.absolute(e_co))<np.sum(np.absolute(e_cx)):
                e_co = peak_fields["e_cx"]
                e_cx = peak_fields["e_co"]
                print("Inverted eco and ecx!")
                invert_flag=True

            cr = peak_fields["cr"] # [azmin, elmin, azmax, elmax]
            d_az = cr[2] - cr[0]
            d_el = cr[3] - cr[1]

            if stitch_wide:
                wide_beam_file = beam_file.replace("v2", "v2_wide") 
                wd_file = open(opj(beamdir, 
                    wide_beam_file+"_{}_fields.pkl".format(det)), "rb")
                wide_fields = pickle.load(wd_file)
                wd_file.close()
                e_co_wide = wide_fields["e_co"]
                e_cx_wide = wide_fields["e_cx"]
                if invert_flag:
                    e_co_wide = wide_fields["e_cx"]
                    e_cx_wide = wide_fields["e_co"]

                cr_wide = wide_fields["cr"] # [azmin, elmin, azmax, elmax]
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
            po_file = opj(beamdir, beam_file+"_{}.npy".format(det))
            np.save(po_file, blm)

        # set common opts
        #fwhm = prop["fwhm"] * 60 # arcmin
        az_off = prop["cx"]
        el_off = prop["cy"]
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

def analysis(analyzis_dir, sim_tag, ideal_map=None, input_map=None,
             calibrate=None, mask_file=None, nside_out=512, lmax=700, 
             l1=100, l2=300, fwhm=60., plot=False, label=None):
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
    calibrate : int
        Whether to recalibrate the scanned map vs ideal and/or input.
        A gain is computed as the average ratio between the ideal/input 
        spectrum and the sim_tag spectrum. The sim_tag map is scaled by 
        sqrt(gain). 0 for TT, 1 for EE etc... (default: None)
    mask : string
        Mask file to apply to the maps (default: None)
    nside_out : int
        Healpix NSIDE of the output map. (default: 512)
    lmax : int
        Maximum l when making power spectra. (default: 700)
    l1 : int
        Lower edge of the calibration window. (default: 100)
    l2 : int
        Higher edge of the calibration window. (default: 300)
    fwhm : float
        fwhm of the beam in arcmin. (default: 60.)
    plot : bool
        Whether to plot the spectra we have 
    label : string
        What the non-ideality is called on the legend
    """

    filename = "maps_"+sim_tag+".fits"
    fstrs = ["TT", "EE", "BB", "TE", "TB", "EB"]
    maps = tools.read_map(opj(analyzis_dir, filename), field=None, fill=np.nan)
    hits = tools.read_map(opj(analyzis_dir, filename.replace("maps_", "hits_")))
    cond = tools.read_map(opj(analyzis_dir, filename.replace("maps_", "cond_")))
    maps = hp.ud_grade(maps, nside_out)
    hits = hp.ud_grade(hits, nside_out)
    cond = hp.ud_grade(cond, nside_out)
    """"
    if mask_file:
        custom_mask = hp.ud_grade(tools.read_map(opj(analyzis_dir, mask)), 
                                  nside_out)
        hits_mask = hits * custom_mask
        fsky = np.sum(hits_mask > 0.) / float(len(hits_mask))
    else:
        fsky = np.sum(hits > 0.) / float(len(hits))
    print("fsky: {:.3f}".format(fsky))
    hits[hits == 0] = np.nan
    if mask_file:
        hits[custom_mask==0] = np.nan
    mask = ~np.isnan(hits)
    for ma in maps:
            ma[~mask] = np.nan
    """
    #Rectangular mask
    mask = np.ones(12*nside_out**2)
    theta, phi = hp.pix2ang(nside_out, np.arange(12*nside_out**2))
    max_dec = np.radians(20.)
    min_dec = np.radians(-80.)
    dec_centre = .5*np.pi - .5*(max_dec+min_dec)
    dec_hwidth = .5*(max_dec-min_dec)
    mask[np.abs(theta-dec_centre)>dec_hwidth]=0.
    mask[maps[0]==hp.UNSEEN]=0.
    fsky = np.sum(mask)/(12*nside_out**2)
    spice_opts2use = get_default_spice_opts(lmax=lmax, fsky=fsky)

    print("fsky is {:.2f}".format(fsky))
    cl = tools.spice(maps, mask=mask, **spice_opts2use)
    bl = hp.gauss_beam(fwhm=np.radians(fwhm/60.), lmax=len(cl[1])-1)
    cl = cl/bl**2
    np.save(opj(analyzis_dir, "spectra", 
            "{}_spectra_rect.npy".format(sim_tag)), cl)

    
    #Versus ideal
    if ideal_map:

        ideal_maps = tools.read_map(opj(analyzis_dir, ideal_map),
            field=None, fill=np.nan)
        ideal_maps = hp.ud_grade(ideal_maps, nside_out)
        
        masked_ideal = ideal_maps.copy()
        #mask the ideal map on unscanned pixels and masked areas
        for mi in masked_ideal:
            mi[mask[:]==0] = 0.
        #Calibration
        gain = 1.
        if calibrate:
            cl_ideal = tools.spice(masked_ideal, mask=mask, **spice_opts2use)
            cl_ideal = cl_ideal/bl**2
            np.save(opj(analyzis_dir, "spectra", 
                   "{}_idealspectra_rect.npy".format(sim_tag)), cl_ideal)
            #Compute Cls for the smoothed map, deconvolve
            gain = np.average(cl_ideal[calibrate, l1:l2]/cl[calibrate, l1:l2])
            print("{} gain for map {} versus ideal is: {:.3f}".format(
                fstrs[calibrate], sim_tag, gain)) 

        #Should difference maps be gain_corrected?
        diff_ideal = maps*np.sqrt(gain) - ideal_maps
        for diffi in diff_ideal:
            diffi[mask[:]==0] = 0.
        diff_ideal_cl = tools.spice(diff_ideal, mask=mask, **spice_opts2use)
        diff_ideal_cl = diff_ideal_cl/bl**2
        np.save(opj(analyzis_dir, "spectra", 
                    "{}_diff_ideal_spectra.npy".format(sim_tag)), diff_ideal_cl)

    #Versus input
    if input_map:

        input_maps = tools.read_map(opj(analyzis_dir, input_map),
            field=None, fill=np.nan)
        input_maps = hp.ud_grade(input_maps, nside_out)
        
        masked_input = input_maps.copy()
        #mask the input map on unscanned pixels and masked areas
        mask = ~np.isnan(hits)
        for mi in masked_input:
            mi[~mask] = np.nan
        #Calibration
        gain = 1.
        if calibrate:
            cl_input= tools.spice(masked_input, mask=mask, **spice_opts2use)
            cl_input = cl_input/bl**2
            #Compute Cls for the smoothed map, deconvolve
            gain = np.average(cl_input[calibrate, l1:l2]/cl[calibrate, l1:l2])
            print("{} gain for map {} versus input is: {:.3f}".format(
                  fstrs[calibrate], sim_tag, gain))

        diff_input = maps*np.sqrt(gain) - input_maps
        for diffi in diff_input:
            diffi[~mask] = np.nan
        diff_input_cl = tools.spice(diff_input, mask=mask, **spice_opts2use)
        diff_input_cl = diff_input_cl/bl**2
        np.save(opj(analyzis_dir, "spectra", 
                    "{}_diff_input_spectra.npy".format(sim_tag)), diff_input_cl)

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
    parser.add_argument("--beam_file", action="store", dest="beam_file", 
        type=str, default=None, help="Text file listing beams to use")
    parser.add_argument("--beam_lmax", action="store", dest="beam_lmax",
        default=2000, type=int, help="Maximum lmax for beam decomposition")
    parser.add_argument("--beam_type", type=str, dest="btype", 
        default="Gaussian", help="Input beam type: [Gaussian, PO]")
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

    #Operations
    parser.add_argument("--grasp", action="store_true", dest="grasp", 
        default=False, help="The beams come as pickles of grasp grids and cuts")
    parser.add_argument("--run", action="store_true", dest="run",
        default=False, help="Create maps")
    parser.add_argument("--analyze", action="store_true", dest="analyze",
        default=False, help="Make difference maps and spectra")

    #Beamconv
    parser.add_argument("--sim_tag", action="store", dest="sim_tag", type=str,
        default="test", help="Identifier for simulation name")
    parser.add_argument("--quiet", action="store_const", dest="verbose",
        const=0, default=1, help="less print statements")
    parser.add_argument("--seed", action="store", dest="seed", type=float,
        default=25)
    parser.add_argument("--el0", action="store", dest="el0", type=float,
        default=50., help="Starting elevation (ballon scan)")
    parser.add_argument("--az0", action="store", dest="az0", type=float,
        default=0., help="Starting azimuth, balloon scan")
    parser.add_argument("--sun_angle", action="store", dest="sun_angle", 
        type=float, default=6., help="Sun at least x degrees under horizon")
 
    #Beamconv timing
    parser.add_argument("--t0", action="store", dest="t0",
        default=1427376366, type=float)
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
    parser.add_argument("--hwp_model", action="store", dest="hwp_model", 
        type=str, default="ideal")
    parser.add_argument("--hwp_phase", action="store", dest="varphi",
        type=float, default=0.)
    parser.add_argument("--hfreq", action="store", dest="hfreq", type=float,
        default=1.0)
    parser.add_argument("--hstepf", action="store", dest="hstepf", type=float,
        default=1./(12*3600))
    parser.add_argument("--filter_4fhwp", action="store_true", 
        dest="filter_4fhwp", default=False)

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
    parser.add_argument("--filter_highpass", dest="filter_highpass", 
        default=False, help="Substract mean of tod per chunk", 
        action="store_true")
    parser.add_argument("--w_c", type=float, default=None, action="store", 
        dest="w_c")
    parser.add_argument("--filter_m", type=int, default=1, action="store", 
        dest="filter_m")

    #Analysis arguments
    parser.add_argument("--ideal_map", type=str, action="store", default=None, 
	help="ideal map for comparaison", dest="ideal_map") 
    parser.add_argument("--input_map", type=str, action="store", default=None, 
	help="input map for comparaison", dest="input_map") 
    parser.add_argument("--mask", type=str, default=None, action="store",
	help="Mask for analysis", dest="mask")
    parser.add_argument("--calibrate", default=None, action="store", 
        dest="calibrate", help="Calibrate vs ideal or input") 
    parser.add_argument("--plot", default=False, action="store_true", 
        dest="plot", help="Plot spectra") 
    parser.add_argument("--label", type=str, action="store", default=None, 
	help="Label of non-ideality on plots", dest="label") 
    args = parser.parse_args()

    if args.system=="OwlAEA":
        #Alex on Owl
        basedir = opj("/","mn","stornext", "u3","aeadler","taurus_systematics")
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

    elif not args.analyze: 
        bfi = open(opj(beamdir, args.beam_file))
        beam_files = bfi.read().splitlines()
        bfi.close()
        npairs = len(beam_files)

    if args.grasp:
        parse_beams(beam_files, beamdir, ss_obj=None, lmax=args.beam_lmax, 
                    stitch_wide=args.stitch_wide, plot=args.plot_beam)

    if args.run:
        if args.alm_type=="synfast":
            cls = np.loadtxt(opj(basedir,"wmap7_r0p03_lensed_uK_ext.txt"), 
                    unpack=True)
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
                        sample_rate=args.sample_rate,
                        t0 = args.t0,
                        npairs=npairs,
                        create_fpu=args.create_fpu,
                        fov=args.fov,
                        beam_files = beam_files,
                        no_pairs=args.no_pairs,
                        btype = args.btype,
                        fwhm=args.fwhm,
                        deconv_q = args.deconv_q,
                        lmax=args.lmax,
                        mmax=args.mmax,
                        add_ghosts=args.add_ghosts,
                        ghost_amp=args.ghost_amp,
                        el0 = args.el0,
                        az0 = args.az0,
                        sun_angle = args.sun_angle,
                        ground = args.ground, 
                        filter_highpass = args.filter_highpass,
                        w_c = args.w_c,
                        filter_m = args.filter_m,
                        hwp_mode=args.hwp_mode,
                        hwp_model=args.hwp_model,
                        varphi=args.varphi,
                        filter_4fhwp=args.filter_4fhwp,
                        hfreq=args.hfreq,
                        hstepf=args.hstepf,
                        nside_spin=args.nside_spin,
                        nside_out=args.nside_out,
                        balloon_track = args.balloon_track,
                        killfrac=args.killfrac,
                        seed=args.seed,
                        preview_pointing=False,
                        freq=args.freq,
                        comm=comm,
                        verbose=args.verbose)
        

        run_sim(args.sim_tag, sky_alm, **run_opts)
    if args.analyze:
        analyzis_dir = opj(basedir, "output")
        analysis(
            analyzis_dir = analyzis_dir, 
            sim_tag = args.sim_tag, 
            ideal_map = args.ideal_map, 
            input_map = args.sky_map,
            calibrate = args.calibrate, 
            mask_file = args.mask,
            nside_out = args.nside_out, 
            lmax = 400,
            fwhm = args.fwhm, 
            l1 = 100, 
            l2 = 300,
            plot = args.plot,
            label = args.label)

    return

if __name__ == "__main__":
    main()

