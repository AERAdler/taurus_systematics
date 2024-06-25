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
import yaml
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

def balloon_night_ctime(**kwargs):

    ctime0 = kwargs.get("ctime0")
    nsamp = kwargs.get("nsamp")
    sample_rate = kwargs.get("sample_rate")
    track_file = kwargs.get("track_file")
    latlon = kwargs.get("latlon")
    sun_angle = kwargs.get("sun_angle")
    track = np.loadtxt(track_file)
    ctime = ctime0+np.arange(nsamp)/sample_rate
    night = np.zeros_like(ctime, dtype=bool)
    lat = np.interp(ctime, track[:,0], track[:,1])
    lon = np.interp(ctime, track[:,0], track[:,2])
    h = 32700+200*np.random.normal()

    gondola = ephem.Observer()
    gondola.elevation = h

    for i in range(0, nsamp, int(sample_rate)):
        #PyEphem calculation of sunset and sunrise
        gondola.lat = ephem.degrees(np.radians(lat[i]))
        gondola.lon = ephem.degrees(np.radians(lon[i]))
        gondola.date = datetime.utcfromtimestamp(ctime[i])
        night[i:i+int(sample_rate)] = ephem.Sun(gondola).alt < -np.radians(sun_angle)
    
    if latlon:
        lat = lat[night]
        lon = lon[night]
        ctime = ctime[night]
        return lat, lon, ctime
    else:
        ctime = ctime[night]
        return ctime

def pass_ctime(**kwargs):
    ctime = kwargs.pop("ctime")
    return ctime


def run_sim(simname, sky_alm,
            basedir=opj("/","mn", "stornext", "u3", "aeadler", "taurus_systematics"), 
            beamdir = "beams", outdir = opj("output", "maps"), mlen=30*24*60*60,
            sample_rate=119.1, ctime0=1427376366,  npairs=None, 
            create_fpu=False, fov=20.0, beam_files=None, no_pairs=False, 
            ab_diff=90., qu_alternate=False, btype="Gaussian", fwhm=33., 
            deconv_q=True, lmax=1000, mmax=4, pol_only=False, no_pol=False, 
            add_ghosts=False, ghost_amp=0.01, point_bias_mode=0, 
            point_bias=[0.,0.,0.], el0=35., az0=0., psi0=0., sun_angle=5., 
            freq=150., ground = None, filter_highpass=False, w_c=None, 
            filter_m=1, hwp_mode=None, hwp_model="ideal", load_mueller=False, 
            varphi=0.0, hfreq=1.0, hstepf=1/(12.*60*60), filter_4fhwp=False, 
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
        Mission duration in seconds (default: 30*24*60*60)
    sample_rate : float
        Number of samples per second (default: 119.1)
    ctime0 : int 
        Mission start-time, Unix time (default: 1427376366)
    npairs : int
        Number of detector pairs
    create_fpu : bool
        Create square focal plane (default : False)
    fov : float
        Field of view on square focal plane, in degrees (default: 20.0)
    beam_files : list
        List containing the name of the beams/the one beam to be loaded 
        (default : None)
    no_pairs : bool
        Only have A detectors (default : False)
    ab_diff : float
        The polarisation angle difference between the a and b detectors 
        in a pair (default : 90.)
    qu_alternate : bool
        Alternate Q and U A detectors (default : False)
    btype : string
        Beam type for the autogenerated FP: Gaussian/EG/PO
        (default: Gaussian)
    fwhm : float 
        Beam FWHM for the autogenerated Gaussian FP in arcmin (default: 33)
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
    point_bias_mode: int
        pointing error mode. 0 none, 1 random shift per beam with std given
        by point_bias, 2 offset all beams by point_error (default : 0)
    point_bias: array-like
        pointing error of [az_off, el_off, polang_off] (default : [0,0,0])
    el0 : float
        Boresight starting elevation in degrees (default: 35.)
    az0 : float
        Boresight starting azimuth in degrees (default: 0.)
    psi0 : float
        Boresight rotation from horizontal in degrees (default: 0.)
    sun_angle : float
        How far under the horizon the sun needs to be in degrees (default : 5.)
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
    track = np.loadtxt(opj(basedir, balloon_track))
    maps = np.zeros((3,hp.nside2npix(nside_out)))
    cond = np.zeros(hp.nside2npix(nside_out))
    hits = np.zeros((hp.nside2npix(nside_out)))
    
    
    nsamp = int(mlen*sample_rate)
    ctime_dict = dict(ctime0 = ctime0, nsamp = nsamp,
        sample_rate = sample_rate, track_file = balloon_track, 
        latlon = True, sun_angle = 6.)
    lat, lon, ctime = balloon_night_ctime(**ctime_dict)
    night_samps = len(ctime)
    total_samps = night_samps
    passct_kwargs = dict(ctime=ctime)
    scan = ScanStrategy(sample_rate=sample_rate, num_samples=night_samps, 
                external_pointing=True, ctime0=ctime0, lat=lat, lon=lon)
    scan_opts = dict(q_bore_func=scan.balloon_night_qbore, 
                     q_bore_kwargs=dict(el0=el0, az0=az0, scan_speed=30.),
                     ctime_func=pass_ctime,
                     ctime_kwargs=passct_kwargs,
                     max_spin=2,
                     nside_spin=nside_spin,
                     preview_pointing=False,
                     interp = True, 
                     filter_highpass = filter_highpass)

    if create_fpu:#A square focal plane
        beam_opts = dict(lmax=lmax, fwhm=fwhm, btype=btype, 
                         sensitive_freq=freq, deconv_q=deconv_q)
        nfloor = int(np.floor(np.sqrt(npairs)))
        if btype=="PO":
            beam_opts["po_file"] = opj(beamdir, beam_files)#It"s just one file here
        scan.create_focal_plane(nrow=nfloor, ncol=nfloor, fov=fov, 
                                ab_diff=ab_diff, qu_alternate=qu_alternate, 
                                **beam_opts)

    else:
        scan.load_focal_plane(beamdir, btype=btype, no_pairs=no_pairs, 
                              sensitive_freq=freq, file_names=beam_files)
    #Insert pointing errors
    if point_bias_mode!=0:
        pbdeg = np.array(point_bias)/60.
        if point_bias_mode==1:
            scan.set_global_prop_random(dict(az_bias=pbdeg[0],
                       el_bias=pbdeg[1], polang_bias=pbdeg[2])) 
        elif point_bias_mode==2:
            scan.set_global_prop(dict(az_bias=pbdeg[0],
                       el_bias=pbdeg[1], polang_bias=pbdeg[2])) 
        else:
            raise ValueError("Unknown pointing error mode")
    #Ghosting
    if add_ghosts:
        ghost_dict = dict(amplitude=ghost_amp)
        scan.create_reflected_ghosts(ghost_tag='refl_ghost',
                                rand_stdev=0., **ghost_dict)
    #HWP selection and insert into beamconv as HWP for each beam
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
    scan.partition_mission(chunksize=int(sample_rate*3600*24))
    scan.allocate_maps(nside=nside_out)
    #Rotation of the HWP
    hwpf = 0.
    if hwp_mode == "stepped":
        hwpf = hstepf
    elif hwp_mode == "continuous":
        hwpf = hfreq   
    scan.set_hwp_mod(mode=hwp_mode, freq=hwpf, varphi=varphi)
    #Rotate the boresight once, belt and suspenders safety on arguments
    scan.set_instr_rot(period=mlen, start_ang=psi0, angles=[psi0, psi0])
    
    if filter_highpass and (w_c is not None):
        scan.set_filter_dict(w_c, m=filter_m)
    if ground: #Fixed ground for now
        if rank==0:
            t1 = time.time()
            """
            ground_template = np.ones(12*4096**2)*2.5e8
            tht, phi = hp.pix2ang(4096, np.arange(12*4096**2))
            sky_tht = np.degrees(tht[np.abs(tht-np.radians(93))<np.pi/60.])
            temp_scaling = np.exp(-.5 * ((sky_tht-96)/0.95)**2 )#roughly 4e8 damping at horizon
            ground_template[np.abs(tht-np.radians(93))<np.pi/60.] *= temp_scaling
            ground_template[tht<np.pi/2.] = 0.
            """
            lat0 = lat[0]
            lon0 = lon[0]
            c0  = ctime[0]
            h = 32700+200*np.random.normal()
            yd = date.fromtimestamp(c0).strftime("%Y%j")
            world_map = hp.read_map(opj(basedir,"ground_input", "SSMIS",
                                    "SSMIS-{}-91H_South.fits".format(yd)))
            ground_template = template_from_position(world_map, lat0, lon0,
                        h, nside_out=4096, cmb=True, freq=freq, frac_bwidth=.2, aposcale=0.95)
            print("Building ground template: {:.2f}s".format(time.time()-t1))
            t2 = time.time()
            ground_alm = hp.map2alm([ground_template, 
                                    np.zeros_like(ground_template), 
                                    np.zeros_like(ground_template)], 
                                    lmax = lmax)
            ground_alm = hp.smoothalm(ground_alm, fwhm = np.radians(1.)) 
            print("Transforming to alms: {:.2f}s".format(time.time()-t2))
        else:
            ground_alm = np.zeros((3,hp.Alm.getsize(lmax)), dtype=complex)
        
        comm.Barrier()
        comm.Bcast(ground_alm, root=0)
        scan_opts["q_bore_kwargs"]["ground"] = True
        t3 = time.time()
        scan.scan_instrument_mpi(sky_alm, ground_alm=ground_alm, **scan_opts)
    else:
        t3 = time.time()
        scan.scan_instrument_mpi(sky_alm, **scan_opts)
    maps, cond, proj = scan.solve_for_map(return_proj=True)

    if scan.mpi_rank==0:
        print("Scanning, mapmaking {:.2f}s".format(time.time()-t3))
        hp.write_map(opj(basedir, outdir, "maps_"+simname+".fits"),
             maps)
        hp.write_map(opj(basedir, outdir, "cond_"+simname+".fits"),
             cond)
        hp.write_map(opj(basedir, outdir, "hits_"+simname+".fits"),
             proj[0])
        
    return

def parse_beams(beam_files, beamdir, ss_obj=None, lmax=2000, psi0=0.,
                no_pairs=False, ab_diff=90., qu_alternate=False, 
                stitch_wide=False, sym_beam=False, plot=False):
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
    psi0 : float
        Rotate the grasp azimuths/elevation by psi0 when building the 
        focal plane properties(default: O.)
    no_pairs : bool
        Only have A detectors (default : False)
    ab_diff : float
        The polarisation angle difference between the a and b detectors 
        in a pair (default : 90.)
    qu_alternate: bool
        If True, have succesive detectors pairs at 45 degree polang from each
        other (default: False)
    stitch_wide : bool
        If True, add a sidelobe (from the pickle file) to the main beam
        (default: False)
    sym_beam: bool
        Supress all non-zero m modes (default: False)
    plot : bool
        Produce plots of the Stokes beams (default: False)
    Notes
    -----
    GRASP output e fields are converted to HEALPix pixels, apodized,
    then spin-0 blm"s are synthesized, which are converted
    to spin-0 and spin-\pm2 by convolution in harmonic space.
    """

    

    # scatter files
    if isinstance(ss_obj, ScanStrategy):
        sat = ss_obj
    else:
        sat = ScanStrategy(sample_rate=10., num_samples=10)

    # perhaps pair all beam files with the appropriate wide beam?

    beam_files = np.asarray(beam_files)
    beam_files_loc = beam_files[sat.mpi_rank::sat.mpi_size]

    num_beams = beam_files_loc.shape[0]
    for bidx in range(num_beams):
        polang_a = num_beams%2 *45 if qu_alternate else 0 #ternaries are cool
        polang_b = polang_a+ab_diff

        beam_file = beam_files_loc[bidx]

        print("Rank = {:03d} | bidx = {:03d} | filename = {}".\
            format(sat.mpi_rank, bidx, beam_file))
        if no_pairs:
            parse_single_det(beamdir, beam_file, lmax=lmax, 
                stitch_wide=stitch_wide, sym_beam=sym_beam, plot=plot)
            prop_file = open(opj(beamdir, beam_file+"_prop.pkl"), "rb")
            prop = pickle.load(prop_file)
            prop_file.close()

        else:
            for det in ["A","B"]:
                detname = beam_file+"{}".format(det)
                parse_single_det(beamdir, beam_file, lmax=lmax, det=det,
                    stitch_wide=stitch_wide, sym_beam=sym_beam, plot=plot)
                prop_file = open(opj(beamdir, beam_file+"_prop.pkl"), "rb")
                prop = pickle.load(prop_file)
                prop_file.close()

        # set common opts
        az_off = prop["cx"]*np.cos(psi0) - prop["cy"]*np.sin(psi0)
        el_off = prop["cx"]*np.sin(psi0) + prop["cy"]*np.cos(psi0) 
        beam_opts = dict(lmax=lmax, deconv_q=True, normalize=True,
                         cross_pol=True, btype="PO",az=az_off, el=el_off)

        # set A specifics
        a_opts = dict(polang=polang_a, pol="A", name=beam_file+"_A")
        a_opts.update(beam_opts)

        # store in pickle file
        with open(opj(beamdir, beam_file+".pkl"), "wb") as handle:
            if no_pairs:
                a_opts.update({"po_file" : opj(beamdir, beam_file+".npy")})
                pickle.dump(a_opts, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                b_opts = dict(polang=polang_b, pol="B", name=beam_file+"B")
                b_opts.update(beam_opts)
                b_opts.update({"po_file" : opj(beamdir, beam_file+"B.npy")})
                a_opts.update({"po_file" : opj(beamdir, beam_file+"A.npy")})
                pickle.dump([a_opts, b_opts], handle, 
                            protocol=pickle.HIGHEST_PROTOCOL)

    sat.barrier()

def parse_single_det(beamdir, beam_file, lmax=2000, det=None, 
                     stitch_wide=False, sym_beam=False, plot=False):
    """
    Load GRASP output, convert into blms and save.

    Arguments
    ---------
    beamdir: str
        Path to beam directory where .npy and .pkl beam files
        will be stored, and where grasp files are located.
        Filenames will be equal to input.
    beam_file: str
        Name of specific beam to sample within beamdir

    Keyword Arguments
    -----------------
    lmax : int
        lmax for blms (default : 2000)
    det : str
        Name of detector in beam pair (default : None)
    stitch_wide : bool
        Whether to add a sidelobe (from the pickle file) to the main beam
        (default: False)
    sym_beam : bool
        Whether to set all non-zero m modes to zero. (default: False)
    plot : bool
        Produce plots of the Stokes beams (default: False)
    Notes
    -----
    GRASP output e fields are converted to HEALPix pixels, apodized,
    then spin-0 blm"s are synthesized, which are converted
    to spin-0 and spin-\pm2 by convolution in harmonic space.
    """
    # fields
    nside_blm = 2048
    apodize = True # Note, apodize doesnt apply in case of sidelobes
    lmax_big = 2000

    pk_file = open(opj(beamdir, beam_file+"_fields.pkl"), "rb")
    peak_fields = pickle.load(pk_file)
    pk_file.close()
    prop_file = open(opj(beamdir, beam_file+"_prop.pkl"), "rb")
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
        wide_beam_file= beam_file.replace("grid", "grid_wide") 
        wb_file = open(opj(beamdir, 
            wide_beam_file+"_fields.pkl"), "rb")
        wide_fields = pickle.load(wb_file)
        wb_file.close()
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
    if sym_beam:
        blm[:, lmax:] = 0.
    # save npy file
    po_file = opj(beamdir, beam_file+det+".npy")
    np.save(po_file, blm)
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
    parser.add_argument("--sym_beam", action="store_true", dest="sym_beam",
        default=False, help="Set m!=0 modes to 0")
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
    parser.add_argument("--ghost_amp", action="store", dest="ghost_amp",
        default=0.01, type=float)
    parser.add_argument("--point_bias_mode", type=int, action="store",
        default=0, dest="point_bias_mode", help="Pointing bias mode: [0,1,2]")
    parser.add_argument("--point_bias", action="store", dest="point_bias", 
         nargs=3, default=[0, 0, 0], help="pointing bias", type=float)
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

    #Beamconv
    parser.add_argument("--sim_tag", action="store", dest="sim_tag", type=str,
        default="test", help="Identifier for simulation name")
    parser.add_argument("--quiet", action="store_const", dest="verbose",
        const=0, default=1, help="less print statements")
    parser.add_argument("--seed", action="store", dest="seed", type=float,
        default=25)
    parser.add_argument("--el0", action="store", dest="el0", type=float,
        default=35., help="Starting elevation (ballon scan)")
    parser.add_argument("--az0", action="store", dest="az0", type=float,
        default=0., help="Starting azimuth, balloon scan")
    parser.add_argument("--psi0", action="store", dest="psi0", type=float,
        default=0., help="rotation of focal plane")
    parser.add_argument("--sun_angle", action="store", dest="sun_angle", 
        type=float, default=5., help="Sun at least x degrees under horizon")
 
    #Beamconv timing
    parser.add_argument("--ctime0", action="store", dest="ctime0",
        default=1427376366, type=float)
    parser.add_argument("--days", action="store", dest="days",
        default=30, type=int)
    parser.add_argument("--sample_rate", action="store", dest="sample_rate",
        default=119.1, type=float)

    #Beamconv fpu
    parser.add_argument("--fov", action="store", dest="fov",
        default=2.0, type=float, help="Field of view in degrees")
    parser.add_argument("--create_fpu", action="store_true", dest="create_fpu",
        default=False)
    parser.add_argument("--npairs", action="store", dest="npairs",
        default=1, type=int, help="Number of beams in fpu")
    parser.add_argument("--no_pairs", action="store_true", dest="no_pairs",
        default=False)
    parser.add_argument("--qu_alternate", action="store_true", 
        dest="qu_alternate", default=False, help="alternate Q/U for detA")
    parser.add_argument("--ab_diff", action="store", dest="ab_diff", 
        default=90., type=float, help="polang between A and B within a pair")
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
    parser.add_argument("--calibrate", type=int, default=None, action="store", 
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
                    no_pairs=args.no_pairs, ab_diff = args.ab_diff, 
                    stitch_wide=args.stitch_wide, sym_beam=args.sym_beam, 
                    plot=args.plot_beam, psi0=args.psi0)

    if args.run:
        if args.alm_type=="synfast":
            dl_th = np.loadtxt(opj(basedir,"planckrelease3_spectra.txt"))
            ell = dl_th[:,0]
            dfac = ell*(ell+1)/(2*np.pi)
            cls = np.zeros((4,dl_th.shape[0]))
            #Change order to match polspice/namaster, let mono and dipole be zero
            cls[0,2:] = dl_th[2:,1]/dfac[2:]#TT
            cls[1,2:] = dl_th[2:,3]/dfac[2:]#EE
            cls[2,2:] = dl_th[2:,4]/dfac[2:]#BB
            cls[3,2:] = dl_th[2:,2]/dfac[2:]#TE
            np.random.seed(args.seed) 
            sky_alm = hp.synalm(cls, lmax=args.lmax, new=True, verbose=True)
        elif args.alm_type=="empty":
            sky_alm = np.zeros((3, hp.Alm.getsize(args.lmax)))
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
                        ctime0 = args.ctime0,
                        npairs=npairs,
                        create_fpu=args.create_fpu,
                        fov=args.fov,
                        beam_files = beam_files,
                        no_pairs=args.no_pairs,
                        qu_alternate = args.qu_alternate,
                        ab_diff = args.ab_diff,
                        btype = args.btype,
                        fwhm=args.fwhm,
                        deconv_q = args.deconv_q,
                        lmax=args.lmax,
                        mmax=args.mmax,
                        add_ghosts=args.add_ghosts,
                        ghost_amp=args.ghost_amp,
                        point_bias_mode = args.point_bias_mode,
                        point_bias=args.point_bias,
                        el0 = args.el0,
                        az0 = args.az0,
                        psi0 = args.psi0,
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
        if rank==0:
            with open(args.sim_tag+".yaml", "w") as simparamfile:
                yaml.dump(run_opts, simparamfile)

    return

if __name__ == "__main__":
    main()

