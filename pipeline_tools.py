'''
Collection of tools
'''
import numpy as np
import healpy as hp
from scipy import stats
import healpy as hp
import shutil
import tempfile as tf
import subprocess as sp
import os

spice_exe = 'spice'

def alm_qb2hp(alm):
    '''
    Convert quickbeam-formatted alm array to healpix-formatted array

    Arguments
    ---------
    alm : array-like
        Quickbeam-formatted alm coeff. as complex numpy array

    Returns
    -------
    alm_hp : array-like
        HEALPix-formatted alm coeff. as complex numpy array

    Notes
    -----
    Mmax is only allowed to be equal to lmax

    Quickbeam ouputs normalized blm coeff. i.e.
    Blm = sqrt( 4pi / (2el+1)) * blm, where blm
    are the SH coefficients.
    '''

    # assume alm.shape = (lmax+1, mmax+1)
    lmax = alm.shape[0] - 1

    alm_hp = np.zeros(hp.Alm.getsize(lmax, mmax=lmax), dtype=np.complex128)

    lm = hp.Alm.getlm(lmax)
    for idx in xrange(alm_hp.size):
        alm_hp[idx] = alm[lm[0][idx], lm[1][idx]]

    return alm_hp

def plot_stokes(stokes, name='', img_dir=None):

    # Calculate the beam profile 0-180 deg in theta and plot
    from scipy import stats

        # count1 = stats.binned_statistic(rs1[~badidx], es1[~badidx],
        #     statistic='count', bins=nbins)[0]
        # rs1b = stats.binned_statistic(rs1[~badidx], rs1[~badidx],
        #     statistic=np.nanmean, bins=nbins)[0]
        # eb1 = stats.binned_statistic(rs1[~badidx], es1[~badidx],
        #     statistic=np.nanmean, bins=nbins)[0]
        # ee1 = stats.binned_statistic(rs1[~badidx], es1[~badidx],
        #     statistic=np.nanstd, bins=nbins)[0]

    nbins = 50
    nside = hp.get_nside(stokes)
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))

    pow_I = stats.binned_statistic(theta, stokes[0], statistic=np.nanmean,
        bins=nbins)[0]

def equiang2hp(bmap, delta_az, delta_el, nside_out=1024, return_thetamax=False,
               apodize=False):
    '''
    Convert a rectangular map pixelized on a equiangular x-y (az-el) grid into a
    HEALPix map. Center input map will be put on north pole.

    Arguments
    ---------
    bmap : array-like
        Beam map on equiangular grid, shape (n_az, n_el)
    delta_az : float
        Az range in degrees
    delta_el : float
        El range in degrees

    Keyword arguments
    -----------------
    nside_out : int
        Nside parameter of output map (default : 1024)

    Returns
    -------
    bmap_out : array-like
        Beammap on HEALPix pixels

    Notes
    -----
    Returned map will be nonzero in a polar cap with
    theta_max = min(delta_az, delta_el)
    The pixelization of the input map should match GRASPs "az and el"
    convention.
    '''

    naz, nel = bmap.shape

    delta_az = np.radians(delta_az)
    delta_el = np.radians(delta_el)

    az_arr = np.linspace(-delta_az / 2., delta_az / 2., naz)
    el_arr = np.linspace(-delta_el / 2., delta_el / 2., nel)

    bmap_out = np.zeros(12 * nside_out**2, dtype=bmap.dtype)

    theta, phi = hp.pix2ang(nside_out, np.arange(bmap_out.size))
    phi += np.pi
    az = np.arctan(np.cos(phi) * np.tan(theta))
    el = np.arcsin(np.sin(phi) * np.sin(theta))

    az_ind = np.digitize(az, az_arr, right=True)
    el_ind = np.digitize(el, el_arr, right=True)

    #fixing edge cases
    az_ind[az_ind == naz] = naz - 1
    el_ind[el_ind == nel] = nel - 1

    bmap_out[:] = bmap[az_ind,el_ind]

    max_az = np.max(az_arr)
    max_el = np.max(el_arr)

    max_rad = min(max_az, max_el)

    if apodize:
        # apodize outer edge with Gaussian
        mu = 0.8 * max_rad
        std = 0.05 * max_rad

        annulus = np.where(np.logical_and(theta<max_rad, theta>0.8*max_rad))
        gauss = np.exp(-0.5*(theta-mu)**2 / std**2)
        bmap_out[annulus] *= gauss[annulus]

    bmap_out[theta>max_rad] = 0

    if return_thetamax:
        return bmap_out, max_rad

    return bmap_out

def azcuts2hp(bcuts, delta_az, delta_el, nside_out=1024, apodize=False):
    '''
    Convert fixed azimuth cuts into a
    HEALPix map. Center input cuts will be put on north pole.

    Arguments
    ---------
    bcuts : array-like
        Beam map on equal spacing cuts, shape (n_az, n_el)
    delta_az : float
        Az range in degrees
    delta_el : float
        El range in degrees

    Keyword arguments
    -----------------
    nside_out : int
        Nside parameter of output map (default : 1024)
    apodize : bool
        Whether to apodize the resulting map at its edge

    Returns
    -------
    bmap_out : array-like
        Beam map on HEALPix pixels

    Notes
    -----
    Returned map will be nonzero in a polar cap of size delta_el
    The pixelization of the input map should match GRASPs "az and el"
    convention.
    '''
    naz, nel = bcuts.shape

    delta_az = np.radians(delta_az)
    delta_el = np.radians(delta_el)

    az_arr = np.linspace(0, delta_az, naz)
    el_arr = np.linspace(0, delta_el, nel)

    bmap_out = np.zeros(12 * nside_out**2, dtype=bcuts.dtype)

    theta, phi = hp.pix2ang(nside_out, np.arange(bmap_out.size))
    phi += np.pi

    az_ind = np.digitize(phi, az_arr, right=True)
    el_ind = np.digitize(theta, el_arr, right=True)

    #fixing edge cases
    az_ind[az_ind == naz] = naz - 1
    el_ind[el_ind == nel] = nel - 1

    bmap_out[:] = bcuts[az_ind,el_ind]
    max_rad = delta_el

    if apodize:
        # apodize outer edge with Gaussian
        mu = 0.8 * max_rad
        std = 0.05 * max_rad

        annulus = np.where(np.logical_and(theta<max_rad, theta>0.8*max_rad))
        gauss = np.exp(-0.5*(theta-mu)**2 / std**2)
        bmap_out[annulus] *= gauss[annulus]

    bmap_out[theta>max_rad] = 0

    return bmap_out

# def uv2hp(bmap, delta_az, delta_el, nside_out=1024, return_thetamax=False):
def uv2hp(bmap, nside_out=1024, return_thetamax=False):
    '''
    Convert a rectangular map pixelized on a equiangular x-y (az-el) grid into a
    HEALPix map. Center input map will be put on north pole.

    Arguments
    ---------
    bmap : array-like
        Beam map on equiangular grid, shape (n_az, n_el)
    delta_az : float
        Az range in degrees
    delta_el : float
        El range in degrees

    Keyword arguments
    -----------------
    nside_out : int
        Nside parameter of output map (default : 1024)

    Returns
    -------
    bmap_out : array-like
        Beammap on HEALPix pixels

    Notes
    -----
    Returned map will be nonzero in a polar cap with
    theta_max = min(delta_az, delta_el)
    The pixelization of the input map should match GRASPs "az and el"
    convention.
    '''



    #delta_az = np.radians(delta_az)
    #delta_el = np.radians(delta_el)

    #u_arr = np.linspace(0., np.radians(180.), nu)
    #v_arr = np.linspace(0., np.radians(360.), nv)

    # u_arr = np.linspace(-1, 1, nu)
    # v_arr = np.linspace(-1, 1, nv)


    # U_arr, V_arr = np.meshgrid(u_arr, v_arr)



    #bmap_out = np.zeros(12 * nside_out**2, dtype=bmap.dtype)
    #theta, phi = hp.pix2ang(nside_out, np.arange(bmap_out.size))



    # print(np.min(np.sqrt(U_arr**2 + V_arr**2)))
    # print(np.max(np.sqrt(U_arr**2 + V_arr**2)))

    # phi += np.pi
    # az = np.arctan(np.cos(phi) * np.tan(theta))
    # el = np.arcsin(np.sin(phi) * np.sin(theta))

    # theta_uv, phi_uv = np.arcsin(np.sqrt(u_arr**2 + v_arr**2)), \
    #     np.arctan2(u_arr, v_arr)

    # theta_uv, phi_uv = np.arcsin(np.sqrt(U_arr**2 + V_arr**2)), \
    #     np.arctan2(U_arr, V_arr)

    nu, nv = bmap.shape

    theta_uv, phi_uv = np.meshgrid(np.linspace(0, np.pi, nu),
        np.linspace(0, 2*np.pi, nv))

    # print(np.min(theta_uv))
    # print(np.max(theta_uv))
    # print(np.min(phi_uv))
    # print(np.max(phi_uv))


    # print(type(bmap.flatten()[0]))
    # print(type(theta_uv.flatten()[0]))

    bmap = np.transpose(bmap)
    #print(np.shape(theta_uv))
    #print(np.shape(phi_uv))
    #print(np.shape(bmap))

    bmap2, hits = map_spin(theta_uv.flatten(), phi_uv.flatten(), bmap.flatten())

    #print(np.sum(hits))

    #u_ind = np.digitize(theta, theta_uv, right=True)
    #v_ind = np.digitize(phi, phi_uv, right=True)

    #u_ind, v_ind = stats.binned_statistic_2d(theta, phi, theta_uv, phi_uv)

    #fixing edge cases
    #u_ind[u_ind == nu] = nu - 1
    #v_ind[v_ind == nv] = nv - 1

    #bmap_out[:] = bmap[u_ind, v_ind]

    #max_az = np.max(az_arr)
    #max_el = np.max(el_arr)
    #max_rad = min(max_az, max_el)
    #bmap_out[theta>max_rad] = 0

    # return bmap_out
    return bmap2

def stitch_bmaps(main_bmap, wide_bmap, delta_az_main, delta_el_main,
    delta_az_wide, delta_el_wide,
    stitch_method='scale', **kwargs):
    '''
    Stitch a high resolution main beammap into a low resolution
    large fov map

    Keyword arguments
    -----------------
    stitch_method : str
        Stitch method, either "scale" or "linear"
    kwargs : {equiang2hp_opts}
    '''

    # Avoid apodizing before stitching.
    # This function always apodizes final outer edge.
    kwargs.pop('apodize', None)

    main_hp, max_rad = equiang2hp(main_bmap, delta_az_main, delta_el_main,
                                  return_thetamax=True, **kwargs)
    wide_hp, max_rad_wide = equiang2hp(wide_bmap, delta_az_wide, delta_el_wide,
                                       return_thetamax=True, **kwargs)
    
    nside = hp.get_nside(main_hp)

    theta, phi = hp.pix2ang(nside, np.arange(main_hp.size))

    # scale sidelobe by matching sum of pixels in outer 20% of main beam
    annulus = np.where(np.logical_and(theta<max_rad, theta>0.8*max_rad))
    ampl_main = np.sum(np.abs(main_hp[annulus]))
    ampl_wide = np.sum(np.abs(wide_hp[annulus]))

    ratio = ampl_main / ampl_wide

    if stitch_method == 'scale':
        wide_hp *= ratio

        cos2 = np.cos(np.pi * (theta - 0.8*max_rad) / 2. / 0.2 / max_rad)**2
        sin2 = np.sin(np.pi * (theta - 0.8*max_rad) / 2. / 0.2 / max_rad)**2

        main_hp[annulus] *= cos2[annulus]
        wide_hp[annulus] *= sin2[annulus]

    elif stitch_method == 'linear':
        d_theta = 0.2 * max_rad
        alpha = (1 - 1/ratio) / d_theta
        lin = -alpha * theta + 1 + alpha * 0.8*max_rad
        main_hp[annulus] *= lin[annulus]

    # stitch
    if stitch_method == 'scale':
        wide_hp[theta<0.8*max_rad] = 0

    elif stitch_method == 'linear':
        wide_hp[theta<max_rad] = 0

    main_hp += wide_hp

    # apodize outer edge with Gaussian
    mu = 0.8 * max_rad_wide
    std = 0.05 * max_rad_wide

    annulus = np.where(np.logical_and(theta<max_rad_wide, theta>0.8*max_rad_wide))
    gauss = np.exp(-0.5*(theta-mu)**2 / std**2)
    main_hp[annulus] *= gauss[annulus]

    return main_hp

def e2iqu(e_co, e_cross, delta_az, delta_el, vpol=False,
          basis='spherical', rot_angle=None,
          e_co_wide=None, e_cross_wide=None,
          delta_az_wide=None, delta_el_wide=None,
          **kwargs):
    '''
    Convert co- and cross-polar fields to I, Q and U beammaps
    on a spherical basis.

    Arguments
    ---------
    e_co : array-like
        Complex co-polar field centered on beam centroid
    e_cross : array-like
        Complex cross-polar field centered on beam centroid
    delta_az : float
        Az range in degrees
    delta_el : float
        El range in degrees

    Keyword arguments
    -----------------
    vpol : bool
        Also compute and return V (circular polarization)
    basis : str
        The polarization basis of the output. Can be either
        "spherical" or "grasp". In the latter case, just bin
        the maps on healpy pixels and form I, Q, U and V.
        Otherwise, rotate to spherical basis, see eq. 18-21
        Challinor et al. 2000. (default : spherical)
    e_co_wide : array-like
        Complex wide-angle co-polar field centered on beam centroid
    e_cross_wide : array-like
        Complex wide-angle cross-polar field centered on beam centroid
    delta_az_wide : float
        Az range of wide-angle fields in degrees
    delta_el_wide : float
        El range of wide-angle fields in degrees
    kwargs : {equiang2hp_opts}

    Returns
    -------
    bmaps : list of array-like
        List of I, Q, U, (V) (real) beammaps

    Notes
    -----
    Co- and cross-polar basis is defined as the convention used by
    GRASP (`linear pol`). This convention is equivalent to Ludwig's 3rd
    definition but interchanges the co and cross unit vectors.
    '''

    # bin input on HEALPix grid
    if e_co_wide is not None and delta_az_wide and delta_el_wide:
        e_co = stitch_bmaps(e_co, e_co_wide, delta_az, delta_el,
                            delta_az_wide, delta_el_wide,
                            **kwargs)
    else:
        e_co = equiang2hp(e_co, delta_az, delta_el, **kwargs)

    if e_cross_wide is not None and delta_az_wide and delta_el_wide:
        e_cross = stitch_bmaps(e_cross, e_cross_wide, delta_az, delta_el,
                            delta_az_wide, delta_el_wide,
                            **kwargs)
    else:
        e_cross = equiang2hp(e_cross, delta_az, delta_el, **kwargs)

    nside = hp.get_nside(e_co)

    # squared combinations of fields
    e_co2 = np.abs(e_co)**2
    e_cross2 = np.abs(e_cross)**2
    e_cocr = e_co * np.conj(e_cross)

    # create Stokes parameters
    I = e_co2 + e_cross2

    if basis == 'grasp':

        Q = (e_co2 - e_cross2)
        U = -2 * np.real(e_cocr)

        # Aoplying a rotation angle
        if rot_angle is not None:

            print('Applying rotation angle')

            Qi = np.cos(rot_angle) * Q + np.sin(rot_angle) * U
            Ui = -np.sin(rot_angle) * Q + np.cos(rot_angle) * U

            Q = Qi
            U = Ui

    elif basis == 'spherical':

        _, phi = hp.pix2ang(nside, np.arange(e_co.size))

        s2 = np.sin(2*phi)
        c2 = np.cos(2*phi)

        # note we change the sign of the cosine (phi -> -phi+pi)
        # with respect to Challinors expressions to get those of Hivon
        Q = (e_co2 - e_cross2) * c2
        Q += 2 * np.real(e_cocr) * s2

        U = -(e_co2 - e_cross2) * s2
        U += 2 * np.real(e_cocr) * c2

    if vpol:
        V = -2 * np.imag(e_cocr)
        return [I, Q, U, V]
    else:
        return [I, Q, U]

def map_spin(theta, phi, sig, norm=False, nside=64, dt2use=np.complex64):

    npix = hp.nside2npix(nside)
    map1 =  np.zeros(npix,dtype=dt2use)
    hits1 = np.zeros(npix, dtype=np.float)
    hpix = hp.ang2pix(nside, theta, phi)
#    maths.idxs_inc(hits1, hpix)
#    maths.idxs_add(map1, hpix, sig)


    '''
    for p,t,r1,d1 in zip(hpix,sig,ra,dec):
        map1[p] += t
        ra1[p] += r1
        dec1[p] += d1
        hits1[p] += 1

    # Hits normalizing
    ra1 = ra1/hits1
    dec1 = dec1/hits1
    map1 = np.absolute(map1)/hits1
    # Finding a distance
    dist1 = np.sqrt(ra1**2+dec1**2)

    '''

    if norm:
        map1[hits1==0] = np.nan
        map1[hits1!=0] /= hits1[hits1!=0]

    return map1, hits1

def euv2iqu(e_co, e_cross, nu, nv, vpol=False,
    basis='spherical', rot_angle=None, **kwargs):
    '''
    Convert co- and cross-polar fields to I, Q and U beammaps
    on a spherical basis.

    Arguments
    ---------
    e_co : array-like
        Complex co-polar field centered on beam centroid
    e_cross : array-like
        Complex cross-polar field centered on beam centroid
    delta_az : float
        Az range in degrees
    delta_el : float
        El range in degrees

    Keyword arguments
    -----------------
    vpol : bool
        Also compute and return V (circular polarization)
    basis : str
        The polarization basis of the output. Can be either
        "spherical" or "grasp". In the latter case, just bin
        the maps on healpy pixels and form I, Q, U and V.
        Otherwise, rotate to spherical basis, see eq. 18-21
        Challinor et al. 2000. (default : spherical)

    kwargs : {equiang2hp_opts}

    Returns
    -------
    bmaps : list of array-like
        List of I, Q, U, (V) (real) beammaps

    Notes
    -----
    Co- and cross-polar basis is defined as the convention used by
    GRASP (`linear pol`). This convention is equivalent to Ludwig's 3rd
    definition but interchanges the co and cross unit vectors.
    '''

    # bin input on HEALPix grid


    e_co = uv2hp(e_co, **kwargs)
    e_cross = uv2hp(e_cross,  **kwargs)

    nside = hp.get_nside(e_co)

    # squared combinations of fields
    e_co2 = np.abs(e_co)**2
    e_cross2 = np.abs(e_cross)**2
    e_cocr = e_co * np.conj(e_cross)

    # create Stokes parameters
    I = e_co2 + e_cross2

    if basis == 'grasp':

        Q = (e_co2 - e_cross2)
        U = -2 * np.real(e_cocr)

        # Aoplying a rotation angle
        if rot_angle is not None:

            print('Applying rotation angle')

            Qi = np.cos(rot_angle) * Q + np.sin(rot_angle) * U
            Ui = -np.sin(rot_angle) * Q + np.cos(rot_angle) * U

            Q = Qi
            U = Ui

    elif basis == 'spherical':

        _, phi = hp.pix2ang(nside, np.arange(e_co.size))

        s2 = np.sin(2*phi)
        c2 = np.cos(2*phi)

        # note we change the sign of the cosine (phi -> -phi+pi)
        # with respect to Challinors expressions to get those of Hivon
        Q = (e_co2 - e_cross2) * c2
        Q += 2 * np.real(e_cocr) * s2

        U = -(e_co2 - e_cross2) * s2
        U += 2 * np.real(e_cocr) * c2

    if vpol:
        V = -2 * np.imag(e_cocr)
        return [I, Q, U, V]
    else:
        return [I, Q, U]

def parse_beams_mpi(main_beams, outdir, comm=None, wide_beams=None,
                    lmax=1000, **kwargs):
    '''
    Load up GRASP output, convert to harmonic coeff. and store
    beam pkl files and blm npy files.

    Arguments
    ---------
    main_beams : array_like, str
        Path(s) to pickle file(s) containing GRASP output for
        main beams
    outdir : str
        Path to ouput directory

    Keyword Arguments
    -----------------
    comm : MPI communicator
        If not set, do not use MPI
    wide_beams : array_like, str
        Path(s) to pickle file(s) containing GRASP output for
        wide beams for each main beam. Stitched to main beam.
        Needs to be same size as main_beams (or single path)
    kwargs : {e2iqu_opts}

    Notes
    -----
    Example use:

    S = ScanStrategy()
    comm = S._comm
    parse_beams_mpi(main_beams, outdir, comm=comm)

    '''

    mpi = False
    if isinstance(comm, MPI.Intracomm):
        mpi = True

    main_beams = np.atleast_1d(main_beams)

    if wide_beams:
        wide_beams = np.atleast_1d(wide_beams)

        if wide_beams.size != 1 or wide_beams.size != main_beams.size:
            raise ValueError('length of wide_beams array not understood')
        if wide_beams.size == 1:
            tmp = np.empty_like(main_beams)
            tmp[:] = wide_beams[0]
            wide_beams = tmp

    else:
        wide_beams = np.empty_like(main_beams)
        wide_beams[:] = ''

    rank = 0
    size = 1
    if mpi:
        rank = comm.Get_rank()
        size = comm.Get_size()

    # scatter beams over cores
    main_beams = np.array_split(main_beams, size)[rank]
    wide_beams = np.array_split(wide_beams, size)[rank]

    for main, wide in zip(main_beams, wide_beams):

        if wide:
            wide_path, wide_name = os.path.split(wide)
            wide_fields = pickle.load(open(opj(wide_path, wide_name), 'rb'))
            prop_name = wide_name.replace('fields', 'prop')
            wide_prop = pickle.load(open(opj(wide_path, prop_name), 'rb'))

            wide_co = wide_fields['e_co']
            wide_cx = wide_fields['e_cx']

            cr = wide_prop['cr'] # [azmin, elmin, azmax, elmax]
            wide_az = cr[2] - cr[0]
            wide_el = cr[3] - cr[1]

        else:
            wide_co = None
            wide_cx = None
            wide_az = None
            wide_el = None

        main_path, main_name = os.path.split(main)
        main_fields = pickle.load(open(opj(main_path, main_name), 'rb'))
        prop_name = main_name.replace('fields', 'prop')
        main_prop = pickle.load(open(opj(main_path, prop_name), 'rb'))

        main_co = main_fields['e_co']
        main_cx = main_fields['e_cx']

        cr = main_prop['cr'] # [azmin, elmin, azmax, elmax]
        main_az = cr[2] - cr[0]
        main_el = cr[3] - cr[1]

        # convert e fields to stokes
        nside_out = kwargs.pop('nside_out', 2048)
        stokes = e2iqu(e_co, e_cx, d_az, d_el, vpol=False,
                             basis='grasp', nside_out=nside_out,
                             e_co_wide=we_co, e_cross_wide=we_cx, delta_az_wide=wd_az,
                             delta_el_wide=wd_el, **kwargs)


        blm_stokes = hp.map2alm(stokes, lmax, pol=False)
        blmm2, blmp2 = beam_tools.get_pol_beam(blm_stokes[1], blm_stokes[2])
        blm = blm_stokes[0]
        blm = np.array([blm, blmm2, blmp2], dtype=np.complex128)

        # save blm npy file
        po_file = opj(outdir, main_name.replace('.pkl', '.npy'))
        np.save(po_file, blm)

        # Create dict with beam options and store in pkl file
        # set common opts
        beam_opts = dict(lmax=lmax, deconv_q=True, normalize=True,
                         po_file=po_file, cross_pol=True, btype='PO')

        # set A and B specifics

        ####### ADD LINES HERE THAT EXTRACT POLANG AND CENTROID FROM
        ####### main_prop AND POPULATE a_opts and b_opts

        # SOMETHING LIKE THIS
        #az_a, el_a = M.get_offsets(channels=chn_a, pol=False)
        #az_b, el_b = M.get_offsets(channels=chn_b, pol=False)
        #polang_a = M.get_polang(channels=chn_a)
        #polang_b = M.get_polang(channels=chn_b)

        #a_opts = dict(az=az_a, el=el_a, polang=polang_a, name=name_a,
        #              dead=bad_a, pol='A')
        #b_opts = dict(az=az_b, el=el_b, polang=polang_b, name=name_b,
        #              dead=bad_b, pol='B')

        #a_opts.update(beam_opts)
        #b_opts.update(beam_opts)

        # store in pickle file
        with open(opj(outdir, main_name), 'wb') as handle:
            pickle.dump([a_opts, b_opts], handle, protocol=pickle.HIGHEST_PROTOCOL)

    if mpi:
        comm.Barrier()

def parse_beam(idx, nu, wide_dir=None, **kwargs):


    if wide_dir:
        wide_150 = opj(wide_dir, 'wide5_150.pkl')
        wide_094 = opj(wide_dir, 'wide5_90.pkl')

        pkl_file = open(wide_150, 'rb')
        wbeam_150 = pickle.load(pkl_file)
        pkl_file.close()

        pkl_file = open(wide_094, 'rb')
        wbeam_094 = pickle.load(pkl_file)
        pkl_file.close()

        cr = wbeam_150['cr'] # [azmin, elmin, azmax, elmax]
        wd_az = cr[2] - cr[0]
        wd_el = cr[3] - cr[1]

        we_co_150 = wbeam_150['e_co']
        we_cx_150 = wbeam_150['e_cx']

        we_co_094 = wbeam_094['e_co']
        we_cx_094 = wbeam_094['e_cx']
    else:
         we_co_094 = None
         we_cx_094 = None
         we_co_150 = None
         we_cx_150 = None
         wd_az = None
         wd_el = None

    det = 'det{:04d}_{:d}'.format(int(idx), int(np.mean(nus)))
    grasp_dir = ''

    pfile = opj(grasp_dir, '{}_fields.pkl'.format(det))
    grasp_out = pickle.load(open(pfile, 'rb'))
    beam_prop = pickle.load(open(pfile.replace('_fields', '_prop'), 'rb'))

    cr = grasp_out['cr'] # [azmin, elmin, azmax, elmax]
    d_az = cr[2] - cr[0]
    d_el = cr[3] - cr[1]
    numel = grasp_out['numel'] # [Naz, Nel]
    e_co = grasp_out['e_co']
    e_cx = grasp_out['e_cx']
    # convert e fields to stokes
    nside_out = kwargs.pop('nside_out', 2048)
    stokes = e2iqu(e_co, e_cx, d_az, d_el, vpol=False,
                         basis='grasp', nside_out=nside_out,
                         e_co_wide=we_co, e_cross_wide=we_cx, delta_az_wide=wd_az,
                         delta_el_wide=wd_el, **kwargs)

    # save input maps
    sam.write_map(opj(outdir, 'input_maps', bfile+'.fits'), stokes)


    blm_stokes = hp.map2alm(stokes, lmax, pol=False)
    blmm2, blmp2 = beam_tools.get_pol_beam(blm_stokes[1], blm_stokes[2])
    blm = blm_stokes[0]

    blm = np.array([blm, blmm2, blmp2], dtype=np.complex128)
    # save npy file
    po_file = opj(outdir, bfile+'.npy')
    np.save(po_file, blm)

    # set common opts
    beam_opts = dict(lmax=lmax, deconv_q=True, normalize=True,
                     po_file=po_file, cross_pol=True, btype='PO')

    # set A and B specifics
    az_a, el_a = M.get_offsets(channels=chn_a, pol=False)
    az_b, el_b = M.get_offsets(channels=chn_b, pol=False)
    polang_a = M.get_polang(channels=chn_a)
    polang_b = M.get_polang(channels=chn_b)

    a_opts = dict(az=az_a, el=el_a, polang=polang_a, name=name_a,
                  dead=bad_a, pol='A')
    b_opts = dict(az=az_b, el=el_b, polang=polang_b, name=name_b,
                  dead=bad_b, pol='B')

    a_opts.update(beam_opts)
    b_opts.update(beam_opts)

    # store in pickle file
    with open(opj(outdir, bfile+'.pkl'), 'wb') as handle:
        pickle.dump([a_opts, b_opts], handle, protocol=pickle.HIGHEST_PROTOCOL)


def mask_good(m, badval=hp.UNSEEN, rtol=1.e-5, atol=1.e-8,
              badnan=True, badinf=True):
    """Returns a bool array with ``False`` where m is close to badval,
    NaN or inf.

    Parameters
    ----------
    m : a map (may be a sequence of maps)
    badval : float, optional
        The value of the pixel considered as bad (:const:`UNSEEN` by default)
    rtol : float, optional
        The relative tolerance
    atol : float, optional
        The absolute tolerance
    badnan : bool, optional
        If True, also mask NaN values
    badinf : bool, optional
        If True, also mask inf values

    Returns
    -------
    a bool array with the same shape as the input map, ``False`` where input map is
    close to badval, NaN or inf, and ``True`` elsewhere.

    See Also
    --------
    mask_bad, ma

    Examples
    --------
    >>> import healpy as hp
    >>> m = np.arange(12.)
    >>> m[3] = hp.UNSEEN
    >>> m[4] = np.nan
    >>> mask_good(m)
    array([ True,  True,  True, False, False,  True,  True,  True,  True,
            True,  True,  True], dtype=bool)
    """
    m = np.asarray(m)
    mask = np.ones_like(m, dtype=bool)
    if badnan:
        mask &= ~np.isnan(m)
    if badinf:
        mask &= np.isfinite(m)
    mask[mask] = hp.mask_good(m[mask], badval=badval, rtol=rtol, atol=atol)
    return mask

def rotate_map(m, coord=['C', 'G'], rot=None, mask=None, pixels=None,
               pol_axis=[0.,0.,1.]):
    '''
    Rotate an input map from one coordinate system to another or to place a
    particular point at centre in rotated map. This does the proper Q and U
    Stokes rotation. Sign of Q U rotation should be correct for inverse
    rotation back to original coords (psi -> -psi)

    e.g. m = rotate_map(m, rot=[phi,90.-theta,0.])

    takes point at original theta, phi to new coord ra=dec=0

    Arguments
    ---------
    m : array_like
        A single map or two (Q,U) or three (I,Q,U) maps
    coord : list of two coordinates, optional.
        Coordinates to rotate between.  Default: ['C', 'G']
    rot : scalar or sequence, optional
        Describe the rotation to apply.
        In the form (lon, lat, psi) (unit: degrees) : the point at
        longitude lon and latitude lat will be at the center of the rotated
        map. An additional rotation of angle psi around this direction is applied
    mask : 1D array
        If supplied, only pixels in the *rotated map* that fall within the mask
        are handled.
    pixels : 1D array
        If supplied, only pixels in the *rotated map* that are also in this list
        are handled. Overrides `mask`.
    pol_axis : 3-vector, optional
        Axis normal to the plane in which the Q/U coordinates are defined.

    Returns
    -------
    mr : array_like
        The rotated map.
    '''

    m = np.atleast_2d(m)

    if m.ndim > 2:
        raise ValueError(
            'Input map array must have no more than two dimensions')
    if m.shape[0] not in [1,2,3]:
        raise ValueError(
            'Input map must have 1 (T only), 2 (Q/U only) or 3 (T/Q/U) columns')

    res = hp.get_nside(m)
    pol = m.shape[0] in [2,3]
    pol_only = m.shape[0] == 2

    if rot is None:
        #use default coord transform C->G
        R = hp.Rotator(coord=coord, inv=True)
    else:
        R = hp.Rotator(rot=rot, inv=True)

    try:
        # qpoint is a 50% speedup
        import qpoint as qp
        Q = qp.QPoint()
        use_qpoint = True
    except ImportError:
        use_qpoint = False

    if pixels is None:
        if mask is not None:
            pixels, = np.where(mask)
        else:
            pixels = np.arange(len(m[0]))

    # rotate new coordinate system to original coordinates
    theta, phi = hp.pix2ang(res, pixels)
    mtheta, mphi = R(theta, phi)

    mr = np.full_like(m, np.nan)

    if use_qpoint:
        ra = np.degrees(mphi)
        dec = 90. - np.degrees(mtheta)
        if not pol_only:
            mr[0, pixels] = Q.get_interp_val(m[0], ra, dec)
    else:
        if not pol_only:
            mr[0, pixels] = hp.get_interp_val(m[0], mtheta, mphi)

    if not pol:
        return mr.squeeze()

    #interpolate Q and U (better before or after rot?)
    if use_qpoint:
        mr[-2, pixels] = Q.get_interp_val(m[-2], ra, dec)
        mr[-1, pixels] = Q.get_interp_val(m[-1], ra, dec)
    else:
        mr[-2, pixels] = hp.get_interp_val(m[-2], mtheta, mphi)
        mr[-1, pixels] = hp.get_interp_val(m[-1], mtheta, mphi)

    rotate_qu_radial(
        mr, coord=coord, rot=rot, pol_axis=pol_axis, inplace=True)

    return mr

def rotate_qu_radial(m, coord=['C', 'G'], rot=None, pol_axis=[0.,0.,1.],
                     inplace=False):
    """
    Rotate a map's Q/U components to radial Qr/Ur coordinates.

    Arguments
    ---------
    m : array_like
        1D (T-only) or 2D (QU or TQU) map to which the rotation is applied.
    coord : list of two coordinates, optional.
        Coordinates to rotate between.  Default: ['C', 'G']
    rot : scalar or sequence, optional
        Describe the rotation to apply.
        In the form (lon, lat, psi) (unit: degrees) : the point at
        longitude lon and latitude lat will be at the center of the rotated
        map. An additional rotation of angle psi around this direction is applied
    pol_axis : 3-vector, optional
        Axis normal to the plane in which the Q/U coordinates are defined.
    inplace : bool, optional
        If True, the rotation is applied in-place in memory.

    Returns
    -------
    m : array_like
        The input map, with the Q/U components rotated to radial
        Qr/Ur coordinates.
    """

    if not inplace:
        m = m.copy()
    nside = hp.get_nside(m)
    pixels, = np.where(mask_good(m[0]))
    mq, mu = m[-2, pixels], m[-1, pixels]

    if rot is None:
        R = hp.Rotator(coord=coord, inv=True)
    else:
        R = hp.Rotator(rot=rot, inv=True)

    vec = hp.pix2vec(nside, pixels)
    vec0 = np.asarray(pol_axis)
    mvec = R(vec)
    mvec0 = R(vec0)
    del vec

    # calculate orientation of local meridian
    # based on Healpix rotate_coord
    mvec = np.asarray(mvec).T
    x = np.cross(vec0, mvec)
    sin_psi = np.dot(x, mvec0)
    cos_psi = np.dot(np.cross(x, mvec), mvec0)
    del mvec
    del x

    norm = sin_psi * sin_psi + cos_psi * cos_psi
    s2psi = 2. * sin_psi * cos_psi / norm
    c2psi = 1. - 2. * sin_psi * sin_psi / norm
    del norm
    del sin_psi
    del cos_psi

    # Rotate to new Q and U wrt to local meridian, in place
    m[-2, pixels], m[-1, pixels] = mq * c2psi + mu * s2psi, mu * c2psi - mq * s2psi

    return m

def mask_bad(m, badval=hp.UNSEEN, rtol=1.e-5, atol=1.e-8,
             badnan=True, badinf=True):
    """Returns a bool array with ``True`` where m is close to badval,
    NaN or inf.

    Parameters
    ----------
    m : a map (may be a sequence of maps)
    badval : float, optional
        The value of the pixel considered as bad (:const:`UNSEEN` by default)
    rtol : float, optional
        The relative tolerance
    atol : float, optional
        The absolute tolerance
    badnan : bool, optional
        If True, also mask NaN values
    badinf : bool, optional
        If True, also mask inf values

    Returns
    -------
    mask
      a bool array with the same shape as the input map, ``True`` where input map is
      close to badval, NaN or inf, and ``False`` elsewhere.

    See Also
    --------
    mask_good

    Examples
    --------
    >>> import healpy as hp
    >>> import numpy as np
    >>> m = np.arange(12.)
    >>> m[3] = hp.UNSEEN
    >>> m[4] = np.nan
    >>> mask_bad(m)
    array([False, False, False,  True,  True, False, False, False, False,
           False, False, False], dtype=bool)
    """
    m = np.asarray(m)
    mask = np.zeros_like(m, dtype=bool)
    if badnan:
        mask |= np.isnan(m)
    if badinf:
        mask |= np.isinf(m)
    mask[~mask] = hp.mask_bad(m[~mask], badval=badval, rtol=rtol, atol=atol)
    return mask

def latlon_mask(nside=512, latrange=(-55, -15), lonrange=(18, 80), coord=None):
    """
    Create a mask that is True where lat and lon lie in a given range.
    Default result is for Spider cmb region in equatorial coordinates.

    Arguments
    =========
    nside : int
        The nside at which to make the mask
    latrange : (min, max) 2-tuple of floats
        Min and max latitude in degrees. In the range -90 to +90
    lonrange : (min, max) 2-tuple of floats
        Min and max longitude in degrees. In the range -180 to +180
    coord : 2-tuple of strings (or list, etc)
        Pair of (input_coord, output_coord) as per rotate_map.

    Returns
    =======
    Boolean array/healpix map that is True in selected region
    """
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    lat = 90 - np.rad2deg(theta)
    lon = np.rad2deg(phi)
    lon[lon > 180] -= 360
    ret = np.logical_and(
            np.logical_and(lon > lonrange[0], lon < lonrange[1]),
            np.logical_and(lat > latrange[0], lat < latrange[1]))
    if coord is not None and len(coord) == 2:
        ret = rotate_map(ret, coord=coord).astype(bool)
    return ret

def spice(map_in, nlmax=None, mask=None, weight=None, fwhm=None, beam=None,
    apodizetype=None, apodizesigma=None, thetamax=None,
    decouple=False, pixwin=False, subav=False, subdipole=False,
    polarization=None, parfile=None, clfile=None, verbose=True,
    outroot=None, use_temp=False,
    kernelsfile=None, return_kernel=False, **kwargs):
    """Estimate cl(s) from input map(s).  Python interface to PolSpice.
    Input maps as arrays or filenames.  Intermediate files are stored
    in a temporary root that is discarded upon completion.

    Parameters
    ----------
    map_in : array or tuple of array
      A map or list of maps (3 maps for polarization)
    nlmax : int, scalar, optional
      Maximum l for cl.  Default: 3*nside - 1
    mask : array or string, optional
      Mask array, same shape as map, or string path to fits file
      Should be True for pixels to include, False for pixels to exclude.
    maskp : array or string, optional
      Polarization mask array (if different than mask), same shape as map,
      or string path to fits file.
    weight : array or string, optional
      Weight array, same shape as map, or string path to fits file
    weightp : array or string, optional
      Weight array for polarization maps (if different than weight),
      same shape as map, or string path to fits file.
    apodizetype : int, optional
      Apodization type, Gaussian (0) or cosine (1)
    apodizesigma : float, optional
      Apodization width (degrees). Default: disabled
      Values close to thetamax are recommended.
    thetamax : float, optional
      Maximum integration angle (degrees). Default: 180
    fwhm : float, optional
      beam width to deconvolve, in **radians**: Default: disabled
    beam : array or string, optional
      beam window function B_ell for correcting the output spectra,
      starting with the ell=0 mode.
    polarization : bool, optional
      If True, treat the input map as polarized.  If not supplied,
      assume True if `map_in` is a list of 3 maps.
    decouple : bool, optional
      If True, return T/E/B spectra, otherwise return T/Q/U spectra
      Default: False
    pixwin : bool or string, optional
      If True, apply default pixel window function for the map nside
      If string, apply supplied window function
      Default: False
    subav : bool, optional
      subtract best-fit monopole from the map
    subdipole : bool, optional
      subtract best-fit monopole and dipole from the map
    windowfile : string, optional
      If supplied and exists, sets the windowfilein parameter to avoid
      recomputing the window.  If supplied and missing, sets the
      windowfileout parameter to store the computed window for future use.
    parfile, mapfile, maskfile[p], weightfile[p], beamfile, clfile, pixelfile :
        string, optional
      Explicit file names for input and output files, stored
      in a location other than the temporary root.
    map[file]2, mask[file][p]2, weight[file][p]2, beam[file]2 :
        array_like or string, optional
      Second map(s) and weight functions for calculating cross-spectra
    symmetric_cl : bool, optional
    verbose : bool, optional
      print stuff or not.
    outroot : string, optional
      output prefix.  if not set, files are stored in a temporary directory
      created using tempfile.mkdtemp()
    use_temp : bool, optional
      allows one to specify an outroot, but still retain the temp directories
    kernelsfile : string or bool, optional
      If True, store the spice kernel in the output root.  Otherwise, should
      be an explicit filename to store in a location other than the output root.
      If False or None, the kernel is not stored.
    return_kernel : bool, optional
      If True, return the spice kernel along with the spectrum.

    Returns
    -------
    cls : array or list of arrays
      power spectrum computed by spice.
      Polarization order (with decouple=True): [TT, EE, BB, TE, TB, EB]
    kernel : array, optional
        Estimator kernel, returned if return_kernel is True.
    """

    executable = kwargs.pop('executable', spice_exe)

    # temporary directory
    if outroot is None or use_temp:
        tmproot = tf.mkdtemp(dir=outroot)
        filetag = ''
    else:
        tmproot, filetag = os.path.split(outroot)
        if not os.path.exists(tmproot):
            os.mkdir(tmproot)
    if filetag and not filetag.endswith('_'):
        filetag += '_'

    def opt(val):
        return 'YES' if val is True else 'NO' if val is False else val

    params = {}

    def get_file(filename, arg, filearg, altarg=None,
                 write_func=write_map):
        if isinstance(arg, str) and arg in kwargs:
            arg = kwargs.pop(arg)
        argname = filearg
        filearg = kwargs.pop(filearg, kwargs.pop(altarg, None))
        filename = os.path.join(tmproot, filetag + filename)
        if isinstance(arg, str):
            arg = None
            filearg = arg
        elif arg is True or arg is False:
            arg = None
            filearg = opt(arg)
        elif filearg is None and arg is not None:
            filearg = filename
            if write_func is write_map:
                arg = np.array(arg)
                arg[mask_bad(arg)] = hp.UNSEEN
            write_func(filearg, arg)
        if filearg is not None and filearg not in ['YES', 'NO'] and \
                not os.path.exists(filearg):
            raise OSError('{} not found'.format(filename))
        params[argname] = opt(filearg)

    # deal with input files
    get_file('map.fits', map_in, 'mapfile')
    if not isinstance(map_in, str) and polarization is None:
        polarization = len(map_in) == 3

    if clfile is None:
        clfile = os.path.join(tmproot, filetag + 'cl.fits')
    params['clfile'] = clfile

    get_file('mask.fits', mask, 'maskfile')
    get_file('maskp.fits', 'maskp', 'maskfilep')
    get_file('weight.fits', weight, 'weightfile')
    get_file('weightp.fits', 'weightp', 'weightfilep')
    get_file('beam.fits', beam, 'beam_file', 'beamfile',
             write_func=hp.write_cl)
    get_file('pixwin.fits', pixwin, 'pixelfile',
             write_func=hp.write_cl)
    if np.isscalar(fwhm):
        params['beam'] = np.degrees(fwhm) * 60

    get_file('map2.fits', 'map2', 'mapfile2')
    get_file('mask2.fits', 'mask2', 'maskfile2')
    get_file('maskp2.fits', 'maskp2', 'maskfilep2')
    get_file('weight2.fits', 'weight2', 'weightfile2')
    get_file('weightp2.fits', 'weightp2', 'weightfilep2')
    get_file('beam2.fits', 'beam2', 'beam_file2', 'beamfile2',
             write_func=hp.write_cl)
    fwhm2 = kwargs.pop('fwhm2', None)
    if np.isscalar(fwhm2):
        params['beam2'] = np.degrees(fwhm2) * 60
    params['symmetric_cl'] = opt(kwargs.pop('symmetric_cl', None))

    # create and populate parameter file
    if parfile is None:
        parfile = os.path.join(tmproot, filetag + 'spice.par')
    f = open(parfile, 'w')

    for k, v in params.items():
        if v is not None:
            f.write('{} = {}\n'.format(k, v))

    if nlmax is not None:
        f.write('nlmax = {:d}\n'.format(nlmax))
    f.write('polarization = {}\n'.format(opt(polarization)))

    ########################################################################
    f.write('tolerance = 1.e-6\n')
    ########################################################################

    f.write('decouple = {}\n'.format(opt(decouple)))
    f.write('subav = {}\n'.format(opt(subav)))
    f.write('subdipole = {}\n'.format(opt(subdipole)))
    if apodizetype in [0,1]:
        f.write('apodizetype = {:d}\n'.format(apodizetype))
    if np.isscalar(apodizesigma):
        f.write('apodizesigma = {:f}\n'.format(apodizesigma))
    if np.isscalar(thetamax):
        f.write('thetamax = {:f}\n'.format(thetamax))
    f.write('fits_out = YES\n')

    if verbose in [True, False]:
        f.write('verbosity = {}\n'.format(opt(verbose)))
    elif verbose in range(2):
        f.write('verbosity = {:d}\n'.format(verbose))

    windowfile = kwargs.pop('windowfile', None)
    if windowfile is not None:
        if windowfile is True:
            windowfile = os.path.join(tmproot, filetag + 'window.fits')
        if os.path.exists(windowfile):
            f.write('windowfilein = {}\n'.format(windowfile))
        else:
            f.write('windowfileout = {}\n'.format(windowfile))

    if return_kernel:
        if kernelsfile is None:
            kernelsfile = True
    if kernelsfile is not None and kernelsfile is not False:
        if kernelsfile is True:
            kernelsfile = os.path.join(tmproot, filetag + 'kernels.fits')
        f.write('kernelsfileout = {}\n'.format(kernelsfile))

    f.close()

    if len(kwargs.keys()):
        if outroot is None:
            shutil.rmtree(tmproot)
        raise TypeError("spice got unexpected keyword argument(s): {}".format(
                ", ".join(kwargs.keys())))

    # run spice
    stdout = None if verbose else open(os.devnull, 'w')
    try:
        sp.check_call([executable, '-optinfile', parfile], stdout=stdout,
                       stderr=sp.STDOUT)
    except (sp.CalledProcessError, KeyboardInterrupt):
        if outroot is None:
            shutil.rmtree(tmproot)
        raise
    if stdout:
        stdout.close()

    # read in output
    cls = np.asarray(hp.read_cl(clfile))
    if return_kernel:
        # do this by hand because kernel isn't stored in the standard hdu...
        import astropy.io.fits as pf
        hdus = pf.open(kernelsfile)
        kernel = hdus[0].data.astype(float)
        hdus.close()

    # cleanup
    if outroot is None or use_temp:
        shutil.rmtree(tmproot)

    # return
    if return_kernel:
        return cls, kernel
    return cls

def rotate_map(m, coord=['C', 'G'], rot=None, mask=None, pixels=None,
    pol_axis=[0.,0.,1.]):
    """
    Rotate an input map from one coordinate system to another or to place a
    particular point at centre in rotated map. This does the proper Q and U
    Stokes rotation. Sign of Q U rotation should be correct for inverse
    rotation back to original coords (psi -> -psi)

    e.g. m = rotate_map(m, rot=[phi,90.-theta,0.])

    takes point at original theta, phi to new coord ra=dec=0

    Arguments
    ---------
    m : array_like
        A single map or two (Q,U) or three (I,Q,U) maps
    coord : list of two coordinates, optional.
        Coordinates to rotate between.  Default: ['C', 'G']
    rot : scalar or sequence, optional
        Describe the rotation to apply.
        In the form (lon, lat, psi) (unit: degrees) : the point at
        longitude lon and latitude lat will be at the center of the rotated
        map. An additional rotation of angle psi around this direction is applied
    mask : 1D array
        If supplied, only pixels in the *rotated map* that fall within the mask
        are handled.
    pixels : 1D array
        If supplied, only pixels in the *rotated map* that are also in this list
        are handled. Overrides `mask`.
    pol_axis : 3-vector, optional
        Axis normal to the plane in which the Q/U coordinates are defined.

    Returns
    -------
    mr : array_like
        The rotated map.
    """

    m = np.atleast_2d(m)

    if m.ndim > 2:
        raise ValueError(
            'Input map array must have no more than two dimensions')
    if m.shape[0] not in [1,2,3]:
        raise ValueError(
            'Input map must have 1 (T only), 2 (Q/U only) or 3 (T/Q/U) columns')

    res = hp.get_nside(m)
    pol = m.shape[0] in [2,3]
    pol_only = m.shape[0] == 2

    if rot is None:
        #use default coord transform C->G
        R = hp.Rotator(coord=coord, inv=True)
    else:
        R = hp.Rotator(rot=rot, inv=True)

    try:
        # qpoint is a 50% speedup
        import qpoint as qp
        Q = qp.QPoint()
        use_qpoint = True
    except ImportError:
        use_qpoint = False

    if pixels is None:
        if mask is not None:
            pixels, = np.where(mask)
        else:
            pixels = np.arange(len(m[0]))

    # rotate new coordinate system to original coordinates
    theta, phi = hp.pix2ang(res, pixels)
    mtheta, mphi = R(theta, phi)
    del theta
    del phi

    mr = np.full_like(m, np.nan)

    if use_qpoint:
        ra = np.degrees(mphi)
        dec = 90. - np.degrees(mtheta)
        del mtheta
        del mphi
        if not pol_only:
            mr[0, pixels] = Q.get_interp_val(m[0], ra, dec)
    else:
        if not pol_only:
            mr[0, pixels] = hp.get_interp_val(m[0], mtheta, mphi)

    if not pol:
        return mr.squeeze()

    #interpolate Q and U (better before or after rot?)
    if use_qpoint:
        mr[-2, pixels] = Q.get_interp_val(m[-2], ra, dec)
        mr[-1, pixels] = Q.get_interp_val(m[-1], ra, dec)
    else:
        mr[-2, pixels] = hp.get_interp_val(m[-2], mtheta, mphi)
        mr[-1, pixels] = hp.get_interp_val(m[-1], mtheta, mphi)

    rotate_qu_radial(
        mr, coord=coord, rot=rot, pol_axis=pol_axis, inplace=True)

    return mr

def write_map(filename, m, nest=False, dtype=np.float64, fits_IDL=True,
    coord=None, partial=False, mask=None, pixels=None, nside=None,
    column_names=None, column_units=None, extra_header=(),
    append=False, return_hdu=False):
    """Writes an healpix map into an healpix file.

    Parameters
    ----------
    filename : str
      the fits file name
    m : array or sequence of 3 arrays
      the map to write. Possibly a sequence of 3 maps of same size.
      They will be considered as I, Q, U maps.
      Supports masked maps, see the `ma` function.
    nest : bool, optional
      If True, ordering scheme is assumed to be NESTED, otherwise, RING. Default: RING.
      The map ordering is not modified by this function, the input map array
      should already be in the desired ordering (run `ud_grade` beforehand).
    fits_IDL : bool, optional
      If True, reshapes columns in rows of 1024, otherwise all the data will
      go in one column. Default: True
    coord : str
      The coordinate system, typically 'E' for Ecliptic, 'G' for Galactic or 'C' for
      Celestial (equatorial)
    partial : bool, optional
      If True, fits file is written as a partial-sky file with explicit indexing.
      Otherwise, implicit indexing is used.  Default: False.
    mask : bool array, optional
      If supplied, mask (1=good, 0=bad) is applied to the input map, and the result
      is stored as a partial map.  Overrides `partial` option.
    pixels : index array, optional
      If supplied, the input map is assumed to be a partial map containing only
      these pixels.  Overrides `mask` and `partial` options.
    nside : int, optional
      If `pixels` is supplied, this argument is required to verify the map shape.
    column_names : str or list
      Column name or list of column names, if None we use:
      I_STOKES for 1 component,
      I/Q/U_STOKES for 3 components,
      II, IQ, IU, QQ, QU, UU for 6 components,
      COLUMN_0, COLUMN_1... otherwise
    column_units : str or list
      Units for each column, or same units for all columns.
    extra_header : list or dict
      Extra records to add to FITS header.
    dtype : np.dtype or list of np.dtypes, optional
      The datatype in which the columns will be stored. Will be converted
      internally from the numpy datatype to the fits convention. If a list,
      the length must correspond to the number of map arrays.
    append : bool
      Set this option to append the map to an existing file as a new HDU.
    return_hdu : bool
      Set this option to return the BinTableHDU that would be written, rather
      that writing it to disk.
    """
    from healpy.fitsfunc import getformat, standard_column_names, pf
    from healpy import pixelfunc
    """
    standard_column_names.update(**{
            4: ['{}_STOKES'.format(comp) for comp in 'IQUV'],
            10: ['II', 'IQ', 'IU', 'IV', 'QQ', 'QU', 'QV', 'UU', 'UV', 'VV']
            })
    """
    kwargs = {
            4: ['{}_STOKES'.format(comp) for comp in 'IQUV'],
            10: ['II', 'IQ', 'IU', 'IV', 'QQ', 'QU', 'QV', 'UU', 'UV', 'VV']
            }
    standard_column_names.update(**{str(k): v for k, v in kwargs.items()})

    if not hasattr(m, '__len__'):
        raise TypeError('The map must be a sequence')

    m = pixelfunc.ma_to_array(m)
    if pixels is None:
        if pixelfunc.maptype(m) == 0: # a single map is converted to a list
            m = [m]
    else:
        m = np.atleast_2d(m)

    # check the dtype and convert it
    try:
        fitsformat = []
        for curr_dtype in dtype:
            fitsformat.append(getformat(curr_dtype))
    except TypeError:
        #dtype is not iterable
        fitsformat = [getformat(dtype)] * len(m)

    if column_names is None:
        column_names = standard_column_names.get(
            len(m), ["COLUMN_%d" % n for n in range(len(m))])
    else:
        assert len(column_names) == len(m), \
            "Length column_names != number of maps"

    if column_units is None or isinstance(column_units, str):
        column_units = [column_units] * len(m)

    # maps must have same length
    assert len(set(map(len, m))) == 1, "Maps must have same length"
    if pixels is None:
        nside = pixelfunc.npix2nside(len(m[0]))
    elif nside is None:
        raise ValueError('Invalid healpix map : nside required')

    if nside < 0:
        raise ValueError('Invalid healpix map : wrong number of pixel')

    cols=[]

    if mask is not None or pixels is not None:
        partial = True

    if partial:
        fits_IDL = False
        if pixels is not None:
            pix = pixels
            if any([mm.shape != pix.shape for mm in m]):
                raise ValueError('Invalid healpix map : pixel index mismatch')
        else:
            if mask is None:
                mask = mask_good(m[0])
            m = [mm[mask] for mm in m]
            pix = np.where(mask)[0]
        if len(pix) == 0:
            raise ValueError('Invalid healpix map : empty partial map')
        ff = getformat(np.min_scalar_type(-pix.max()))
        if ff is None:
            ff = 'I'
        cols.append(pf.Column(name='PIXEL',
                              format=ff,
                              array=pix,
                              unit=None))

    for cn, cu, mm, curr_fitsformat in zip(column_names, column_units, m,
                                           fitsformat):
        if len(mm) > 1024 and fits_IDL:
            # I need an ndarray, for reshape:
            mm2 = np.asarray(mm)
            cols.append(pf.Column(name=cn,
                                  format='1024%s' % curr_fitsformat,
                                  array=mm2.reshape(int(mm2.size/1024),1024),
                                  unit=cu))
        else:
            cols.append(pf.Column(name=cn,
                                  format='%s' % curr_fitsformat,
                                  array=mm,
                                  unit=cu))

    tbhdu = pf.BinTableHDU.from_columns(cols)
    # add needed keywords
    tbhdu.header.set('PIXTYPE', 'HEALPIX', 'HEALPIX pixelisation')
    tbhdu.header.set('ORDERING', 'NESTED' if nest else 'RING',
                     'Pixel ordering scheme, either RING or NESTED')
    if coord:
        tbhdu.header.set('COORDSYS', coord,
                         'Ecliptic, Galactic or Celestial (equatorial)')
    tbhdu.header.set('EXTNAME', 'xtension',
                     'name of this binary table extension')
    tbhdu.header.set('NSIDE', nside, 'Resolution parameter of HEALPIX')
    if not partial:
        tbhdu.header.set('FIRSTPIX', 0, 'First pixel # (0 based)')
        tbhdu.header.set('LASTPIX', pixelfunc.nside2npix(nside) - 1,
                         'Last pixel # (0 based)')
    tbhdu.header.set('INDXSCHM', 'EXPLICIT' if partial else 'IMPLICIT',
                     'Indexing: IMPLICIT or EXPLICIT')
    tbhdu.header.set('OBJECT', 'PARTIAL' if partial else 'FULLSKY',
                     'Sky coverage, either FULLSKY or PARTIAL')

    if not isinstance(extra_header, dict):
        for args in extra_header:
            if args[0] == 'COMMENT':
                tbhdu.header.add_comment(*args[1:])
            else:
                tbhdu.header.set(*args)
    else:
        tbhdu.header.update(extra_header)

    if return_hdu:
        return tbhdu

    if not append:
        from astropy import __version__ as astropy_version
        if astropy_version >= "1.3":
            tbhdu.writeto(filename, overwrite=True)
        else:
            tbhdu.writeto(filename, clobber=True)
    else:
        if isinstance(filename, str):
            if not os.path.exists(filename):
                # doesn't exist yet. write normally, with dummy Primary HDU
                tbhdu.writeto(filename)
        pf.append(filename, tbhdu.data, tbhdu.header, verify=False)

def write_alm(filename,alms,out_dtype=None,lmax=-1,mmax=-1,mmax_in=-1):
    """Write alms to a fits file.

    In the fits file the alms are written
    with explicit index scheme, index = l*l + l + m +1, possibly out of order.
    By default write_alm makes a table with the same precision as the alms.
    If specified, the lmax and mmax parameters truncate the input data to
    include only alms for which l <= lmax and m <= mmax.

    Parameters
    ----------
    filename : str
      The filename of the output fits file
    alms : array, complex or list of arrays
      A complex ndarray holding the alms, index = m*(2*lmax+1-m)/2+l, see Alm.getidx
    lmax : int, optional
      The maximum l in the output file
    mmax : int, optional
      The maximum m in the output file
    out_dtype : data type, optional
      data type in the output file (must be a numpy dtype). Default: *alms*.real.dtype
    mmax_in : int, optional
      maximum m in the input array
    """

    from healpy import Alm
    from healpy.fitsfunc import getformat, pf
    from healpy import cookbook as cb

    if not cb.is_seq_of_seq(alms):
        alms = [alms]

    l2max = Alm.getlmax(len(alms[0]),mmax=mmax_in)
    if (lmax != -1 and lmax > l2max):
        raise ValueError("Too big lmax in parameter")
    elif lmax == -1:
        lmax = l2max

    if mmax_in == -1:
        mmax_in = l2max

    if mmax == -1:
        mmax = lmax
    if mmax > mmax_in:
        mmax = mmax_in

    if (out_dtype == None):
        out_dtype = alms[0].real.dtype

    l,m = Alm.getlm(lmax)
    idx = np.where((l <= lmax)*(m <= mmax))
    l = l[idx]
    m = m[idx]

    idx_in_original = Alm.getidx(l2max, l=l, m=m)

    index = l**2 + l + m + 1

    hdulist = pf.HDUList()
    for alm in alms:
        out_data = np.empty(len(index),
                             dtype=[('index','i'),
                                    ('real',out_dtype),
                                    ('imag',out_dtype)])
        out_data['index'] = index
        out_data['real'] = alm.real[idx_in_original]
        out_data['imag'] = alm.imag[idx_in_original]

        cindex = pf.Column(name="index", format=getformat(np.int32),
                           unit="l*l+l+m+1", array=out_data['index'])
        creal = pf.Column(name="real", format=getformat(out_dtype),
                          unit="unknown", array=out_data['real'])
        cimag = pf.Column(name="imag", format=getformat(out_dtype),
                          unit="unknown", array=out_data['imag'])

        tbhdu = pf.BinTableHDU.from_columns([cindex,creal,cimag])
        hdulist.append(tbhdu)
    from astropy import __version__ as astropy_version
    if astropy_version >= "1.3":
        hdulist.writeto(filename, overwrite=True)
    else:
        hdulist.writeto(filename, clobber=True)

def read_map(filename, field=0, dtype=np.float64, nest=False, partial=False,
             fill=np.nan, hdu=1, h=False, verbose=False, memmap=False,
             return_part=False, return_pix=True, return_names=False):
    """Read an healpix map from a fits file.  Partial sky files are expanded
    to full size and filled with UNSEEN.

    Parameters
    ----------
    filename : str
      The fits file name.
      Can also be an HDUList object from astropy.io.fits.open or a
      particular HDU from the list.
    field : int or tuple of int, or None, optional
      The column to read. Default: 0.
      By convention 0 is temperature, 1 is Q, 2 is U.
      Field can be a tuple to read multiple columns (0,1,2)
      If the fits file is a partial-sky file, field=0 corresponds to the
      first column after the pixel index column.
      If None, all columns are read in.
    dtype : data type or list of data types, optional
      Force the conversion to some type. Passing a list allows different
      types for each field. In that case, the length of the list must
      correspond to the length of the field parameter. Default: np.float64
    nest : bool, optional
      If True return the map in NEST ordering, otherwise in RING ordering;
      use fits keyword ORDERING to decide whether conversion is needed or not
      If None, no conversion is performed.
    partial : bool, optional
      If True, fits file is assumed to be a partial-sky file with explicit indexing,
      if the indexing scheme cannot be determined from the header.
      If False, implicit indexing is assumed.  Default: False.
      A partial sky file is one in which OBJECT=PARTIAL and INDXSCHM=EXPLICIT,
      and the first column is then assumed to contain pixel indices.
      A full sky file is one in which OBJECT=FULLSKY and INDXSCHM=IMPLICIT.
      At least one of these keywords must be set for the indexing
      scheme to be properly identified.
    return_part : bool, optional
      If the map is a partial-sky file (see 'partial' above), don't fill
      out to full-sky. Return the map and the pixels array (if `return_pix` is True).
    return_pix : bool, optional
      If the map is a partial-sky file (see 'partial' above), and `return_part`
      is True, return the pixel array.
    return_names : bool, optional
      If True, return the names of fields that have been read.
    fill : scalar, optional
      Fill the bad pixels with this value, if supplied.  Default: NaN.
    hdu : int, optional
      the header number to look at (start at 0)
    h : bool, optional
      If True, return also the header. Default: False.
    verbose : bool, optional
      If True, print a number of diagnostic messages
    memmap : bool, optional
      Argument passed to astropy.io.fits.open, if True, the map is not read into memory,
      but only the required pixels are read when needed. Default: False.

    Returns
    -------
    m | (m0, m1, ...) [, header] : 1D or 2D array
      The map(s) read from the file
    pix : (If return_part is True and return_pix is True) 1D array
      List of pixels contained in partial-sky map. If return_part=True but
      the map is not partial-sky, this will be None
    nside : (If return_part is True) int
      healpix nside of the map. Needed for partial-sky
    header : (if h is True) The FITS header
    """
    import warnings
    from healpy.fitsfunc import pf, HealpixFitsWarning
    from healpy import pixelfunc, UNSEEN

    if isinstance(filename, str):
        hdulist = pf.open(filename, memmap=memmap)
        fits_hdu = hdulist[hdu]
    elif isinstance(filename, pf.HDUList):
        fits_hdu = filename[hdu]
    else:
        # assume it's an HDU directly
        fits_hdu = filename

    if not isinstance(fits_hdu, pf.BinTableHDU):
        raise TypeError("FITS error: Healpix map must be a binary table")

    # check nside
    nside = fits_hdu.header.get('NSIDE')
    if nside is None:
        warnings.warn(
            "No NSIDE in the header file : will use length of array",
            HealpixFitsWarning)
    else:
        nside = int(nside)
    if verbose:
        print('NSIDE = {0:d}'.format(nside))
    if not pixelfunc.isnsideok(nside):
        raise ValueError('Wrong nside parameter.')
    sz = pixelfunc.nside2npix(nside)

    # check ordering
    ordering = fits_hdu.header.get('ORDERING', 'UNDEF').strip()
    if ordering == 'UNDEF':
        ordering = (nest and 'NESTED' or 'RING')
        warnings.warn("No ORDERING keyword in header file : "
                      "assume {}".format(ordering))
    if verbose:
        print('ORDERING = {0:s} in fits file'.format(ordering))

    # partial sky: check OBJECT, then INDXSCHM
    obj = fits_hdu.header.get('OBJECT', 'UNDEF').strip()
    if obj != 'UNDEF':
        if obj == 'PARTIAL':
            partial = True
        elif obj == 'FULLSKY':
            partial = False

    schm = fits_hdu.header.get('INDXSCHM', 'UNDEF').strip()
    if schm != 'UNDEF':
        if schm == 'EXPLICIT':
            if obj == 'FULLSKY':
                raise ValueError('Incompatible INDXSCHM keyword')
            partial = True
        elif schm == 'IMPLICIT':
            if obj == 'PARTIAL':
                raise ValueError('Incompatible INDXSCHM keyword')
            partial = False

    if schm == 'UNDEF':
        schm = 'EXPLICIT' if partial else 'IMPLICIT'
        #warnings.warn("No INDXSCHM keyword in header file : "
                       #"assume {}".format(schm))
    if verbose:
        print('INDXSCHM = {0:s}'.format(schm))

    # check field
    if field is None:
        field = range(len(fits_hdu.data.columns) - 1*partial)
    if not (hasattr(field, '__len__') or isinstance(field, str)):
        field = (field,)
    ret = []

    if return_names:
        names = fits_hdu.data.names

    if not return_part:
        return_pix = False

    if partial:
        # increment field counters
        field = tuple(f if isinstance(f, str) else f+1
                      for f in field)
        if return_pix or not return_part:
            try:
                pix = fits_hdu.data.field(0).astype(int).ravel()
            except pf.VerifyError as e:
                print(e)
                print("Trying to fix a badly formatted header")
                fits_hdu.verify("fix")
                pix = fits_hdu.data.field(0).astype(int).ravel()
        else:
            pix = None
    else:
        pix = None

    if return_names:
        rnames = [f if isinstance(f, str) else names[f] for f in field]
    else:
        rnames = None

    try:
        assert len(dtype) == len(field), \
            "The number of dtypes are not equal to the number of fields"
    except TypeError:
        dtype = [dtype] * len(field)

    for ff, curr_dtype in zip(field, dtype):
        try:
            m = fits_hdu.data.field(ff).astype(curr_dtype).ravel()
        except pf.VerifyError as e:
            print(e)
            print("Trying to fix a badly formatted header")
            fits_hdu.verify("fix")
            m = fits_hdu.data.field(ff).astype(curr_dtype).ravel()

        if partial and not return_part:
            mnew = fill * np.ones(sz, dtype=curr_dtype)
            mnew[pix] = m
            m = mnew

        if (not pixelfunc.isnpixok(m.size) or \
            (sz>0 and sz != m.size)) and verbose:
            print('nside={0:d}, sz={1:d}, m.size={2:d}'.format(nside, sz, m.size))
            raise ValueError('Wrong nside parameter.')

        if not nest is None: # no conversion with None
            if nest and ordering == 'RING':
                idx = pixelfunc.nest2ring(
                    nside, np.arange(m.size, dtype=np.int32))
                m = m[idx]
                if verbose:
                    print('Ordering converted to NEST')
            elif (not nest) and ordering == 'NESTED':
                idx = pixelfunc.ring2nest(
                    nside, np.arange(m.size, dtype=np.int32))
                m = m[idx]
                if verbose:
                    print('Ordering converted to RING')

        try:
            m[mask_bad(m)] = fill
        except OverflowError:
            pass
        ret.append(m)

    # convert list of map arrays to 1D or 2D
    if len(ret) == 1:
        ret = ret[0]
    else:
        ret = np.asarray(ret)
    # append pixel array or FITS header as requested
    ret = ((ret,) + ((pix,) * return_pix + (nside,)) * return_part +
           (fits_hdu.header.items(),) * h + (rnames,) * return_names)
    if len(ret) == 1:
        return ret[0]
    return ret

def make_hitsmask(projs, thresholds=[.6, .9, 15000, 25000], trim=True, trim_thresh=.8,
                  apodize=True, apod_fwhm=np.radians(.5)):
    """
    Generates a mask based on the given projs and thresholds.

    Arguments
    =========
    projs : array of strings or numpy arrays
        The proj[0] arrays on which the cut will be performed. Either an array of
        filenames and paths or an array of proj arrays.
    thresholds : array of floats (same length as projs array)
        The threshold cuts for the respective proj maps, one for each.
        Suggested thresholds (those used to make the default hitsmask, with the
           dccleang06 product):
           [90_net_weighted, 90_unweighted] = [.6, 15000]
           [150_net_weighted, 150_unweighted] = [.9, 25000]
    trim : bool, optional
        Trim the edges of the map so they're smoothed.
    trim_thresh: float between [0,1], optional
        Trimming is done by smoothing the boolean mask with a 5 degree
        beam, then only saving the values above the trim_thresh, so
        reducing this value trims less of the mask.
    apodize : bool, optional
        Apodize the mask edges of the map.
    apod_fwhm : float
        Fwhm in radians for apodization. Default, 0.0087266... radians (.5 degrees)
        based on past analysis to have spice and anafast play well together at
        high ell (ell > 500).

    Returns
    =======
    Array/healpix map with values between 0 and 1.
    """

    if len(thresholds) != len(projs):
        raise ValueError('Array of thresholds needs to be the same length as the number of projs.')
    for p,proj in enumerate(projs):
        if isinstance(proj, basestring):
            proj = read_map(proj)
        if p==0:
            themask = np.ones_like(proj)
        themask[np.isnan(proj)] = 0.
        themask[np.where(proj<thresholds[p])] = 0.

    if trim:
        themask = trim_map(themask, trim_thresh=trim_thresh)

    if apodize:
        themask = apodize_map(themask, fwhm=apod_fwhm)

    return themask

def trim_map(themap, trim_thresh=0.8, fwhm=np.radians(5.)):
    '''
    Convenience function created to trim the ragged edges of a map so
    they're smooth.

    Arguments
    =========
       themap : array
          The map to be apodized
       trim_thresh: float between [0,1], optional
          Trimming is done by smoothing the boolean mask with a 5 degree
          beam, then only saving the values above the trim_thresh, so
          reducing this value trims less of the mask.
       fwhm: radians, optional
          To change the fwhm for smoothing.

    Returns
    =======
    The trimmed map.
    '''

    trimmedmap = smoothing(themap.astype(float), fwhm=fwhm)
    themap = trimmedmap > trim_thresh

def smoothalm(alms, fwhm=0.0, sigma=None, beam=None, pol=True,
              mmax=None, verbose=True, inplace=True):
    """Smooth alm with a Gaussian symmetric beam or custom window function.

    Parameters
    ----------
    alms : array or sequence of 3 arrays
      Either an array representing one alm, or a sequence of arrays.
      See *pol* parameter.
    fwhm : float, optional
      The full width half max parameter of the Gaussian. Default:0.0
      [in radians]
    sigma : float, optional
      The sigma of the Gaussian. Override fwhm.
      [in radians]
    beam : array or sequence of 3 arrays, optional
      If supplied, the beam function is applied instead of a Gaussian
      beam to each alm.
    pol : bool, optional
      If True, assumes input alms are TEB. Output will be TQU maps.
      (input must be 1 or 3 alms)
      If False, apply spin 0 harmonic transform to each alm.
      (input can be any number of alms)
      If there is only one input alm, it has no effect. Default: True.
    mmax : None or int, optional
      The maximum m for alm. Default: mmax=lmax
    inplace : bool, optional
      If True, the alm's are modified inplace if they are contiguous arrays
      of type complex128. Otherwise, a copy of alm is made. Default: True.
    verbose : bool, optional
      If True prints diagnostic information. Default: True

    Returns
    -------
    alms : array or sequence of 3 arrays
      The smoothed alm. If alm[i] is a contiguous array of type complex128,
      and *inplace* is True the smoothing is applied inplace.
      Otherwise, a copy is made.
    """

    # make imports identical to healpy source for easy porting
    from healpy.sphtfunc import almxfl, Alm
    from healpy import cookbook as cb
    import numpy as np
    import six

    if beam is None:
        if sigma is None:
            sigma = fwhm / (2.*np.sqrt(2.*np.log(2.)))

        if verbose:
            print("Sigma is {0:f} arcmin ({1:f} rad) ".format(sigma*60*180/np.pi,sigma))
            print("-> fwhm is {0:f} arcmin".format(sigma*60*180/np.pi*(2.*np.sqrt(2.*np.log(2.)))))

    # Check alms
    if not cb.is_seq(alms):
        raise ValueError("alm must be a sequence")

    if sigma == 0 and beam is None:
        # nothing to be done
        return alms

    lonely = False
    if not cb.is_seq_of_seq(alms):
        alms = [alms]
        lonely = True

    # check beam
    if beam is not None:
        if not cb.is_seq(beam):
            raise ValueError("beam must be a sequence")
        if not lonely:
            if not cb.is_seq_of_seq(beam):
                beam = [beam]*len(alms)
            else:
                if len(beam) != len(alms):
                    raise ValueError("alm and beam shape mismatch")
        else:
            if cb.is_seq_of_seq(beam):
                raise ValueError("alm and beam shape mismatch")
            else:
                beam = [beam]

    # we have 3 alms -> apply smoothing to each map.
    # polarization has different B_l from temperature
    # exp{-[ell(ell+1) - s**2] * sigma**2/2}
    # with s the spin of spherical harmonics
    # s = 2 for pol, s=0 for temperature
    retalm = []
    for ialm, alm in enumerate(alms):
        lmax = Alm.getlmax(len(alm), mmax)
        if lmax < 0:
            raise TypeError('Wrong alm size for the given '
                            'mmax (len(alms[%d]) = %d).'%(ialm, len(alm)))
        if beam is None:
            ell = np.arange(lmax + 1.)
            s = 2 if ialm >= 1 and pol else 0
            fact = np.exp(-0.5 * (ell * (ell + 1) - s ** 2) * sigma ** 2)
        else:
            fact = beam[ialm]
        res = almxfl(alm, fact, mmax = mmax, inplace = inplace)
        retalm.append(res)
    # Test what to return (inplace/not inplace...)
    # Case 1: 1d input, return 1d output
    if lonely:
        return retalm[0]
    # case 2: 2d input, check if in-place smoothing for all alm's
    for i in six.moves.xrange(len(alms)):
        samearray = alms[i] is retalm[i]
        if not samearray:
            # Case 2a:
            # at least one of the alm could not be smoothed in place:
            # return the list of alm
            return retalm
    # Case 2b:
    # all smoothing have been performed in place:
    # return the input alms
    return alms

def smoothing(map_in, fwhm=0.0, sigma=None, beam=None, pol=True,
              iter=3, lmax=None, mmax=None, use_weights=False,
              fill=np.nan, datapath=None, verbose=True):
    """Smooth a map with a Gaussian symmetric beam or custom window function.

    No removal of monopole or dipole is performed.

    Parameters
    ----------
    map_in : array or sequence of 3 arrays
      Either an array representing one map, or a sequence of
      3 arrays representing 3 maps, accepts masked arrays
    fwhm : float, optional
      The full width half max parameter of the Gaussian [in radians].
      Default:0.0
    sigma : float, optional
      The sigma of the Gaussian [in radians]. Override fwhm.
    beam : array or sequence of 3 arrays, optional
      If supplied, the beam function is applied instead of a Gaussian
      beam to each alm.
    pol : bool, optional
      If True, assumes input maps are TQU. Output will be TQU maps.
      (input must be 1 or 3 alms)
      If False, each map is assumed to be a spin 0 map and is
      treated independently (input can be any number of alms).
      If there is only one input map, it has no effect. Default: True.
    iter : int, scalar, optional
      Number of iteration (default: 3)
    lmax : int, scalar, optional
      Maximum l of the power spectrum. Default: 3*nside-1
    mmax : int, scalar, optional
      Maximum m of the alm. Default: lmax
    use_weights: bool, scalar, optional
      If True, use the ring weighting. Default: False.
    fill : scalar, optional
      Fill the bad pixels with this value, if supplied.  Default: NaN.
    datapath : None or str, optional
      If given, the directory where to find the weights data.
    verbose : bool, optional
      If True prints diagnostic information. Default: True

    Returns
    -------
    maps : array or list of 3 arrays
      The smoothed map(s)
    """

    # make imports identical to healpy source for easy porting
    from healpy import pixelfunc
    from healpy.sphtfunc import map2alm, alm2map
    from healpy import cookbook as cb
    import numpy as np

    if not cb.is_seq(map_in):
        raise TypeError("map_in must be a sequence")

    # save the masks of inputs
    masks = mask_bad(map_in, badnan=True, badinf=True)
    if np.any(masks):
        map_in = np.array(map_in, copy=True)
        map_in[masks] = hp.UNSEEN

    if cb.is_seq_of_seq(map_in):
        nside = pixelfunc.npix2nside(len(map_in[0]))
        n_maps = len(map_in)
    else:
        nside = pixelfunc.npix2nside(len(map_in))
        n_maps = 0

    if pol or n_maps in (0, 1):
        # Treat the maps together (1 or 3 maps)
        alms = map2alm(map_in, lmax = lmax, mmax = mmax, iter = iter,
                       pol = pol, use_weights = use_weights,
                       datapath = datapath)
        smoothalm(alms, fwhm = fwhm, sigma = sigma, beam = beam,
                  inplace = True, verbose = verbose)
        output_map = alm2map(alms, nside, pixwin = False, verbose=verbose)
    else:
        # Treat each map independently (any number)
        output_map = []
        for m in map_in:
            alm = map2alm(m, lmax = lmax, mmax = mmax, iter = iter, pol = pol,
                          use_weights = use_weights, datapath = datapath)
            smoothalm(alm, fwhm = fwhm, sigma = sigma, beam = beam,
                      inplace = True, verbose = verbose)
            output_map.append(alm2map(alm, nside, pixwin = False, verbose=verbose))

    output_map = np.asarray(output_map)
    output_map[masks] = fill
    return output_map

def apodize_map(themap, fwhm=np.radians(.7), pol=False):
    '''
    Convenience function created to apodize a map.

    Arguments
    =========
    themap : array
      The map to be apodized
    fwhm : float, radians, optional
    pol : bool, optional
      If True, assumes input maps are TQU. Output will be TQU maps.

    Returns
    =======
    The apodized map.
    '''

    themap = smoothing(themap.astype(float), fwhm=fwhm, pol=pol)
    themap[themap < 0.] = 0.
    themap[themap > 1.] = 1.

    return themap

def bin_spectrum(cls, lmin=8, lmax=None, binwidth=25, return_error=False):
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

def circle_mask(nside=512, center=(-30.0, 49.0), radius=15.0, coord=None):
    """
    Create a mask that is True where  lat and lon lie in a given range.
    Default result is for Spider cmb region in equatorial coordinates.

    Arguments
    =========
    nside : int
        The nside at which to make the mask
    center : (lat, lon) 2-tuple of floats
        Coordinate to center the circular mask.
    radius : float
        radius of the circle.
    coord : 2-tuple of strings (or list, etc)
        Pair of (input_coord, output_coord) as per rotate_map.

    Returns
    =======
    Boolean array/healpix map that is True in selected region
    """
    ret = np.zeros(hp.nside2npix(nside), dtype=bool)
    vec = hp.ang2vec(center[1], center[0], lonlat=True)
    ret[hp.query_disc(nside, vec, np.radians(radius))] = True
    if coord is not None and len(coord) == 2:
        ret = rotate_map(ret, coord=coord).astype(bool)
    return ret

#az = np.linspace(-5, 5, 1000)
#el = np.linspace(-5, 5, 1000)
#azaz, elel = np.meshgrid(az, el)
#ga = np.exp(-0.5*(azaz**2 + elel**2) / 0.25)
#ga_cr = np.zeros_like(ga)
##ga_cr = ga
#stokes = e2iqu(ga, ga_cr, 10, 10, vpol=True)
