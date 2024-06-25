import healpy as hp
import numpy as np
from scipy import interpolate

h = 6.62e-34
c = 3e8
k_b = 1.38e-23
T_cmb = 2.72548 

def tb2b(tb, nu):
    #Convert blackbody temperature to spectral
    x = h*nu/(k_b*tb)
    return 2*h*nu**3/c**2/(np.exp(x) - 1)

def dBdT(tb, nu):
    x = h*nu/(k_b*tb)
    slope = 2*k_b*nu**2/c**2*((x/2)/np.sinh(x/2))**2
    return slope

def rotate_to_point(inmap, lat, lon):
    """
    Rotates healpy such that the North Pole is at the point defined by
    (lat lon). Returns rotated map
    Arguments: 
    ----------
    inmap : (12*N*N,) array of floats
    Input map in the healpix RING format
    lat   : float
    Latitude of point on the map [-90;90]
    lon   : float
    Longitude of point on the map [-180;80]
    """

    nside = hp.npix2nside(inmap.shape[0])
    rlon = np.radians(lon)-np.pi
    rlat = np.pi/2.-np.radians(lat)
    x0, y0, z0 = hp.pix2vec(512, np.arange(hp.nside2npix(512)))
    x1 =  x0*np.cos(rlat)+z0*np.sin(rlat)
    z = -x0*np.sin(rlat)+z0*np.cos(rlat)
    x = x1*np.cos(rlon)-y0*np.sin(rlon)
    y = x1*np.sin(rlon)+y0*np.cos(rlon)
    pix_prime = hp.vec2pix(512, x,y,z)
    return inmap[pix_prime]

def telescope_view_angles(nside, h, surf_h=0., R = 6.371e6):
    """
    Calculates how the coordinates of a sphere of radius R with a given
    nside project on the view of an outside observer located at a distance
    h away from the north pole. Returns visible coordinates on the sphere 
    and corresponding coordinates for the observer.

    Arguments: 
    ----------
    nside  : int
    Healpix nside of the input map
    h      : float
    Altitude of observer above reference level in m

    Keyword arguments:
    ----------
    surf_h : float
    Altitude of ground at pole above reference level (default : 0.)
    R      : float
    Radius of sphere (default : Earth's Radius)
    """

    r_ground = R+surf_h
    h_abg = h - surf_h
    theta_fov = np.arccos((r_ground) / (r_ground + h_abg))
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    theta_visible = theta[theta<theta_fov]
    phi_visible = phi[theta<theta_fov]

    theta_from_tel = np.arctan2(r_ground*np.sin(theta_visible), 
                                r_ground*(1 - np.cos(theta_visible)) + h_abg)
    theta_from_tel = np.pi - theta_from_tel #Up is down
    phi_from_tel = np.pi - phi_visible #Therefore left is right
    return(theta_visible, phi_visible, theta_from_tel, phi_from_tel)

def ground_template(inmap, theta_visible, phi_visible, theta_from_tel,
                    phi_from_tel, nside_out=128, cmb=True, freq=95., 
                    frac_bwidth=.2, aposcale=1.):
    """
    Creates a ground template given a world map and sets of coordinates 
    (see telescope_view_angles) Returns a filled-out ground template.
    Arguments: 
    ----------
    inmap          : (12*N*N,) array of floats
    Input map in the healpix RING format
    theta_visible  : (M,) array of floats
    Colatitude of the visible points of inmap in radians
    phi_visible    : (M,) array of floats
    Longitude of the visible points of inmap in radians
    theta_from_tel : (M,) array of floats
    Correponding colatitude in other coordinate system
    phi_from_tel   : (M,) array of floats
    Correponding longtitude in other coordinate system

    Keyword arguments:
    ----------
    nside_out   : int
    Healpix nside of the output ground template (default : 128)
    cmb         : bool
    Convert the temperatures to CMB temperature units (default : True)
    freq        : float
    Frequency at which temperature is measured, in GHz (default : 95)
    frac_bwidth : float
    Bandwidth of the measurement, as a fraction of freq (default : 0.2)
    aposcale    : float
    Scale of apodization (degrees)
    """
    
    nside_world = hp.npix2nside(inmap.shape[0])
    ground_map = np.ones(hp.nside2npix(nside_out))*hp.UNSEEN
    ground_pix = hp.ang2pix(nside_out, theta_from_tel, phi_from_tel)
    ground_map[ground_pix] = inmap[hp.ang2pix(nside_world, theta_visible, phi_visible)]

    if cmb:
        freq_band=np.linspace(freq*(1-frac_bwidth/2.), freq*(1+frac_bwidth/2.), 
                              201)
        for i, tb in enumerate(ground_map):
            if tb!=hp.UNSEEN:
                bolo = np.trapz(tb2b(tb, freq_band), freq_band)
                corr = np.trapz(dBdT(T_cmb, freq_band), freq_band)
                ground_map[i] = bolo/corr*1e6

    pix_pow = int(np.log2(nside_out))
    map_low = hp.ud_grade(ground_map, 2)

    for i in range(pix_pow+1):
        nside = int(2**i)
        map_normal = hp.ud_grade(ground_map, nside)
        map_horizon = np.amin(hp.ang2pix(nside, theta_from_tel, phi_visible))
        map_normal[map_horizon:] = np.where(
            map_normal[map_horizon:]!=hp.UNSEEN, map_normal[map_horizon:], 
            hp.ud_grade(map_low, nside)[map_horizon:])
        map_low = map_normal
    """
    pix = np.arange(hp.nside2npix(nside_out))
    theta, phi = hp.pix2ang(nside_out,pix)
    #Find horizon and transition zone
    theta_horizon = np.amin(theta_from_tel)+np.pi/3600.
    apo_mid = 0.5*(theta_horizon+0.5*np.pi)
    apo_range = (theta_horizon-0.5*np.pi)
    #Sample along horizon
    phi_horizon = np.linspace(0, 2*np.pi, 3601)
    horpix = hp.ang2pix(nside_out, np.ones(3601)*theta_horizon, phi_horizon)
    tphi = map_normal[horpix]
    phi_horizon = phi_horizon[tphi!=hp.UNSEEN] #Exclude bad pixels
    tphi = tphi[tphi!=hp.UNSEEN]
    horizon_cs = interpolate.interp1d(phi_horizon, tphi)
    #Transition zone
    apod_pixels = pix[np.abs(theta-apo_mid)<0.5*apo_range]
    aposcale = np.radians(aposcale)
    map_normal[apod_pixels] = horizon_cs(phi[apod_pixels])
    #roughly 4e8 damping at horizon
    temp_scaling = np.exp(-.5 * ((theta[apod_pixels]-theta_horizon)/aposcale)**2 )
    map_normal[apod_pixels] *= temp_scaling
    """
    return map_normal

def template_from_position(earth_map, lat, lon, h, nside_out=128, 
                           cmb=True, freq=95., frac_bwidth=.2, aposcale=1.):
    """
    Creates a ground template given a world map, a position and an altitude.
    Returns a filled-out ground template.
    Arguments: 
    ----------
    earth_map          : (12*N*N,) array of floats
    Input map in K, in the healpix RING format
    lat   : float
    Latitude of point on the map [-90;90]
    lon   : float
    Longitude of point on the map [-180;80]
    h      : float
    Altitude of observer above reference level in m

    Keyword arguments:
    ----------
    nside_out   : int
    Healpix nside of the output ground template (default : 128)
    cmb         : bool
    Convert the temperatures to CMB temperature units (default : True)
    freq        : float
    Frequency at which temperature is measured, in GHz (default : 95)
    frac_bwidth : float
    Bandwidth of the measurement, as a fraction of freq (default : 0.2)
    """

    nside_world = hp.npix2nside(earth_map.shape[0])
    earth_rot = rotate_to_point(earth_map, lat, lon)
    theta_visible, phi_visible, theta_from_tel, phi_from_tel = telescope_view_angles(
        nside_world, h, surf_h=0, R=6.371e6)
    ground_temp = ground_template(earth_rot, theta_visible, phi_visible, 
                                  theta_from_tel, phi_from_tel, 
                                  nside_out=nside_out, cmb=cmb, freq=freq, 
                                  frac_bwidth=frac_bwidth, aposcale=aposcale)
    return ground_temp

