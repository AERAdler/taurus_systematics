#Print residuals
import numpy as np
import os
from scipy import stats
opj = os.path.join

def bin_spectrum(cls, lmin=2, lmax=None, binwidth=18, return_error=False):
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


spec_dir = opj("output", "dust_sims", "spectra")
spec_list = [ f for f in os.listdir(spec_dir) if f.endswith(".npy") ]
dspec_list = [f for f in os.listdir(spec_dir) if "diff" in f]
print("File name                                          2<l<7 7<l<12")
for file in sorted(dspec_list):
    spec = np.load(opj(spec_dir, file))
    ellb, clsb = bin_spectrum(spec, lmin=2, binwidth=5)
    dfacb = ellb*(ellb+1)/(2*np.pi)
    print("{}         {:.2e} {:.2e}".format(file, (dfacb*clsb)[1,0]*1e4, (dfacb*clsb)[1,1]*1e4))#Match unit of table


