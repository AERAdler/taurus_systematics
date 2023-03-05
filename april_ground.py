#Project the april NSDIC maps to healpix. See brightness_temp
#jupyter notebook in thefloorisdata github repo

from netCDF4 import Dataset
import os 
import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
opj = os.path.join

folder="ground_maps"
satname = "NSIDC-0630-EASE2_S3.125km-F18_SSMIS-"

dates = range(85, 119, 1)
example = satname+"2015085-91H-E-SIR-CSU-v1.3.nc"
nside = 512

#De-projection
data_for_proj = Dataset(opj(folder, example), "r", format="NETCDF4")
X = np.array(data_for_proj["x"])
Y = np.array(data_for_proj["y"])
XX, YY = np.meshgrid(X, Y)
rho2 = XX**2+YY**2
e = 0.0818191908426
a = 6378137
qp = 1. - (1-e**2)*.5*np.log((1-e)/(1+e))/e
beta = -np.arcsin(1 -rho2/(qp*a**2))
lat =  .5*np.pi - (beta + e**2/3.*np.sin(2*beta))
lon = np.arctan2(XX,YY) - np.pi 
pixes = hp.ang2pix(nside, lat, lon)

#Iterate over month
for i in range(34):
    day = str(dates[i]).zfill(3)
    file_mor = satname+"2015{}-91H-M-SIR-CSU-v1.3.nc".format(day)
    data_mor = Dataset(opj(folder, file_mor), "r", format="NETCDF4")
    temp_mor = np.array(data_mor["TB"])
    
    map_mor = np.zeros(12*nside*nside)
    map_mor[pixes] = np.array(temp_mor[0])
    map_mor[map_mor==0] = hp.UNSEEN
    mask_mor = hp.mask_bad(map_mor)
    
    file_eve = satname+"2015{}-91H-E-SIR-CSU-v1.3.nc".format(day)
    data_eve = Dataset(opj(folder, file_eve), "r", format="NETCDF4")
    temp_eve = np.array(data_eve["TB"])

    map_eve = np.zeros(12*nside*nside)
    map_eve[pixes] = np.array(temp_eve[0])
    map_eve[map_eve==0] = hp.UNSEEN
    mask_eve = hp.mask_bad(map_eve)
    
    daymap = .5*(map_mor+map_eve)
    daymap[np.logical_xor(mask_mor, mask_eve)] = np.maximum(map_mor, map_eve)[np.logical_xor(mask_mor, mask_eve)]
    mapname = "SSMIS-2015{}-91H_South.fits".format(day)
    hp.write_map(opj(folder, mapname), daymap, dtype='float64', overwrite=True)
