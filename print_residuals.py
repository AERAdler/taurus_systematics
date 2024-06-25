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


ellb, cvEE = np.load("cv_EE_fullsky.npy")
cvEE = np.concatenate(([0,0], cvEE))
_, bincvEE_five = bin_spectrum(cvEE, lmin=2, binwidth=5)
print(bincvEE_five[1]/0.44/np.sqrt(5))
_, bincvEE_ten = bin_spectrum(cvEE, lmin=5, binwidth=10)

spec_dir = opj("output", "taurus_mission_maps", "spectra")
spec_list = [ f for f in os.listdir(spec_dir) if f.endswith(".npy") ]
dspec_list = [f for f in os.listdir(spec_dir) if "diff" in f]


beam_cols = ["gauss_", "wingfit_", "po_", "poside_"]
cal_cols = ["calno", "calTT", "calEE"]
hwp_cols = ["BR1", "BR3", "BR5"]
point_cols = ["azel", "polang"]
# Table: Dust Ghost beam 
#2a is 5<ell<15, 2b is 7<ell<12, 2c is 12<ell<17

table2_a = ["Beam model & No cal. & $TT$ cal & $EE$ cal", 
          "Gaussian", 
          "Fitted", 
          "PO", 
          "PO+Side"]
table2_b = table2_a.copy()
table2_c = table2_a.copy()

for cal in cal_cols:
    for file in sorted(dspec_list):
        spec = np.load(opj(spec_dir, file))
        _, clsbig = bin_spectrum(spec, lmin=5, binwidth=10)
        _, clssmall = bin_spectrum(spec, lmin=2, binwidth=5)
        frac_a = clsbig[1,0]/bincvEE_ten[0]*0.44*np.sqrt(10) #Fractional residual
        frac_b = clssmall[1,1]/bincvEE_five[1]*0.44*np.sqrt(5) 
        frac_c = clssmall[1,2]/bincvEE_five[2]*0.44*np.sqrt(5) 
        if all(element in file for element in ["pure", "ghost_1e-2", "dust", cal]):
            for idx, beam in enumerate(beam_cols):
                if beam in file:
                    table2_a[idx+1] += f" & {frac_a:.2f}"
                    table2_b[idx+1] += f" & {frac_b:.2f}"
                    table2_c[idx+1] += f" & {frac_c:.2f}"

print("Ghost")
print("5<ell<15")
print(("\\\\\hline\n").join(table2_a))
print("\n7<ell<12")
print(("\\\\\hline\n").join(table2_b))
print("\n12<ell<17")
print(("\\\\\hline\n").join(table2_c))

#Beam mismatch with Ideal HWP
table3_a = ["\\hline Sky, calibration & Fitted Beam & PO Beam & PO+Side", 
          "CMB, no cal.", 
          "Dust, no cal.", 
          "CMB, $TT$ cal.",
          "Dust, $TT$ cal.",
          "CMB, $TT$ cal.",
          "Dust, $EE$ cal."]
table3_b = table3_a.copy()
table3_c = table3_a.copy()

for beam in beam_cols[1:]:
    for file in sorted(dspec_list):
        spec = np.load(opj(spec_dir, file))
        _, clsbig = bin_spectrum(spec, lmin=5, binwidth=10)
        _, clssmall = bin_spectrum(spec, lmin=2, binwidth=5)
        frac_a = clsbig[1,0]/bincvEE_ten[0]*0.44*np.sqrt(10) 
        frac_b = clssmall[1,1]/bincvEE_five[1]*0.44*np.sqrt(5) 
        frac_c = clssmall[1,2]/bincvEE_five[2]*0.44*np.sqrt(5) 
        if all(element in file for element in ["pure","ideal_taumis", beam]):
            if any(element in file for element in [ "diffidealgauss", "vsgauss"]):
                for idx, cal in enumerate(cal_cols):
                    if cal in file:
                        peg = 2 if "dust" in file else 1
                        table3_a[2*idx+peg] += f" & {frac_a:.2f}"
                        table3_b[2*idx+peg] += f" & {frac_b:.2f}"
                        table3_c[2*idx+peg] += f" & {frac_c:.2f}"

print("\nBeam mismatch")
print("5<ell<15")
print(("\\\\\hline\n").join(table3_a))
print("\n7<ell<12")
print(("\\\\\hline\n").join(table3_b))
print("\n12<ell<17")
print(("\\\\\hline\n").join(table3_c))

#HWP non-ideality

table4_a = ["HWP model, Sky & No & $TT$ & $EE$ & No & $TT$ & $EE$ & No & $TT$ & $EE$ & No & $TT$ & $EE$",
          "BR1, CMB", 
          "BR1, Dust",
          "BR3, CMB",
          "BR3, Dust",
          "BR5, CMB", 
          "BR5, Dust"]
table4_b = table4_a.copy()
table4_c = table4_a.copy()

#HWP+beam non-ideality

table5_a = ["HWP model, Sky & No & $TT$ & $EE$ & No & $TT$ & $EE$ & No & $TT$ & $EE$",
          "BR1, CMB", 
          "BR1, Dust",
          "BR3, CMB",
          "BR3, Dust",
          "BR5, CMB", 
          "BR5, Dust"]
table5_b = table5_a.copy()
table5_c = table5_a.copy()

table6_a = table4_a.copy()
table6_b = table4_a.copy()
table6_c = table4_a.copy()

table7_a = ["Pointing error & No & $TT$ & $EE$ & No & $TT$ & $EE$ & No & $TT$ & $EE$ & No & $TT$ & $EE$",
          "Common \\SI{1}{\\arcmin} az-el offset", 
          "Random \\SI{1}{\\arcmin} az-el offset", 
          "Common \\SI{1}{\\degree} $\\xi$ offset", 
          "Random \\SI{1}{\\degree} $\\xi$ offset"]
table7_b = table7_a.copy()
table7_c = table7_a.copy()


for beam in beam_cols:
    for cal in cal_cols:
        for file in sorted(dspec_list):
            spec = np.load(opj(spec_dir, file))
            _, clsbig = bin_spectrum(spec, lmin=5, binwidth=10)
            _, clssmall = bin_spectrum(spec, lmin=2, binwidth=5)
            frac_a = clsbig[1,0]/bincvEE_ten[0]*0.44*np.sqrt(10) 
            frac_b = clssmall[1,1]/bincvEE_five[1]*0.44*np.sqrt(5) 
            frac_c = clssmall[1,2]/bincvEE_five[2]*0.44*np.sqrt(5) 
            if all(element in file for element in ["pure", cal, beam]):
                for idx, hwp in enumerate(hwp_cols):
                    if hwp in file:
                        peg = 2 if "dust" in file else 1
                        if "diffideal_" in file:
                            table4_a[2*idx+peg] += f" & {frac_a:.2f}"
                            table4_b[2*idx+peg] += f" & {frac_b:.2f}"
                            table4_c[2*idx+peg] += f" & {frac_c:.2f}"
                        if ((("diffidealgauss" in file) or ("vsgauss" in file)) and beam !="gauss_"):
                            table5_a[2*idx+peg] += f" & {frac_a:.2f}"
                            table5_b[2*idx+peg] += f" & {frac_b:.2f}"
                            table5_c[2*idx+peg] += f" & {frac_c:.2f}"

            if all(element in file for element in ["halfdeg", cal, beam]):
                for idx, hwp in enumerate(hwp_cols):
                    if hwp in file:
                        peg = 2 if "dust" in file else 1
                        table6_a[2*idx+peg] += f" & {frac_a:.2f}"
                        table6_b[2*idx+peg] += f" & {frac_b:.2f}"
                        table6_c[2*idx+peg] += f" & {frac_c:.2f}"

            if all(element in file for element in ["point", "pure", cal, beam]):
                block = 2 if "polang" in file else 0
                peg = 2 if "randm" in file else 1
                table7_a[block+peg] += f" & {frac_a:.2f}"
                table7_b[block+peg] += f" & {frac_b:.2f}"
                table7_c[block+peg] += f" & {frac_c:.2f}"



print("\nHWP only")
print("5<ell<15\n")
print(("\\\\\hline\n").join(table4_a))
print("\n7<ell<12")
print(("\\\\\hline\n").join(table4_b))
print("\n12<ell<17")
print(("\\\\\hline\n").join(table4_c))

print("\nHWP and beam")
print("5<ell<15")
print(("\\\\\hline\n").join(table5_a))
print("\n7<ell<12")
print(("\\\\\hline\n").join(table5_b))
print("\n12<ell<17")
print(("\\\\\hline\n").join(table5_c))

print("\nHWP angle error")
print("5<ell<15")
print(("\\\\\hline\n").join(table6_a))
print("\n7<ell<12")
print(("\\\\\hline\n").join(table6_b))
print("\n12<ell<17")
print(("\\\\\hline\n").join(table6_c))

print("\n Pointing error")
print("5<ell<15")
print(("\\\\\hline\n").join(table7_a))
print("\n7<ell<12")
print(("\\\\\hline\n").join(table7_b))
print("\n12<ell<17")
print(("\\\\\hline\n").join(table7_c))
