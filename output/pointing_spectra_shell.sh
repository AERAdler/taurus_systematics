for COMP in "azel" "polang"
do  
    for VAR in "randm" "fixed"  
    do
        python spectra_analysis.py --sim_tag "${VAR}_${COMP}_point_fp2_gauss" \
        --mask "gal070_eq_smo2.fits" --ideal_map "maps_cmb_ideal_fp2_gauss_150.fits"
    done
done