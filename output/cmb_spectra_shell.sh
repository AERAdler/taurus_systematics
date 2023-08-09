for HWP in 1 3 5
do  
    for BEAM in "gauss" "gaussside" "po" "poside"  
    do
        python spectra_analysis.py --sim_tag "cmb_BR${HWP}_fp2_${BEAM}_150avg" \
        --mask "gal070_eq_smo2.fits" \
        --ideal_map "maps_cmb_ideal_fp2_${BEAM}_150.fits"
    done
done