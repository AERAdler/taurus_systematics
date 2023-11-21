for HWP in 1 3 5
do  
    for BEAM in "33gauss" "wingfit"  
    do
        python spectra_analysis.py --sim_tag "cmb_BR${HWP}_${BEAM}_150avg" \
        --mask gal070_eq_smo2.fits --calibrate 1 --analyzis_dir cmb_sims \
        --ideal_map "maps_cmb_ideal_${BEAM}_150.fits" --fwhm 33
        python spectra_analysis.py --sim_tag "dust_BR${HWP}_${BEAM}_150avg" \
        --mask gal070_eq_smo2.fits --calibrate 1 --analyzis_dir dust_sims \
        --ideal_map "maps_dust_ideal_${BEAM}_150avg.fits" --fwhm 33
    done
    for BEAM in "po" "poside"
    do
        python spectra_analysis.py --sim_tag "cmb_BR${HWP}_fp2_${BEAM}_150avg" \
        --mask gal070_eq_smo2.fits --calibrate 1 --analyzis_dir cmb_sims \
        --ideal_map "maps_cmb_ideal_fp2_${BEAM}_150.fits" --fwhm 33
        python spectra_analysis.py --sim_tag "dust_BR${HWP}_fp2_${BEAM}_150avg" \
        --mask gal070_eq_smo2.fits --calibrate 1 --analyzis_dir dust_sims \
        --ideal_map "maps_dust_ideal_fp2_${BEAM}_150avg.fits" --fwhm 33

    done
done
