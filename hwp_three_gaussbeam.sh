#Run the balloon_mapper over the 150GHz band with a non-ideal AHWP
for FREQ in {130..170..5}  
do 
    mpirun -n 25 python balloon_mapper.py --run --freq $FREQ --npairs 25 \
    --fwhm 60 --days 31 --create_fpu --fov 28. --hwp_mode stepped \
    --hwp_model "band3" --balloon_track balloontrack1.txt \
    --sim_tag "BR3_gauss_${FREQ}" --hwp_phase 32.21
done
