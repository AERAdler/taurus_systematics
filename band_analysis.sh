for FREQ in {130..170..5}
do 
     mpirun -n 25 python balloon_mapper.py --run --freq $FREQ --npairs 25 --fwhm 60 \
     --days 31 --sim_tag "taurus_niHWP_sim_${FREQ}" --create_fpu --fov 28. \
     --hwp_mode stepped --hwp_model SPIDER_150 --balloon_track balloontrack1.txt
done 
