FOR FREQ in 130 135 140 145 150 155 160 165 170
do 
     mpirun -n 25 python balloon_mapper.py --freq $FREQ --ndets 25 --fwhm 60 --ndays 31 --sim_tag "taurus_niHWP_sim_${FREQ}" --create_fpu --fov 28. --hwp_mode stepped --hwp_model SPIDER_150 
