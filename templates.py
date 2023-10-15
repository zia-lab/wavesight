#!/usr/bin/env python3

bash_template_1 = '''#!/bin/bash

#nCladding       : {nCladding}
#nCore           : {nCore}
#coreRadius      : {coreRadius}
#free_wavelength : {wavelength} um
#nUpper          : {nUpper}
#numModes        : {numModes}
#MEEP_resolution : {MEEP_resolution}

cd {wavesight_dir}
echo "{nCladding},{nCore},{coreRadius},{wavelength},{nUpper},{numModes},{MEEP_resolution},{config_root}" >> sim_log.txt
# Check if the memory and time requirements have been already calculated
if [[ -f "{config_root}.req" ]]; then
echo "Requirements have already been estimated ..."
config_job_id=1
else
echo "Requirements have not been determined, submitting a job for this purpose ..."
# Submit the first array job
sbatch_output=$(sbatch <<EOL
#!/bin/bash
#SBATCH -n 1
#SBATCH --job-name=req_run_{config_root}
#SBATCH --mem=64GB
#SBATCH -t 2:00:00

#SBATCH -o {config_root}-req.out
#SBATCH -e {config_root}-req.out

cd {wavesight_dir}
~/anaconda/meep/bin/python /users/jlizaraz/CEM/wavesight/fiber_platform.py --config_dict "config_dict-{config_root}.pkl" --num_time_slices {num_time_slices} --mode_idx 0 --sim_time 0
EOL
)
# get the job id
config_job_id=$(echo "$sbatch_output" | awk '{{print $NF}}')
fi

if [ "$config_job_id" -eq 1 ]
then
#submit the axiliary job with no dependency
echo "Submitting the array job with no dependency ..."
sbatch <<EOL
#!/bin/bash
#SBATCH -n 1 
#SBATCH --job-name=buddy_job_{config_root}
#SBATCH --mem=1GB
#SBATCH -t 00:10:00
#SBATCH -o {config_root}-buddy.out
#SBATCH -e {config_root}-buddy.out

cd {wavesight_dir}
bash {config_root}-2.sh
EOL
else
#submit the axiliary job with dependency
echo "Submitting the array job with dependency on config job ..."
sbatch --dependency=afterany:$config_job_id <<EOL
#!/bin/bash
#SBATCH -n 1 
#SBATCH --job-name=buddy_job_{config_root}
#SBATCH --mem=1GB
#SBATCH -t 00:10:00
#SBATCH -o {config_root}-buddy.out
#SBATCH -e {config_root}-buddy.out

cd {wavesight_dir}
bash {config_root}-2.sh
EOL
fi
'''

bash_template_2 = '''#!/bin/bash

#nCladding       : {nCladding}
#nCore           : {nCore}
#coreRadius      : {coreRadius}
#free_wavelength : {wavelength} um
#nUpper          : {nUpper}
#numModes        : {numModes}
#MEEP_resolution : {MEEP_resolution}

cd {wavesight_dir}
# Check if the memory and time requirements have been already calculated
if [[ -f "{config_root}.req" ]]; then
echo "Reading resource requirements ..."
IFS=',' read -r memreq timereq diskreq simtime < {config_root}.req
else
echo "Requirements have not been determined."
exit 1
fi

echo "sbatch resources: ${{memreq}}GB,${{timereq}},${{diskreq}}MB"

# Submit the first array job
sbatch_output=$(sbatch <<EOL
#!/bin/bash
#SBATCH -n 1
#SBATCH --job-name=light_array_{config_root}
#SBATCH --mem=${{memreq}}GB
#SBATCH -t ${{timereq}}
#SBATCH --array=1-{num_modes}

#SBATCH -o {config_root}-%a.out
#SBATCH -e {config_root}-%a.out

cd {wavesight_dir}
~/anaconda/meep/bin/python /users/jlizaraz/CEM/wavesight/fiber_platform.py --config_dict "config_dict-{config_root}.pkl" --num_time_slices 0 --mode_idx \$SLURM_ARRAY_TASK_ID --sim_time ${{simtime}}
EOL
)

# get the job id
array_job_id=$(echo "$sbatch_output" | awk '{{print $NF}}')

#submit the analysis job
sbatch_output_plotter=$(sbatch --dependency=afterany:$array_job_id <<EOL
#!/bin/bash
#SBATCH -n 1 
#SBATCH --job-name=light_plot_{config_root}
#SBATCH --mem=${{memreq}}GB
#SBATCH -t ${{timereq}}
#SBATCH -o {config_root}-plotter.out
#SBATCH -e {config_root}-plotter.out

cd {wavesight_dir}
~/anaconda/meep/bin/python /users/jlizaraz/CEM/wavesight/fiber_plotter.py {config_root}
EOL
)
# get the job id
plotter_job_id=$(echo "$sbatch_output_plotter" | awk '{{print $NF}}')

# submit the guardian job
sbatch --dependency=afterany:$plotter_job_id <<EOL
#!/bin/bash
#SBATCH --job-name=calling_home_{config_root}
#SBATCH --output=calling_home.out
#SBATCH --error=calling_home.err
#SBATCH --time=00:01:00

cd {wavesight_dir}
~/anaconda/meep/bin/python /users/jlizaraz/CEM/wavesight/hail_david.py "Finished {config_root}."
EOL
'''
