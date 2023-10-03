#!/bin/bash

#nCladding       : 1.4
#nCore           : 1.5
#coreRadius      : 1.5
#free_wavelength : 0.401 um
#nUpper          : 1.0
#numModes        : 84

cd /users/jlizaraz/CEM/wavesight
# Check if the memory and time requirements have been already calculated
if [[ -f "IHUM-IKIPreq" ]]; then
    echo "Reading resource requirements ..."
    IFS=',' read -r memreq timereq < IHUM-IKIP.req
else
    echo "Requirments have not been determined, running a single mode for this purpose."
    ~/anaconda/meep/bin/python /users/jlizaraz/CEM/wavesight/fiber_platform.py config_dict-IHUM-IKIP.pkl 30 0
    sleep 1
    IFS=',' read -r memreq timereq < IHUM-IKIP.req
fi

echo "sbatch resources: ${memreq}GB,${timereq}"

# Submit the first array job
sbatch_output=$(sbatch <<EOL
#!/bin/bash
#SBATCH -n 1
#SBATCH --job-name=light_array
#SBATCH --mem=${memreq}GB
#SBATCH -t ${timereq}
#SBATCH --array=1-83

#SBATCH -o IHUM-IKIP-%a.out
#SBATCH -e IHUM-IKIP-%a.out

cd /users/jlizaraz/CEM/wavesight
~/anaconda/meep/bin/python /users/jlizaraz/CEM/wavesight/fiber_platform.py config_dict-IHUM-IKIP.pkl 30 \$SLURM_ARRAY_TASK_ID
EOL
)
# get the job id
array_job_id=$(echo "$sbatch_output" | awk '{print $NF}')

#submit the analysis job
sbatch_output_plotter=$(sbatch --dependency=afterany:$array_job_id <<EOL
#!/bin/bash
#SBATCH -n 1 
#SBATCH --job-name=light_plot
#SBATCH --mem=${memreq}GB
#SBATCH -t ${timereq}
#SBATCH -o IHUM-IKIP-plotter.out
#SBATCH -e IHUM-IKIP-plotter.out

cd /users/jlizaraz/CEM/wavesight
~/anaconda/meep/bin/python /users/jlizaraz/CEM/wavesight/fiber_plotter.py IHUM-IKIP
EOL
)
# get the job id
plotter_job_id=$(echo "$sbatch_output_plotter" | awk '{print $NF}')

# submit the guardian job
sbatch --dependency=afterany:$plotter_job_id <<EOL
#!/bin/bash
#SBATCH --job-name=calling_home
#SBATCH --output=calling_home.out
#SBATCH --error=calling_home.err
#SBATCH --time=00:01:00

cd /users/jlizaraz/CEM/wavesight
~/anaconda/meep/bin/python /users/jlizaraz/CEM/wavesight/hail_david.py "Finished IHUM-IKIP."
EOL

