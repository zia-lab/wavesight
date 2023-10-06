#!/usr/bin/env python3
# fiber_bundle.py

import pickle
import argparse
import warnings
import numpy as np
import wavesight as ws

warnings.filterwarnings('ignore', 'invalid value encountered in scalar add')
warnings.filterwarnings('ignore', 'invalid value encountered in add')
warnings.filterwarnings('ignore', 'invalid value encountered in scalar subtract')
warnings.filterwarnings('ignore', 'invalid value encountered in subtract')

wavesight_dir = '/users/jlizaraz/CEM/wavesight'
num_time_slices = 30 # approx how many time samples of fields

bash_template_1 = '''#!/bin/bash

#nCladding       : {nCladding}
#nCore           : {nCore}
#coreRadius      : {coreRadius}
#free_wavelength : {wavelength} um
#nUpper          : {nUpper}
#numModes        : {numModes}

cd {wavesight_dir}
# Check if the memory and time requirements have been already calculated
if [[ -f "{config_root}.req" ]]; then
echo "Requirements have already been estimated ..."
config_job_id=1
else
echo "Requirements have not been determined, running a single mode for this purpose."
# Submit the first array job
sbatch_output=$(sbatch <<EOL
#!/bin/bash
#SBATCH -n 1
#SBATCH --job-name=req_run
#SBATCH --mem=64GB
#SBATCH -t 2:00:00

#SBATCH -o {config_root}-req.out
#SBATCH -e {config_root}-req.out

cd {wavesight_dir}
~/anaconda/meep/bin/python /users/jlizaraz/CEM/wavesight/fiber_platform.py config_dict-{config_root}.pkl {num_time_slices} 0
EOL
)
# get the job id
config_job_id=$(echo "$sbatch_output" | awk '{{print $NF}}')
fi

if [ "$config_job_id" -eq 1 ]
then
#submit the axiliary job with no dependency
sbatch <<EOL
#!/bin/bash
#SBATCH -n 1 
#SBATCH --job-name=buddy_job
#SBATCH --mem=1GB
#SBATCH -t 00:10:00
#SBATCH -o {config_root}-buddy.out
#SBATCH -e {config_root}-buddy.out

cd {wavesight_dir}
bash {config_root}-2.sh
EOL
else
#submit the axiliary job with dependency
sbatch --dependency=afterany:$config_job_id <<EOL
#!/bin/bash
#SBATCH -n 1 
#SBATCH --job-name=buddy_job
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

cd {wavesight_dir}
# Check if the memory and time requirements have been already calculated
if [[ -f "{config_root}.req" ]]; then
echo "Reading resource requirements ..."
IFS=',' read -r memreq timereq diskreq < {config_root}.req
else
echo "Requirements have not been determined."
exit 1
fi

echo "sbatch resources: ${{memreq}}GB,${{timereq}},${{diskreq}}MB"

# Submit the first array job
sbatch_output=$(sbatch <<EOL
#!/bin/bash
#SBATCH -n 1
#SBATCH --job-name=light_array
#SBATCH --mem=${{memreq}}GB
#SBATCH -t ${{timereq}}
#SBATCH --array=1-{num_modes}

#SBATCH -o {config_root}-%a.out
#SBATCH -e {config_root}-%a.out

cd {wavesight_dir}
~/anaconda/meep/bin/python /users/jlizaraz/CEM/wavesight/fiber_platform.py config_dict-{config_root}.pkl {num_time_slices} \$SLURM_ARRAY_TASK_ID
EOL
)

# get the job id
array_job_id=$(echo "$sbatch_output" | awk '{{print $NF}}')

#submit the analysis job
sbatch_output_plotter=$(sbatch --dependency=afterany:$array_job_id <<EOL
#!/bin/bash
#SBATCH -n 1 
#SBATCH --job-name=light_plot
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
#SBATCH --job-name=calling_home
#SBATCH --output=calling_home.out
#SBATCH --error=calling_home.err
#SBATCH --time=00:01:00

cd {wavesight_dir}
~/anaconda/meep/bin/python /users/jlizaraz/CEM/wavesight/hail_david.py "Finished {config_root}."
EOL
'''

def approx_time(sim_cell, spatial_resolution, run_time, kappa=3.06e-6):
    rtime = (kappa * sim_cell.x * sim_cell.y * sim_cell.z
             * run_time * spatial_resolution**3)
    return rtime

def fan_out(nCladding, nCore, coreRadius, λFree, nUpper):
    fiber_spec = {'nCladding': nCladding,
                'nCore': nCore,
                'coreRadius': coreRadius,
                'grid_divider': 4,
                'nUpper': nUpper,
                'λFree': λFree}
    fiber_sol = ws.multisolver(fiber_spec,
                            solve_modes = 'all',
                            drawPlots=False,
                            verbose=True)
    numModes = fiber_sol['totalModes']
    fiber_sol = ws.calculate_numerical_basis(fiber_sol, verbose=False)
    a, b, Δs, xrange, yrange, ρrange, φrange, Xg, Yg, ρg, φg, nxy, crossMask, numSamples = fiber_sol['coord_layout']
    nUpper = fiber_sol['nUpper']
    λUpper = λFree / nUpper
    sample_resolution = 10
    MEEP_resolution  = 20
    slack_channel = 'nvs_and_metalenses'
    distance_to_monitor = 1.5 * λUpper
    fiber_alpha = np.arcsin(np.sqrt(nCore**2-nCladding**2))
    config_dict = {}
    config_dict['ρrange'] = ρrange
    config_dict['Xg'] = Xg
    config_dict['Yg'] = Yg
    config_dict['λUpper'] = λUpper
    config_dict['sample_resolution'] = sample_resolution
    config_dict['MEEP_resolution'] = MEEP_resolution
    config_dict['slack_channel']   = slack_channel
    config_dict['num_time_slices'] = num_time_slices
    config_dict['distance_to_monitor'] = distance_to_monitor
    config_dict['fiber_alpha'] = fiber_alpha
    config_dict['eigennums']   = fiber_sol['eigenbasis_nums']
    config_dict['fiber_sol'] = fiber_sol
    config_dict['nUpper'] = nUpper
    config_dict['numModes'] = numModes
    print("There are %d modes to solve." % numModes)
    batch_rid = ws.rando_id()
    config_fname = 'config_dict-'+batch_rid+'.pkl'
    with open(config_fname,'wb') as file:
        print("Saving configuration parameters to %s" % config_fname)
        pickle.dump(config_dict, file)
    bash_script_fname_1 = batch_rid+'-1.sh'
    bash_script_fname_2 = batch_rid+'-2.sh'
    batch_script_1 = bash_template_1.format(wavesight_dir=wavesight_dir,
                    config_fname = config_fname,
                    config_root  = batch_rid,
                    coreRadius   = coreRadius,
                    nCladding    = nCladding,
                    nCore        = nCore,
                    nUpper       = nUpper,
                    wavelength   = λFree,
                    numModes     = numModes,
                    num_time_slices = num_time_slices,
                    num_modes=(numModes-1))
    batch_script_2 = bash_template_2.format(wavesight_dir=wavesight_dir,
                    config_fname = config_fname,
                    config_root  = batch_rid,
                    coreRadius   = coreRadius,
                    nCladding    = nCladding,
                    nCore        = nCore,
                    nUpper       = nUpper,
                    wavelength   = λFree,
                    numModes     = numModes,
                    num_time_slices = num_time_slices,
                    num_modes=(numModes-1))
    with open(bash_script_fname_1, 'w') as file:
        print("Saving bash script to %s" % bash_script_fname_1)
        file.write(batch_script_1+'\n')
    with open(bash_script_fname_2, 'w') as file:
        print("Saving bash script to %s" % bash_script_fname_2)
        file.write(batch_script_2+'\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A simple CLI that accepts four parameters.')
    parser.add_argument('nCladding', type=float, help='The refractive index of the cladding.')
    parser.add_argument('nCore', type=float, help='The refractive index of the core.')
    parser.add_argument('coreRadius', type=float, help='The radius of the core.')
    parser.add_argument('free_space_wavelength', type=float, help='The free space wavelength.')
    parser.add_argument('nUpper', type=float, help='The refrective index of the upper medium.')
    args = parser.parse_args()
    fan_out(args.nCladding, args.nCore, args.coreRadius, args.free_space_wavelength, args.nUpper)
