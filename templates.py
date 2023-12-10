#!/usr/bin/env python3

bash_template_1 = '''#!/bin/bash

#nCladding       : {nCladding}
#nCore           : {nCore}
#coreRadius      : {coreRadius}
#free_wavelength : {wavelength} um
#nBetween        : {nBetween}
#numModes        : {numModes}
#MEEP_resolution : {MEEP_resolution}

cd {wavesight_dir}
echo "{nCladding},{nCore},{coreRadius},{wavelength},{nBetween},{numModes},{MEEP_resolution},{waveguide_id}" >> sim_log.txt
# Check if the memory and time requirements have been already calculated
if [[ -f "{waveguide_dir}/{waveguide_id}.req" ]]; then
echo "Requirements have already been estimated ..."
config_job_id=1
else
echo "Requirements have not been determined, submitting a job for this purpose ..."
# Submit the first array job
sbatch_output=$(sbatch <<EOL
#!/bin/bash
#SBATCH -n {MEEP_num_cores}
#SBATCH --job-name=req_run_{waveguide_id}
#SBATCH --mem={req_run_mem_in_GB}GB
#SBATCH -t {req_run_time_in_hours}:00:00

#SBATCH -o "{waveguide_dir}/{waveguide_id}-req.out"
#SBATCH -e "{waveguide_dir}/{waveguide_id}-req.err"

cd {wavesight_dir}
{python_bin_MEEP} {code_dir}fiber_platform.py --waveguide_sol "waveguide_sol-{waveguide_id}.pkl" --num_time_slices {num_time_slices} --mode_idx 0 --sim_time 0
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
#SBATCH --job-name=buddy_job_{waveguide_id}
#SBATCH --mem=1GB
#SBATCH -t 00:10:00
#SBATCH -o "{waveguide_dir}/{waveguide_id}-buddy.out"
#SBATCH -e "{waveguide_dir}/{waveguide_id}-buddy.err"

cd "{waveguide_dir}"
bash {waveguide_id}-2.sh
EOL
else
#submit the axiliary job with dependency
echo "Submitting the array job with dependency on config job ..."
sbatch --dependency=afterany:$config_job_id <<EOL
#!/bin/bash
#SBATCH -n 1 
#SBATCH --job-name=buddy_job_{waveguide_id}
#SBATCH --mem=1GB
#SBATCH -t 00:10:00
#SBATCH -o "{waveguide_dir}/{waveguide_id}-buddy.out"
#SBATCH -e "{waveguide_dir}/{waveguide_id}-buddy.err"

cd {wavesight_dir}
bash {waveguide_id}-2.sh
EOL
fi
'''

bash_template_2 = '''#!/bin/bash

#nCladding       : {nCladding}
#nCore           : {nCore}
#coreRadius      : {coreRadius}
#free_wavelength : {wavelength} um
#nBetween        : {nBetween}
#numModes        : {numModes}
#MEEP_resolution : {MEEP_resolution}

cd {wavesight_dir}
# Check if the memory and time requirements have been already calculated
if [[ -f "{waveguide_dir}/{waveguide_id}.req" ]]; then
echo "Reading resource requirements ..."
IFS=',' read -r memreq timereq diskreq simtime memreq2 timereq2 < "{waveguide_dir}/{waveguide_id}.req"
else
echo "Requirements have not been determined."
exit 1
fi

echo "sbatch resources: ${{memreq}}GB,${{timereq}},${{diskreq}}MB,${{memreq2}}GB,${{timereq2}}"

# Submit the first array job
sbatch_output=$(sbatch <<EOL
#!/bin/bash
#SBATCH -n {MEEP_num_cores}
#SBATCH --job-name=fiber_platform_{waveguide_id}
#SBATCH --mem=${{memreq}}GB
#SBATCH -t ${{timereq}}
#SBATCH --array=1-{num_modes}

#SBATCH -o "{waveguide_dir}/{waveguide_id}-%a.out"
#SBATCH -e "{waveguide_dir}/{waveguide_id}-%a.err"

cd {wavesight_dir}
{python_bin_MEEP} {code_dir}fiber_platform.py --waveguide_sol "waveguide_sol-{waveguide_id}.pkl" --num_time_slices 0 --mode_idx \$SLURM_ARRAY_TASK_ID --sim_time ${{simtime}}
EOL
)

# get the job id
array_job_id=$(echo "$sbatch_output" | awk '{{print $NF}}')

#submit the analysis job
sbatch_output_plotter=$(sbatch --dependency=afterany:$array_job_id <<EOL
#!/bin/bash
#SBATCH -n 1 
#SBATCH --job-name=fiber_plotter_{waveguide_id}
#SBATCH --mem=${{memreq}}GB
#SBATCH -t ${{timereq}}
#SBATCH -o "{waveguide_dir}/{waveguide_id}-plotter.out"
#SBATCH -e "{waveguide_dir}/{waveguide_id}-plotter.err"

cd {wavesight_dir}
{python_bin} {code_dir}fiber_plotter.py {waveguide_id}
EOL
)

# get the job id
plotter_job_id=$(echo "$sbatch_output_plotter" | awk '{{print $NF}}')

# submit the propagator job
sbatch_output_bridge=$(sbatch --dependency=afterany:$plotter_job_id <<EOL
#!/bin/bash
#SBATCH -n 1 
#SBATCH --job-name=fiber_bridge_{waveguide_id}
#SBATCH --mem=${{memreq}}GB
#SBATCH -t ${{timereq}}
#SBATCH -o "{waveguide_dir}/{waveguide_id}-bridge.out"
#SBATCH -e "{waveguide_dir}/{waveguide_id}-bridge.err"

cd {wavesight_dir}
{python_bin} {code_dir}fiber_bridge.py {waveguide_id} {zProp} {nProp}
EOL
)

bridge_job_id=$(echo "$sbatch_output_bridge" | awk '{{print $NF}}')

# once the fields have been propagated across the gap, propagate them across the metalens

sbatch_output=$(sbatch --dependency=afterany:$bridge_job_id <<EOL
#!/bin/bash
#SBATCH -n {MEEP_num_cores}
#SBATCH --job-name=across_ML_{waveguide_id}
#SBATCH --mem=${{memreq2}}GB
#SBATCH -t ${{timereq2}}
#SBATCH --array=0-{num_modes}

#SBATCH -o "{waveguide_dir}/{waveguide_id}-acrossML-%a.out"
#SBATCH -e "{waveguide_dir}/{waveguide_id}-acrossML-%a.err"

cd {wavesight_dir}
{python_bin_MEEP} {code_dir}across_ml.py {waveguide_id} \$SLURM_ARRAY_TASK_ID
EOL
)

# get the job id
across_job_id=$(echo "$sbatch_output" | awk '{{print $NF}}')

# once the field have been propagated across the metalens, make plots, and propagate them to the final volume

sbatch_output_h4_plot=$(sbatch --dependency=afterany:$across_job_id <<EOL
#!/bin/bash
#SBATCH -n 1
#SBATCH --job-name=EH4-plotter_{waveguide_id}
#SBATCH --mem=8GB
#SBATCH --time=01:00:00

#SBATCH -o "{waveguide_dir}/{waveguide_id}-EH4-plotter.out"
#SBATCH -e "{waveguide_dir}/{waveguide_id}-EH4-plotter.err"

cd {wavesight_dir}
{python_bin} {code_dir}EH4_plotter.py {waveguide_id}
EOL
)

sbatch_output=$(sbatch --dependency=afterany:$across_job_id <<EOL
#!/bin/bash
#SBATCH -n 1
#SBATCH --job-name=EH4-to-EH5_{waveguide_id}
#SBATCH --mem=${{memreq2}}GB
#SBATCH --time=01:00:00
#SBATCH --array=0-4

#SBATCH -o "{waveguide_dir}/{waveguide_id}-EH4-to-EH5-%a.out"
#SBATCH -e "{waveguide_dir}/{waveguide_id}-EH4-to-EH5-%a.err"

cd {wavesight_dir}
{python_bin} {code_dir}EH4-to-EH5.py --waveguide_id {waveguide_id} --zPropindex -1
EOL
)

# get the job id
eh4_to_eh5_id=$(echo "$sbatch_output" | awk '{{print $NF}}')

# put the EH5 fields together

sbatch_output=$(sbatch --dependency=afterany:$eh4_to_eh5_id <<EOL
#!/bin/bash
#SBATCH -n 4
#SBATCH --job-name=h5_assembler_{waveguide_id}
#SBATCH --mem=2GB
#SBATCH --output="{waveguide_dir}/{waveguide_id}-h5_assembler.out"
#SBATCH --error="{waveguide_dir}/{waveguide_id}-h5_assembler.err"
#SBATCH --time=00:20:00

cd {wavesight_dir}
{python_bin} {code_dir}EH5-assembly.py {waveguide_id}
EOL
)

h5_assembler_id=$(echo "$sbatch_output" | awk '{{print $NF}}')

# submit the housekeeping job
sbatch --dependency=afterany:$h5_assembler_id <<EOL
#!/bin/bash
#SBATCH --job-name=housekeeping_{waveguide_id}
#SBATCH --output="{waveguide_dir}/{waveguide_id}-housekeeping.out"
#SBATCH --error="{waveguide_dir}/housekeeping.err"
#SBATCH --time=00:01:00

cd {wavesight_dir}
{python_bin} {code_dir}housekeeping.py {waveguide_id}
EOL
'''

config_file = '''{
    // The location of the python binary
    "python_bin": "~/anaconda/pameep/bin/python",
    // Directory where the code is located
    "code_dir": "/users/jlizaraz/CEM/wavesight/",
    // Whether to use parallel MEEP
    "parallel_MEEP": true,
    // depth of emitter in crystal host
    "emDepth": 25,
    // these two following parameters determine the size of the box where Eh5 is calculated
    // uncertainty in depth
    "emDepth_Δz": 1.6,
    // transverse uncertainty in position
    "emDepth_Δxy": 1.5, 
    // if zmin is not provided here, then it is calculated from emDepth and emDepth_Δz
    // if zmin is provided here, then zmax also need to be provided
    // and these two together then determine the axial range of the box where Eh5 is calculated
    // both zmin and zmax assume a coordinate system where the origin is at
    // the base of the pillars that create the metasurface
    "zmin": 24,
    "zmax": 32,
    // similarly xymin and xymax determine the transverse range of the box where Eh5 is calculated
    // if provided they override the values that are calculated from emDepth_Δxy
    "xymin": -3.0,
    "xymax":  3.0,
    // refractive index of crystal host
    "nHost": 2.41,
    // free-space wavelength of monochromatic field
    "λFree": 0.532,
    // ref index space between waveguide and ML
    "nBetween": 1.0, 

    // waveguide parameters
    // the radius of the waveguide core
    "coreRadius": 2.0,
    // ref index of core
    "nCore": 1.4607, 
    // ref index of cladding
    "nCladding": 1.4573, 
    // number of pixels per um
    "MEEP_resolution": 20,

    // params for metasurface design
        // metalens aperture
        "mlDiameter": 8.0,
        // the height of the cylindrical posts for ML
        "post_height":0.8, 
        // lattice constant of hexagonal lattice
        "lattice_const": 0.25, 
        // number of basis vectors for RCWA
        "numG" : 70,
        // this is how many post widths are considered for the phase/geometry calculation
        "num_post_widths": 30,
        // the direction of the linear polarization of the incident field
        "linear_polarization_direction": 0.0,
        // the minimum diameter of ML cylindrical posts
        "MIN_FEATURE_SIZE": 0.05,

    // This controls if the launching simulations are automatically scheduled
    "autorun": true,

    // The amount of memory alloted for the requirement run
    "req_run_mem_in_GB": 64,

    // Whether to post the progress to Slack
    "send_to_slack": true,
    "slack_channel": "nvs_and_metalenses",
    "show_plot": false
}'''