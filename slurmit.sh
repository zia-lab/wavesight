#!/bin/bash
# receive a command as an argument and submit it as a SLURM job
# Usage: slurmit.sh 'command_to_run'
# Check if command is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 'command_to_run'"
    exit 1
fi

# Extract command and current directory
CMD="$1"
CURRENT_DIR=$(pwd)

# Generate a unique identifier for the job using a timestamp and a random number
UNIQUE_ID=$(date +"%Y%m%d_%H%M%S")_${RANDOM}

# Create a SLURM script content
SLURM_SCRIPT="#!/bin/bash
#SBATCH --job-name=slurm_it_${UNIQUE_ID}
#SBATCH --output=slurm_it_${UNIQUE_ID}_output.txt
#SBATCH --error=slurm_it_${UNIQUE_ID}_error.txt
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8G

# Change to the directory from which the bash script was called
cd $CURRENT_DIR

$CMD
"

# Write the SLURM script to a file
SCRIPT_FILE="schedule_${UNIQUE_ID}.slurm"
echo "$SLURM_SCRIPT" > $SCRIPT_FILE

echo "SLURM script created: $SCRIPT_FILE"

# Submit the SLURM job
sbatch $SCRIPT_FILE

