#!/bin/bash

# Script Name: slurmit.sh
# Description: A script to receive a command as an argument and submit it as a SLURM job.
# Version: 1.0

# Default variables
CMD=""
CURRENT_DIR=$(pwd)

# Functions

usage() {
    echo "Usage: $0 [OPTIONS] 'command_to_run'"
    echo
    echo "Options:"
    echo "    -h | --help             Display this help message"
    echo "    -t | --time [DURATION]  Specify job duration (default: 03:00:00)"
    echo "    -m | --memory [SIZE]    Specify job memory (default: 8G)"
}

log() {
    local LEVEL=$1
    shift
    echo "$@"
}

generate_slurm_script() {
    local CMD="$1"
    local UNIQUE_ID=$(date +"%Y%m%d_%H%M%S")_${RANDOM}_${RANDOM}

    echo "#!/bin/bash
#SBATCH --job-name=slurm_it_${UNIQUE_ID}
#SBATCH --output=slurm_it_${UNIQUE_ID}_output.out
#SBATCH --error=slurm_it_${UNIQUE_ID}_error.err
#SBATCH --time=${JOB_TIME:-03:00:00}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=${JOB_MEMORY:-8G}

# Change to the directory from which the bash script was called
cd $CURRENT_DIR

$CMD
"
}

main() {
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            -t|--time)
                if [[ -n "$2" && "$2" != -* ]]; then
                    JOB_TIME="$2"
                    shift
                else
                    log "Error: Argument for $1 is missing" >&2
                    usage
                    exit 1
                fi
                shift
                ;;
            -m|--memory)
                if [[ -n "$2" && "$2" != -* ]]; then
                    JOB_MEMORY="$2"
                    shift
                else
                    log "Error: Argument for $1 is missing" >&2
                    usage
                    exit 1
                fi
                shift
                ;;
            *)
                if [[ -z "$CMD" ]]; then
                    CMD="$1"
                    shift
                else
                    log "Error: Extra arguments provided" >&2
                    usage
                    exit 1
                fi
                ;;
        esac
    done

    if [[ -z "$CMD" ]]; then
        log "Error: No command provided" >&2
        usage
        exit 1
    fi

    SLURM_SCRIPT=$(generate_slurm_script "$CMD")
    SCRIPT_FILE="schedule_$(date +"%Y%m%d_%H%M%S")_${RANDOM}_${RANDOM}.slurm"
    echo "$SLURM_SCRIPT" > "$SCRIPT_FILE"

    log "SLURM script created: $SCRIPT_FILE"

    sbatch "$SCRIPT_FILE"
}

# Initialization and main program execution
main "$@"
