#!/bin/bash

# Define the number of sessions you want to create
start_index=1
end_index=9

# Loop over a sequence of numbers
for i in $(seq $start_index $end_index)
do
    # Construct the session name using the index
    session_name="variac-$i"
    sleep_time=$((i*30))
    # Create a new tmux session in detached mode
    tmux new-session -d -s "$session_name"

    # Send your commands to the tmux session
    # Here, we use the index in the echo command for demonstration
    tmux send-keys -t "$session_name" "conda activate pameep" C-m
    tmux send-keys -t "$session_name" "date" C-m
    tmux send-keys -t "$session_name" "sleep $sleep_time" C-m
    tmux send-keys -t "$session_name" "python metaorchestra.py variant_config_$i.jsonc" C-m
done

echo "$num_sessions sessions are set up!"
