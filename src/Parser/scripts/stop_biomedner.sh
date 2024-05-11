#!/bin/bash

# Define a function to stop a process by its name
stop_process() {
  process_name="$1"
  pid=$(ps auxww | grep "$process_name" | grep -v grep | awk '{print $2}' | sort -r)
  if [ "$pid" != "" ]; then
    # Kill each PID one by one
    for p in $pid; do
      kill -9 "$p"
      echo "Stopped $process_name (PID: $p)"
    done
  else
    echo "No $process_name found to stop."
  fi
}

# Call the function for each process
stop_process "biomedner_server.py"
stop_process "GNormPlusServer.main.jar"
stop_process "tmVar2Server.main.jar"
stop_process "disease_normalizer_21.jar"
stop_process "gnormplus-normalization_21.jar"

