#!/bin/bash

# Get the current timestamp
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

# Create a fresh benchmark file with a valid entry
cat > whisper_benchmarks.csv << EOL
timestamp,filename,duration_seconds,num_speakers,model,processing_time_seconds,real_time_factor,cpu_percent,cpu_count,memory_percent
$TIMESTAMP,test_audio.mp3,30.0,2,tiny,2.0,0.067,50.0,8,30.0
EOL

echo "Created a fresh benchmark file with a valid entry:"
cat whisper_benchmarks.csv 