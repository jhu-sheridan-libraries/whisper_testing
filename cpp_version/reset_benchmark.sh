#!/bin/bash

# Reset the benchmark file to a clean state
echo "Resetting benchmark file to a clean state..."
echo "timestamp,filename,duration_seconds,num_speakers,model,processing_time_seconds,real_time_factor,cpu_percent,cpu_count,memory_percent" > whisper_benchmarks.csv
echo "2025-03-25 00:10:24,SFd2CK2qiVI.mp3,1300.885,1,base,8814.24968290329,6.7755794577562884,5.3,8,72.9" >> whisper_benchmarks.csv
echo "Benchmark file has been reset." 