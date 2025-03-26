#!/bin/bash

# Simple script to fix the benchmark file
BENCHMARK_FILE="whisper_benchmarks.csv"

if [ ! -f "$BENCHMARK_FILE" ]; then
    echo "Benchmark file not found: $BENCHMARK_FILE"
    exit 1
fi

# Create a temporary file with just the header and valid rows
echo "Fixing benchmark file: $BENCHMARK_FILE"
echo "timestamp,filename,duration_seconds,num_speakers,model,processing_time_seconds,real_time_factor,cpu_percent,cpu_count,memory_percent" > temp_benchmark.csv

# Find and append only non-empty rows
grep -v "^,,0,0,,0,0,0,0,0" "$BENCHMARK_FILE" | grep -v "^$" | grep -v "^timestamp" >> temp_benchmark.csv || true

# Replace the original file
mv temp_benchmark.csv "$BENCHMARK_FILE"

echo "Benchmark file fixed. Contents now:"
cat "$BENCHMARK_FILE" 