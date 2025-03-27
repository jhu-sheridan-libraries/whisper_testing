#!/bin/bash

BENCHMARK_FILE="whisper_benchmarks.csv"
HEADER="timestamp,filename,duration_seconds,num_speakers,model,processing_time_seconds,real_time_factor,cpu_percent,cpu_count,memory_percent"

cleanup_and_exit() {
    rm -f temp_benchmark*.csv
    exit "${1:-0}"
}

create_new_file() {
    echo "$HEADER" > "$BENCHMARK_FILE"
    echo "Created new benchmark file with header"
}

fix_benchmark_file() {
    # Create temporary file with correct header
    echo "$HEADER" > temp_benchmark.csv
    
    # If file exists, process its contents
    if [ -f "$BENCHMARK_FILE" ]; then
        # Skip header if present, append non-empty and non-zero rows
        awk -F, '
            NR > 1 || $0 !~ /timestamp/ {
                # Skip empty lines or lines with all zeros
                if (NF > 1 && !($0 ~ /^,,0,0,,0,0,0,0,0/)) {
                    print
                }
            }
        ' "$BENCHMARK_FILE" >> temp_benchmark.csv
    fi

    # Replace original file with fixed version
    mv temp_benchmark.csv "$BENCHMARK_FILE"
    echo "Benchmark file fixed successfully"
}

main() {
    # Ensure we clean up on script exit
    trap cleanup_and_exit EXIT

    if [ ! -f "$BENCHMARK_FILE" ]; then
        echo "Benchmark file not found: $BENCHMARK_FILE"
        create_new_file
    else
        echo "Fixing benchmark file: $BENCHMARK_FILE"
        fix_benchmark_file
    fi

    # echo -e "\nBenchmark file contents:"
    # cat "$BENCHMARK_FILE"
}

main