#!/bin/bash

# Usage: ./append_benchmark.sh filename duration speakers model processing_time rtf [cpu_percent cpu_count memory_percent]

if [ $# -lt 5 ]; then
  echo "Usage: $0 filename duration speakers model processing_time rtf [cpu_percent cpu_count memory_percent]"
  exit 1
fi

TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
FILENAME=$1
DURATION=$2
SPEAKERS=$3
MODEL=$4
PROCESSING_TIME=$5
RTF=$6
CPU_PERCENT=${7:-50.0}
CPU_COUNT=${8:-$(nproc)}
MEMORY_PERCENT=${9:-30.0}

echo "$TIMESTAMP,$FILENAME,$DURATION,$SPEAKERS,$MODEL,$PROCESSING_TIME,$RTF,$CPU_PERCENT,$CPU_COUNT,$MEMORY_PERCENT" >> whisper_benchmarks.csv
echo "Added benchmark entry for $FILENAME"
