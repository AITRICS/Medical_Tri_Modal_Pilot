#!/bin/bash

if [ $# -ne 3 ]; then
    exit 1
fi

tag=$1
t1=$2
t2=$3

elapsed=$(( $(date -d "$t2" "+%s") - $(date -d "$t1" "+%s") ))

echo "start  $tag $t1"
echo "finish $tag $t2 (elapsed: $elapsed seconds)"
echo ""
