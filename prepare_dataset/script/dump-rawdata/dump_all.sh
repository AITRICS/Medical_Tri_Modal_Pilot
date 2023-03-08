#!/bin/bash

env_file_list=(
        env_file.SVRC
        env_file.MIMIC
        )

for env_file in "${env_file_list[@]}"; do
    echo "${env_file} start!"
    ./dump.sh --env=${env_file}
    echo "${env_file} finish!"
done