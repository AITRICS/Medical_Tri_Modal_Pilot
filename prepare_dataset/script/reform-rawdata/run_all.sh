#!/bin/bash

env_file_list=(
        env_file.SVRC
        # env_file.SVRCLIVE
        env_file.ILSAN
        env_file.MIMIC
        )

for env_file in "${env_file_list[@]}"; do
    echo "${env_file} start!"
    ./run.sh --env=${env_file}
    echo "${env_file} finish!"
done
