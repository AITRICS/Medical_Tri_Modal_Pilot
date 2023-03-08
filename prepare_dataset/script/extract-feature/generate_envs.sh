#!/bin/bash

cd `dirname $0`

dataset_list=(
        SVRCLIVE
        SVRC
        ILSAN
        MIMIC
        SVRCKN
        )

main()
{
    generate
}

generate()
{
    for dataset in "${dataset_list[@]}"; do
        generate_each ${dataset}
    done
}

generate_each()
{
    dataset=$1

    in_file=_env_file.template
    out_file=env_file.${dataset}


    sed -e "s/@__DATASET__@/${dataset}/g" \
        ${in_file} > ${out_file}

    echo ${out_file}
}

main $@
