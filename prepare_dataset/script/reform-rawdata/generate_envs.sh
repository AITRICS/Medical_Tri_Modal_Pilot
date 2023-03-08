#!/bin/bash

cd `dirname $0`

dataset_list=(
        SVRC
        MIMIC
        )

main()
{
    generate
}

generate()
{
    for dataset in "${dataset_list[@]}"; do
        if [ $dataset == SVRC ]; then
            chid_hash_size=1000
        elif [ $dataset == MIMIC ]; then
            chid_hash_size=1000
        else
            exit
        fi

        generate_each ${dataset} ${chid_hash_size}
    done
}

generate_each()
{
    dataset=$1
    chid_hash_size=$2

    in_file=_env_file.template
    out_file=env_file.${dataset}

    sed -e "s/@__DATASET__@/${dataset}/g" \
        -e "s/@__CHID_HASH_SIZE__@/${chid_hash_size}/g" \
        ${in_file} > ${out_file}

    echo $out_file
}

main $@
