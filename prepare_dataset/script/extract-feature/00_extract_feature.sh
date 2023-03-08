#!/bin/bash

cd `dirname $0`
ENV_FILE=./env_file
for i in "$@"; do
    case $i in
        -e=*|--env=*)
        ENV_FILE="${i#*=}"
        shift
        ;;
    esac
done
if [[ ! -e ${ENV_FILE} ]]; then
    echo "ERROR>>> $ENV_FILE does not exist!"
    exit 1
fi
source ${ENV_FILE}

jobname=extract_feature
out_dir=$DATA/$jobname

main()
{
    start_time=`date`
    echo "$jobname $ENV_FILE"

    echo "DATASET : $DATASET"
    run

    finish_time=`date`
    util/print_process_time.sh "$jobname $DATASET" "$start_time" "$finish_time"
}

run()
{
    # in_dir=$PROJ_HOME/data/reform-rawdata/${DATASET}/split_by_chid
    in_dir=$PROJ_HOME_2/reform-rawdata/${DATASET}/merge_by_unitNo
    # in_dir=$PROJ_HOME_2/data/reform-rawdata/${DATASET}/split_by_chid

    rm -rf $out_dir || true
    mkdir -p $out_dir

    if [[ ${DATASET} == SVRCLIVE ]]; then
        app=extract_feature_svrclive
    elif [[ ${DATASET} == SVRC ]]; then
        # app=extract_feature_svrc
        app=extract_feature_svrc_old
    elif [[ ${DATASET} == ILSAN ]]; then
        app=extract_feature_ilsan
    elif [[ ${DATASET} == MIMIC ]]; then
        app=extract_feature_mimic
    elif [[ ${DATASET} == MIMICED ]]; then
        app=extract_feature_mimiced
        in_dir=$PROJ_HOME_2/reform-rawdata/${DATASET}/merge_by_stayid
    elif [[ ${DATASET} == SVRCKN ]]; then
        app=extract_feature_svrckn
    else
        echo "invalid dataset $DATASET"
        exit 1
    fi

    python ${SRC}/${app}.py     --out_dir=${out_dir} \
                                --in_dir=${in_dir}

    # if [[ ${DATASET} == SVRC ]]; then
    #     python ${SRC}/update_extra_features.py
    # fi
}

main $@
