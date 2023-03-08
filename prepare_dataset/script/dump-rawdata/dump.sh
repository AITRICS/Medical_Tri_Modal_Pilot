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
if [ ! -e $ENV_FILE ]; then
    echo "ERROR>>> $ENV_FILE does not exist!"
    exit 1
fi
source $ENV_FILE

jobname=dump

main()
{
    start_time=`date`
    echo "$jobname $ENV_FILE"

    echo "DATASET : $DATASET"
    copy_data

    finish_time=`date`
    util/print_process_time.sh "$jobname $DATASET" "$start_time" "$finish_time"

}

copy_data()
{

    if [ $DATASET = SVRC ]; then
        path_src=${RAWDATA_ORG_SVRC}
    elif [ $DATASET = MIMIC ]; then
        path_src=${RAWDATA_ORG_MIMIC}
    elif [ $DATASET = MIMICED ]; then
        path_src=${RAWDATA_ORG_MIMICED}
    else
        echo "ERROR>>> $DATASET is not defined dataset"
        exit 1
    fi

    path_dst=$RAWDATA

    rm -rf $path_dst || true
    mkdir -p $path_dst
    rsync -e 'ssh -p 6150' -avL --include="*.csv" --exclude="*" $path_src/ $path_dst
}

main $@
