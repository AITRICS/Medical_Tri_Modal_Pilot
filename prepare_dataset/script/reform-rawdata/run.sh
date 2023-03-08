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

jobname=run

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
    run_each ./00_split_by_hash.sh
    run_each ./01_split_by_chid.sh
    run_each ./02_merge_by_unitNo.sh
}

run_each()
{
    bash $1 --env=$ENV_FILE
}

main $@
