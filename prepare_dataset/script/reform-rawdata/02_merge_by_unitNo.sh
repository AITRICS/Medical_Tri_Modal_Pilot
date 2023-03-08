#!/bin/bash
# @Author: steve
# @Date:   2022-03-03 16:34:41
# @Last Modified by:   steve
# @Last Modified time: 2022-03-03 17:45:44

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

if [[ ${DATASET} == MIMICED ]]; then
    jobname=merge_by_stayid
else
    jobname=merge_by_unitNo
fi

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
    in_dir=${DATA}/split_by_chid
    # rm -rf $out_dir || true
    # mkdir -p $out_dir

    python $SRC/${jobname}.py   --out_dir=${out_dir} \
                                --in_dir=${in_dir} \
                                --rawdata=${RAWDATA} \
                                --dataset=${DATASET}
}

main $@
