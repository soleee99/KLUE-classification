LR=$1
EPOCHS=$2
BSZ=$3
MAX_SEQ_LEN=$4
FT_TYPE=$5

NOW=$(date +"%Y-%m-%d_%H-%M-%S")     
RUNDIR="outputs/${FT_TYPE}/lr-${LR}_epoch-${EPOCHS}_bsz-${BSZ}_msl-${MAX_SEQ_LEN}/${NOW}"
mkdir -p $RUNDIR
mkdir -p ${RUNDIR}/ckpt
mkdir -p ${RUNDIR}/log

RUN_CMD="python main.py conf/base.yaml \
        hydra.run.dir=${RUNDIR} \
        train_path='${PWD}/data/ynat-v1.1/ynat-v1.1_train.json' \
        dev_path='${PWD}/data/ynat-v1.1/ynat-v1.1_dev.json' \
        save_path='${PWD}/${RUNDIR}/ckpt' \
        epochs=$EPOCHS \
        batch_size=$BSZ \
        lr=$LR \
        finetune_type=$FT_TYPE \
        max_seq_len=$MAX_SEQ_LEN \
        val_interval=1 \
        log_interval=1 \
        save_interval=10 \
        train_log_path='${PWD}/${RUNDIR}/log/train.log' \
        test_log_path='${PWD}/${RUNDIR}/log/test.log' \
        "

echo $RUN_CMD
eval $RUN_CMD
