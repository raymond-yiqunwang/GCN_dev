/home/yiqun/miniconda3/bin/python main.py \
    --root /data2/yiqun/cgcnn_dev_data/data \
    --target energy_per_atom \
    --task regression \
    --gpu-id 3 \
    | tee 2>&1 job.log
