python main.py \
    --root data_gen/data/ \
    --task regression \
    --target band_gap \
    --atom-fea-len 64 \
    --h-fea-len 32 \
    --n-conv 4 \
    --gpu-id 1 \
    | tee 2>&1 band_gap.log
    #--resume pre-trained/band_gap.pth.tar \
