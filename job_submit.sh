python main.py \
    --root data_gen/data/ \
    --task classification \
    --target MIT \
    --resume best_models/MIT.pth.tar \
    --epochs 2 \
    --batch-size 128 \
    --atom-fea-len 64 \
    --h-fea-len 32 \
    --n-conv 4 \
    --gpu-id 1 \
    | tee 2>&1 MIT.log
