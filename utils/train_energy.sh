python main.py \
    --root data_gen/data/ \
    --task regression \
    --target energy_per_atom \
    --resume best_models/energy_per_atom.pth.tar \
    --atom-fea-len 64 \
    --h-fea-len 32 \
    --n-conv 4 \
    --gpu-id 1 \
    | tee 2>&1 energy.log
