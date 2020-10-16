python predict.py \
    --modelpath ./best_models/MIT.pth.tar \
    --cifpath ./perovskite_test/test/ \
    --task classification \
    --target MIT \
    --batch-size 18 \
    | tee 2>&1 MIT.log
