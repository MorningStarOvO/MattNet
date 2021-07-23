GPU_ID=$1

DATASET=refcoco+
SPLITBY=unc
ID="coco+_erase_qxy"
# ID="coco+_pretrain_qxy_bert_way3"

case ${DATASET} in
    refcoco)
        for SPLIT in testA testB
        do
            CUDA_VISIBLE_DEVICES=${GPU_ID} python -u ./tools/eval_easy.py \
                --dataset ${DATASET} \
                --splitBy ${SPLITBY} \
                --split ${SPLIT} \
                --id ${ID} \
		2>&1 | tee logs/test_${ID}_${SPLIT}
        done
    ;;
    refcoco+)
        for SPLIT in testA testB
        do
            CUDA_VISIBLE_DEVICES=${GPU_ID} python -u ./tools/eval_easy.py \
                --dataset ${DATASET} \
                --splitBy ${SPLITBY} \
                --split ${SPLIT} \
                --id ${ID} \
		2>&1 | tee logs/test_${ID}_${SPLIT}
        done
    ;;
    refcocog)
        for SPLIT in val test
        do
            CUDA_VISIBLE_DEVICES=${GPU_ID} python -u ./tools/eval_easy.py \
                --dataset ${DATASET} \
                --splitBy ${SPLITBY} \
                --split ${SPLIT} \
                --id ${ID} \
		2>&1 | tee logs/test_${ID}_${SPLIT}
        done
    ;;
esac
