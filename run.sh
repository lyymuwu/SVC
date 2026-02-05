K=8
# export CUDA_VISIBLE_DEVICES=1

#  ViT-B-16 ViT-L-14
for BACKBONE in ViT-B-32; do
    echo "Backbone: ${BACKBONE}"

    echo "Stage: Merging model training"
    for merge in TA DARE TIES TSV-M ISO-C ISO-CTS; do
        echo "Merging method: ${merge}"
        if [ "${merge}" = "TSV-M" ]; then
            alpha=1.0
        else
            alpha=$(echo "scale=4; 1/${K}" | bc)
        fi
        python main.py --model ${BACKBONE} --merge ${merge} --alpha ${alpha}
        python main.py --model ${BACKBONE} --merge ${merge} --alpha ${alpha} --c
    done
done
