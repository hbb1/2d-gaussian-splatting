data=$1
postix=gaustudio_wmask_wnormal
# gs-init -s $data -o ${data}_${postix} --w_mask mask
python scripts/init_normal.py -s ${data}_${postix}
rm -r ${data}_${postix}/result_2
python train_progressive.py -s ${data}_${postix} -r 1  --lambda_dist 1000 \
        --w_mask masks --lambda_mask 0.1 --w_normal_prior normals \
        --max_screen_size 5 -m ${data}_${postix}/result_2 --iteration 7000
gs-extract-pcd -m ${data}_${postix}/result_2 -o ${data}_${postix}/result_2/fusion_2 \
        --meshing sap --config 2dgs