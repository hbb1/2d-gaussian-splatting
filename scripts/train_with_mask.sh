data=$1
postix=gaustudio_wmask
gs-init -s $data -o ${data}_${postix} --w_mask mask
rm -r ${data}_${postix}/result_2
python train_progressive.py -s ${data}_${postix} -r 1  --lambda_dist 1000 \
        --w_mask masks --lambda_mask 0.1 --max_screen_size 5 \
        -m ${data}_${postix}/result_2 --iteration 20000
gs-extract-pcd -m ${data}_${postix}/result_2 -o ${data}_${postix}/result_2/fusion \
        --meshing sap --config 2dgs
texrecon ${data}_${postix}/result_2/fusion/images ${data}_${postix}/result_2/fusion/fused_mesh.ply \
        ${data}_${postix}/result_2/fusion/textured_mesh --outlier_removal=gauss_clamping \
        --data_term=area --no_intermediate_results