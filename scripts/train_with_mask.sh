data=$1

gs-init -s $data -o ${data}_gaustudio_wmask --w_mask mask
python train_progressive.py -s ${data}_gaustudio_wmask -r 1  --lambda_dist 1000 \
        --w_mask masks --lambda_mask 0.1 --max_screen_size 5 -m ${data}_gaustudio_wmask/result \
        --position_lr_init 0.000016 --iteration 20000
gs-extract-pcd -m ${data}_gaustudio_wmask/result -o ${data}_gaustudio_wmask/result/fusion --meshing poisson
texrecon ${data}_gaustudio_wmask/result/fusion/images ${data}_gaustudio_wmask/result/fusion/fused_mesh.ply \
        ${data}_gaustudio_wmask/result/fusion/textured_mesh --outlier_removal=gauss_clamping --data_term=area --no_intermediate_results