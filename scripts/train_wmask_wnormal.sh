data=$1

gs-init -s $data -o ${data}_gaustudio_wmask_wnormal --w_mask mask
python scripts/init_normal.py -s ${data}_gaustudio_wmask_wnormal
rm -r ${data}_gaustudio_wmask_wnormal/result
python train_progressive.py -s ${data}_gaustudio_wmask_wnormal -r 1  --lambda_dist 1000 \
        --w_mask masks --lambda_mask 0.1 --max_screen_size 5 -m ${data}_gaustudio_wmask_wnormal/result \
        --w_normal_prior normals  --position_lr_init 0.000016 --iteration 20000
gs-extract-pcd -m ${data}_gaustudio_wmask_wnormal/result -o ${data}_gaustudio_wmask_wnormal/result/fusion --meshing poisson-9
texrecon ${data}_gaustudio_wmask_wnormal/result/fusion/images ${data}_gaustudio_wmask_wnormal/result/fusion/fused_mesh.ply \
        ${data}_gaustudio_wmask_wnormal/result/fusion/textured_mesh --outlier_removal=gauss_clamping --data_term=area --no_intermediate_results