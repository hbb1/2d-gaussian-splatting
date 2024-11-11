data=$1
postix=gaustudio_wnormal
# gs-init -s $data -o ${data}_${postix} --pcd combined
# python scripts/init_normal.py -s ${data}_${postix}
rm -r ${data}_${postix}/result_2
python train.py -s ${data}_${postix} -r 1  --contribution_prune_ratio 0.5 \
                            --lambda_normal_prior 1 --lambda_dist 10 \
                            --densify_until_iter 3000 --iteration 7000 \
                            -m ${data}_${postix}/result_2 --w_normal_prior normals
