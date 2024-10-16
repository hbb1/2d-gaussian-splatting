data=$1

python scripts/init_bg_gs.py
gs-init -s $data -o ${data}_gaustudio --overwrite --pcd combined
python train_with_bg.py -s ${data}_gaustudio -r 2  --lambda_dist 100 -m ${data}_gaustudio/result
gs-extract-pcd -m ${data}_gaustudio/result -o ${data}_gaustudio/result/fusion --meshing nksr