data=$1

gs-init -s $data -o ${data}_gaustudio --overwrite --w_mask mask
python train_progressive.py -s ${data}_gaustudio -r 1  --lambda_dist 100 --w_mask masks --lambda_mask 0.1 --max_screen_size 5 -m ${data}_gaustudio/result
gs-extract-pcd -m ${data}_gaustudio/result -o ${data}_gaustudio/result/fusion --meshing nksr
