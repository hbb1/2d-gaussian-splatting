data=$1
postix=gaustudio_fast
gs-init -s $data -o ${data}_${postix}
rm -r ${data}_${postix}/result_2
python train_fast.py -s ${data}_${postix} -r 1  -m ${data}_${postix}/result_2 --iteration 7000