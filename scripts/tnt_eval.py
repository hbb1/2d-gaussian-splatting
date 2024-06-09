import os
from argparse import ArgumentParser

tnt_360_scenes = ['Barn', 'Caterpillar', 'Ignatius', 'Truck']
tnt_large_scenes = ['Meetingroom', 'Courthouse']

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval/tnt")
parser.add_argument('--TNT_data', "-TNT_data", required=True, type=str)
args, _ = parser.parse_known_args()

if not args.skip_metrics:
    parser.add_argument('--TNT_GT', required=True, type=str)
    args = parser.parse_args()


if not args.skip_training:
    common_args = " --quiet --test_iterations -1 --depth_ratio 1.0 -r 2 "
    
    for scene in tnt_360_scenes:
        source = args.TNT_data + "/" + scene
        print("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args + ' --lambda_dist 100')
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)

    for scene in tnt_large_scenes:
        source = args.TNT_data + "/" + scene
        print("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args+ ' --lambda_dist 10')
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)


if not args.skip_rendering:
    all_sources = []
    common_args = " --quiet --depth_ratio 1.0 "

    for scene in tnt_360_scenes:
        source = args.TNT_data + "/" + scene
        print("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + common_args + ' --num_cluster 1 --voxel_size 0.004 --sdf_trunc 0.016 --depth_trunc 3.0')
        os.system("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + common_args + '  --num_cluster 1 --voxel_size 0.004 --sdf_trunc 0.016 --depth_trunc 3.0')

    for scene in tnt_large_scenes:
        source = args.TNT_data + "/" + scene
        print("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + common_args + ' --num_cluster 1 --voxel_size 0.006 --sdf_trunc 0.024 --depth_trunc 4.5')
        os.system("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + common_args + ' --num_cluster 1 --voxel_size 0.006 --sdf_trunc 0.024 --depth_trunc 4.5')

if not args.skip_metrics:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    all_scenes = tnt_360_scenes + tnt_large_scenes

    for scene in all_scenes:
        ply_file = f"{args.output_path}/{scene}/train/ours_{iteration}/fuse_post.ply"
        string = f"OMP_NUM_THREADS=4 python {script_dir}/eval_tnt/run.py " + \
            f"--dataset-dir {args.TNT_GT}/{scene} " + \
            f"--traj-path {args.TNT_data}/{scene}/{scene}_COLMAP_SfM.log " + \
            f"--ply-path {ply_file}"
        print(string)
        os.system(string)