import os
import glob
import json
import pandas as pd
import argparse

def main(result_dirs):
    # Find all results.json files
    results = glob.glob(os.path.join(result_dirs, '*', 'results.json'))

    # Initialize lists to store the data
    data = []

    # Read each JSON file and extract the metrics
    for result_file in results:
        with open(result_file, 'r') as f:
            result = json.load(f)
        
        # Extract the experiment name from the file path
        exp_name = result_file.split('/')[-2]
        
        # Extract the metrics
        psnr = result['ours_30000'].get('PSNR', 'N/A')
        ssim = result['ours_30000'].get('SSIM', 'N/A')
        lpips = result['ours_30000'].get('LPIPS', 'N/A')
        
        # Append the data to the list
        data.append({
            'Experiment': exp_name,
            'PSNR': psnr,
            'SSIM': ssim,
            'LPIPS': lpips
        })

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)

    # Sort the DataFrame by experiment name
    df = df.sort_values('Experiment')

    # Display the table
    print(df.to_string(index=False))

    # calculate average PSNR, SSIM, and LPIPS
    avg_psnr = df['PSNR'].mean()
    avg_ssim = df['SSIM'].mean()
    avg_lpips = df['LPIPS'].mean()

    print(f"Average PSNR: {avg_psnr}")
    print(f"Average SSIM: {avg_ssim}")
    print(f"Average LPIPS: {avg_lpips}")

    # Optionally, save the table to a CSV file
    # df.to_csv('results_summary.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process results from JSON files.")
    parser.add_argument("--model_path", "-m", help="model path")
    args = parser.parse_args()
    main(args.model_path)