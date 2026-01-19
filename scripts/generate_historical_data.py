"""
Generate Multi-Year Training Data
Runs prepare_training_data.py for years 2019-2024.
"""
import subprocess
import sys
import os

YEARS = [2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016]
SAMPLES_PER_YEAR = 300 # 300 fire + 300 no-fire = 600 per year

def main():
    print(f"Starting multi-year data generation for: {YEARS}")
    
    python_cmd = sys.executable # Use current python interpreter
    
    for year in YEARS:
        print(f"\n========================================")
        print(f"PROCESSING YEAR {year}")
        print(f"========================================")
        
        output_dir = f"data/training/{year}"
        labels_path = os.path.join(output_dir, "labels.csv")
        
        # Skip if already done
        if os.path.exists(labels_path):
            print(f"  [SKIP] Data for {year} already exists in {output_dir}")
            continue
        
        cmd = [
            python_cmd, "scripts/prepare_training_data.py",
            "--year", str(year),
            "--fire-samples", str(SAMPLES_PER_YEAR),
            "--nofire-samples", str(SAMPLES_PER_YEAR),
            "--output-dir", output_dir
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully generated data for {year}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {year}: {e}")
            # Continue to next year even if one fails
            continue

if __name__ == "__main__":
    main()
