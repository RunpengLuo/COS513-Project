#!/bin/bash
#SBATCH --job-name=pmf              # Job name
#SBATCH --time=02:00:00                  # Time limit (hh:mm:ss)
#SBATCH --nodes=1                      # node count
#SBATCH --ntasks=1                       
#SBATCH --cpus-per-task=4                # Number of CPU cores per task
#SBATCH --mem=8G                         # Memory per node
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end                # send email when job ends
#SBATCH --mail-user=xb4719@princeton.edu
#SBATCH --out=pmf.out


# Load any required modules (if needed)
# module load python/3.9  # adjust based on your environment

# Activate virtual environment if needed
# source ~/envs/myenv/bin/activate

# Run the Python script
python main.py data/yelp.tsv outputs/
