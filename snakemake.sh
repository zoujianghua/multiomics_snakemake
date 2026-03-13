#!/bin/bash
#SBATCH --job-name=snakemake-ctl
#SBATCH --partition=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=run.out
#SBATCH --error=run.err


set -euo pipefail
echo "=== Start Snakemake (controller) ==="
date; hostname; echo "PWD: $(pwd)"

PROJECT_ROOT="/public/home/zoujianghua/multiomics_project"
cd "$PROJECT_ROOT"

# 1) 先把系统模块干掉（防止再把奇怪的 env01、老 gcc 之类塞进来）
module purge 2>/dev/null || true

unset LD_LIBRARY_PATH 

# 2) 初始化 conda（只用你自己的 miniconda）
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
fi

# 3) 激活 snakemake 环境
conda activate snakemake7

# 不要再 export PATH="$HOME/bin:..."，让 conda 自己管 PATH
# 不要再碰 LD_LIBRARY_PATH

snakemake \
  -s "$PROJECT_ROOT/Snakefile" \
  --profile "$PROJECT_ROOT/profiles/slurm" \
  --use-conda --conda-frontend conda \
  --rerun-triggers mtime params code \
  --rerun-incomplete \
  --show-failed-logs \
  --cores 999 -j 300 \
  --latency-wait 7000 \
  --resources netio=600 \
  --printshellcmds --show-failed-logs --keep-going \
  all_hsi

echo "=== Snakemake Finished ==="
date

