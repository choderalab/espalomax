#BSUB -W 23:59
#BSUB -n 32
#BSUB -R "rusage[mem=5] span[hosts=1]"
#BSUB -o %J.stdout
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -q gpuqueue
#BSUB -R V100

python run_compile.py

