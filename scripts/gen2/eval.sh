#BSUB -W 1:59
#BSUB -n 1
#BSUB -R "rusage[mem=40] span[hosts=1]"
#BSUB -o %J.stdout
#BSUB -gpu "num=1:j_exclusive=yes"
#BSUB -q gpuqueue
#####BSUB -R V100

python eval.py

