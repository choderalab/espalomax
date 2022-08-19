#BSUB -W 0:10
#BSUB -R "rusage[mem=1] span[ptile=1]"
#BSUB -o %J.stdout
#BSUB -J "data[10000-19999]"

python data.py $LSB_JOBINDEX

