#BSUB -W 0:10
#BSUB -R "rusage[mem=1] span[ptile=1]"
#BSUB -o %J.stdout
#BSUB -J "data[1-1100]"

python preprocess.py $((LSB_JOBINDEX-1))

