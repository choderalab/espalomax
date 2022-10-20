#BSUB -W 0:15
#BSUB -n 1
#BSUB -o %J.stdout
#BSUB -J "md[1-10]"

python grab.py $LSB_JOBINDEX





