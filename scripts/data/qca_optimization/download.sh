#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -o %J.stdout
#BSUB -W 8:00

python download.py

