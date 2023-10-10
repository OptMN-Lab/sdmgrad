mkdir -p ./save
mkdir -p ./trainlogs

method=sdmgrad
seed=0
lamda=0.3
niter=20

nohup python -u trainer.py --method=$method --seed=$seed --lamda=$lamda --niter=$niter > trainlogs/sdmgrad-lambda$lamda-sd$seed.log 2>&1 &
