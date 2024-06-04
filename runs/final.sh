
for SEED in 20241 20242 20243
do
    nohup python main.py task=${TASK} model.rot=1 &
    nohup python main.py task=${TASK} model.rot=3 &
    nohup python main.py task=${TASK} model.rot=6 &
    wait
    nohup python main.py task=${TASK} model.rot=1 model.insta=True model.insta_params.num_samples=1 &
    nohup python main.py task=${TASK} model.rot=1 model.insta=True model.insta_params.num_samples=3 &
    wait
    nohup python main.py task=${TASK} model.rot=1 model.insta=True model.insta_params.num_samples=6 &
    wait
    nohup python main.py task=${TASK} model.rot=3 model.partial=True &
    nohup python main.py task=${TASK} model.rot=6 model.partial=True &
    wait
done
