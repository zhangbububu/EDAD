export CUDA_VISIBLE_DEVICES=4

python main.py --anomaly_ratio 0.6 --num_epochs 100   --batch_size 256  --mode train --dataset SMD  --data_path dataset/kdd/dataset/SMD   --input_c 38 --output_c 38
# python main.py --anomaly_ratio 0.6 --num_epochs 100   --batch_size 256     --mode test    --dataset SMD   --data_path dataset/SMD     --input_c 38     --pretrained_model 20