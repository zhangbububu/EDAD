export CUDA_VISIBLE_DEVICES=0
python main.py --anormly_ratio 1 --num_epochs 3    --batch_size 256  --mode train --dataset PSM  --data_path dataset/PSM --input_c 25    --output_c 25
python main.py --anormly_ratio 1  --num_epochs 10       --batch_size 256     --mode test    --dataset PSM   --data_path dataset/PSM  --input_c 25    --output_c 25  --pretrained_model 20

CUDA_VISIBLE_DEVICES=0 python main-all.py  --lr 0.0005 \
    --input_c 25 \
    --output_c 25 \
    --dataset PSM \
    --data_path ./dataset/kdd/dataset/PSM \
    --tem 2 \
    --anomaly_ratio 1 \
    --win_size 100 \
    --num_epochs 20 





    # parser.add_argument('--lr', type=float, default=0.0005)
    # parser.add_argument('--input_c', type=int, default=25)
    # parser.add_argument('--output_c', type=int, default=25)
    # parser.add_argument('--dataset', type=str, default='PSM')
    # parser.add_argument('--data_path', type=str, default='./ataset/PSM')
    # parser.add_argument('--tem', type=float, default=2)
    # parser.add_argument('--anomaly_ratio', type=float, default=1)
    # parser.add_argument('--win_size', type=int, default=100)
    # parser.add_argument('--num_epochs', type=int, default=20)

