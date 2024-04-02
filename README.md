# EDAD
This repository is the implementation of: An Encode-then-Decompose Approach for Unsupervised Time
Series Anomaly Detection. We propose the EDAD framework for unsupervised anomaly detection and evaluate its performance on nine open-source datasets.


## Get Started

1. Install Python 3.10, PyTorch >= 2.0.0, Wandb. Then run the following command.
    ```bash
    pip install -r requirements.txt
    ```

2. Before running EDAD, download the publicly available dataset from the [link](https://1drv.ms/f/s!AkhEmCUtJamUobN469La8ZF0d4Sbyw?e=cQcrut), unzip it and place it in the /dataset directory.


4. Use the following command to run the algorithm. 

    ```bash
    python main-all.py --lr 0.0005 \
    --input_c 25 \
    --output_c 25 \
    --dataset PSM \
    --win_size 100 \
    --d_model 512 \
    --critic sep \
    --batch_size 256 \
    --l_intra_s 1 \
    --l_intra_r 1 \
    --l_mi 1 
    ```

5. In order to run EDAD on other data sets, you need to prepare the data set and put it in the /dataset directory, and add the read operation of the dataset in the dataloder.
