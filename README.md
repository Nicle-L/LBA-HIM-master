# LBA-HIM


## Dataset Preparation and Model Training Instructions

### 1. Dataset Preparation
1. Download the dataset from the link provided in our paper.
2. Process the dataset following the method described by Guo et al. [1].
3. Convert the processed data into `.txt` format.
4. Split the dataset into training and testing sets.
5. Ensure that both the processed data and the `.txt` files for the training and testing sets are placed in the same directory (e.g., `total_data`).

### 2. Updating Paths
1. Modify `root_dir = ""` in `dataset.py` to the path of the `total_data` directory.
2. Create three directories to store logs, test results, and model checkpoints:
   - `out_dir_train` (for training logs)
   - `out_dir_test` (for test results)
   - `checkpoint_dir` (for model weights)
3. Set `train_path` and `test_path` in the `train.py` command to the paths of the prepared training and testing sets.
4. Configure the lambda value according to the instructions in our paper.

### 3. Running the Training Script
1. Use the `cd` command to navigate to the project directory.
2. Run the following command in the terminal:

   ```sh
   python train.py
   ```
### 3. Weight 
Link: https://pan.baidu.com/s/1OGlI4Gs-LKn4OJRDjx807g?pwd=vvbf Extract code: vvbf 
### 4. Our Dataset 
Link: https://pan.baidu.com/s/1BtZffhKCKaZGLC9_b3EK-Q?pwd=wqh4 Extract code: wqh4 

The txt file for our cross-validation dataset is avirix2_file_list.txt. 
The txt file name for the training set is train1.txt.
The txt file name for the test set is test1.txt.
please change the dataset paths as per our step by step procedure to ensure smooth training or testing. 

[1] Guo, Y., Tao, Y., Chong, Y., Pan, S., & Liu, M. (2023). "Edge-Guided Hyperspectral Image Compression With Interactive Dual Attention." IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1-17.
DOI: 10.1109/TGRS.2022.3233375.
## Citation

If you find this work useful for your research, please consider citing our paper:

J. Liu, L. Zhang, J. Wang, and L. Qu,  
“Lightweight Band-Adaptive Hyperspectral Image Compression With Feature Decouple and Recurrent Model,”  
*IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*,  
vol. 18, pp. 16733–16749, 2025.  
doi: 10.1109/JSTARS.2025.358493
