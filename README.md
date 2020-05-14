## 2D and 3D Hand Pose Estimation from a Single RGB Image

<p align="center">
  <img src=pose_estimation_gif.gif/>
</p>

<p align="center">
  <img src=pose_estimate.jpg/>
</p>


### Introduction
This is a CSCI 5561 course project done by Eng Hock Lee and Chih-Tien Kuo. 
This project aimed to improve the existing work by [Ge et al.](https://github.com/3d-hand-shape/hand-graph-cnn) [1]. 
Specifically, we seeked to improve upon their method of 3D hand pose estimation by introducing
a biologically inspired loss function to further enhance the machine learning model generalization. 
Furthermore, we intended to resolve the issue of image occlusion by utilizing [FreiHAND dataset](https://github.com/lmb-freiburg/freihand):
a new public available hand dataset which contain images with hand occlusion.
More information about the implementation and result can be found in the report/CSCI5561_Final_Report_3D_Hand_Pose_Estimation.pdf.


### Installation
1. Install pytorch >= v0.4.0 following [official instruction](https://pytorch.org/).
2. Clone this repo, and we'll call the directory that you cloned as ${HAND_ROOT}.
3. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

### Pre-trained models
Pre-trained models can be found in ${HAND_ROOT}$/model/FreiHAND_BoneLoss_models.

### Running the code
#### Evaluation
1. Evaluate on FreiHAND dataset (first 200 samples are provided in ${HAND_ROOT}/data/FreiHAND_testset):
    ```
    python 6.evaluation_FreiHAND.py --config-file "configs/eval_FreiHAND_dataset.yaml"
    ```
    The visualization results will be saved to ${HAND_ROOT}/output/configs/eval_FreiHAND_dataset.yaml/

    The pose estimation results will be saved to ${HAND_ROOT}/output/configs/eval_FreiHAND_dataset.yaml/pose_estimations.mat
    
#### Training
1. Put the downloaded FreiHAND dataset in ${HAND_ROOT}$/data/

2. Change the background set (0-3, 4 sets in total) and training data size (0-32960) in ${HAND_ROOT}/configs/train_FreiHAND_dataset.yaml

3. Train the hourglass network:
    ```
    python 1.train_hg_FreiHAND_baseline2.py --config-file "configs/train_FreiHAND_dataset.yaml"
    ```

For Baseline2 model:

4. Train the MLP without bone loss:
    ```
    python 2.train_mlp_FreiHAND_baseline2.py --config-file "configs/train_FreiHAND_dataset.yaml"
    ```

5. Train the full model without bone loss:
    ```
    python 3.train_full_model_FreiHAND_baseline2.py --config-file "configs/train_FreiHAND_dataset.yaml"
    ```

For our proposed model:

4. Train the MLP with bone loss:
    ```
    python 4.train_mlp_FreiHAND_bone_loss.py --config-file "configs/train_FreiHAND_dataset.yaml"
    ```

5. Train the full model with bone loss:
    ```
    python 5.train_full_model_FreiHAND_bone_loss.py --config-file "configs/train_FreiHAND_dataset.yaml"
    ```

For each step, the trained model weights (mlp.pth and net_hm.pth) will be located at ${HAND_ROOT}$. Simply copy and paste the trained model into 
${HAND_ROOT}/model/FreiHAND_separate_trained_models before the next step.


#### Real-time hand pose estimation
1. Obtain the intrinsic camera parameter K by going through ${HAND_ROOT}$/camera_parameter_K folder.

2. Run 7.real_time_3D_handpose_estimation.py:
   ```
   python 7.real_time_3D_handpose_estimation.py --config-file "configs/eval_webcam.yaml"
   ```

### Limitation
1. The trained model only utilized 15% of the FreiHAND dataset (due to limited computing capacity). Therefore, it is possible the model can 
achieve better hand pose estimation if more data is used. 

2. We found that the hand pose estimation work best when the hand is positioned at the center of the image, and failed when the hand is not 
at the center. This can be explained by the fact that all hand images in FreiHAND dataset are positioned at the center of the images. 

3. We also found that the the model work best when the hand is at the certain depth position relative to the camera. If the hand is too close
or too far from the camera the model will failed to estimate the hand pose. The can also be explained by the FreiHAND dataset, where all the 
images contain similar hand size.
