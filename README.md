# CGNet
PyTorch code for 2023 paper ["Raw Image Based Over-Exposure Correction Using Channel-Guidance Strategy"](https://ieeexplore.ieee.org/abstract/document/10239166)



# Highlights
- We present the first benchmark to explore the superiority of RAW images based on a detailed analysis in the over-exposure correction. Building upon these insights, we have developed a novel end-to-end RAW-to-sRGB network that leverages a data-driven approach to address the challenges of real-world over-exposure photography.
![image](https://github.com/whiteknight-WJN/CGNet/assets/90306495/bfd23b48-c841-4aeb-9c39-9358daca861c)

- We develop a new RAW-based non-green channel guidance strategy to maximize the utilization of useful information from red and blue channels and exploit the effective way to use it.
![image](https://github.com/whiteknight-WJN/CGNet/assets/90306495/8fee0572-9265-4b28-83eb-759688eb0827)

- We have assembled a new real-world dataset specifically designed for over-exposure correction. This dataset encompasses both RAW and sRGB images across a broad spectrum of scenes. Quantitative and qualitative results on both synthetic and real-world datasets show that our CGNet achieves state-of-the-art performance on overexposure correction.
# Prerequisites
We provide Prerequisites for reference, please refer to requirements.txt.
- matplotlib==3.5.3
- numpy==1.21.6
- opencv-python==4.7.0.72
- Pillow==9.4.0
- rawpy==0.18.0
- scipy==1.7.3
- torch==1.10.0+cu111
- torchvision==0.11.0+cu111
- tqdm==4.65.0

# Dataset
Over-exposed Raw image processing has been rarely studied due to limited available data. In order to bridge the gap of datasets and make it feasible for RAW-based end-to-end learning, we construct a large-scale RAW-based synthetic dataset mainly for model pretraining, and collect a real-world dataset from real photography that contains diverse over-exposed image pairs for training, fine-tuning and evaluation. Both of them are created in both RAW and sRGB formats and contain paired over-exposed and properly-exposed images.

## Synthetic RAW Image Dataset
To simulate realistic multi-exposure errors, we render over-exposed RAW images by multiplying reference RAW images with 4 different digital ratios of 3, 5, 8, and 10. 1595 high-quality reference original images are finally retained in our SOF dataset. Each properly-exposed reference image corresponds to 4 overexposed images of different degrees.Since the original data files are difficult to verify, the synthetic data we finally collected (processed from the MIT-Adobe FiveK dataset) totaled 3051 groups. During the test process, the program obtains the corresponding test data by reading the txt file.

For more information ,please click [here](https://github.com/whiteknight-WJN/SOF-Dataset)

The complete Synthetic RAW Image Dataset (~94.31GB) is available via link ：https://pan.baidu.com/s/14sm4ePAr2xBf442hmjmqlA 
Extraction code：kfkg 

## Real-World RAW Image Dataset
The collected Real-world Paired Over-exposure (RPO) dataset contains 650 indoor and outdoor scenarios. For each scene, we collect a sequence of overexposed RAW images with 4 pre-set over-exposure ratios to evaluate over-exposure correction methods. This yields a total of 2,600 over-exposure RAW images, with the corresponding sRGB images.

For more information ,please click [here]([https://github.com/whiteknight-WJN/SOF-Dataset](https://github.com/whiteknight-WJN/RPO-Dataset))

The complete Real-World RAW Image Dataset (~22.32GB) is available via link：https://pan.baidu.com/s/1L6Fog7X6Xd3tD_aRsyVNtg 
Extraction code：stzm 

# CItation
If you find our work helpful to your research or work, Please cite our paper.

```
@ARTICLE{10239166,
  author={Fu, Ying and Hong, Yang and Zou, Yunhao and Liu, Qiankun and Zhang, Yiming and Liu, Ning and Yan, Chenggang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Raw Image Based Over-Exposure Correction Using Channel-Guidance Strategy}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2023.3311766}}
```
# Results 
![image](https://github.com/whiteknight-WJN/CGNet/assets/90306495/d270a0da-6022-4242-81dc-bfba3ce2edec)



