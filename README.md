# CGNet
PyTorch code for 2023 paper "Raw Image Based Over-Exposure Correction Using Channel-Guidance Strategy"



# Highlights
- We present the first benchmark to explore the superiority of RAW images based on a detailed analysis in the over-exposure correction. Building upon these insights, we have developed a novel end-to-end RAW-to-sRGB network that leverages a data-driven approach to address the challenges of real-world over-exposure photography.
![image](https://github.com/whiteknight-WJN/CGNet/assets/90306495/bfd23b48-c841-4aeb-9c39-9358daca861c)

- We develop a new RAW-based non-green channel guidance strategy to maximize the utilization of useful information from red and blue channels and exploit the effective way to use it.
![image](https://github.com/whiteknight-WJN/CGNet/assets/90306495/8fee0572-9265-4b28-83eb-759688eb0827)

- We have assembled a new real-world dataset specifically designed for over-exposure correction. This dataset encompasses both RAW and sRGB images across a broad spectrum of scenes. Quantitative and qualitative results on both synthetic and real-world datasets show that our CGNet achieves state-of-the-art performance on overexposure correction.
# Prerequisites
- Python  >=  3.7
- torch  =  1.10.0+cu111

# Dataset
Over-exposed Raw image processing has been rarely studied due to limited available data. In order to bridge the gap of datasets and make it feasible for RAW-based end-to-end learning, we construct a large-scale RAW-based synthetic dataset mainly for model pretraining, and collect a real-world dataset from real photography that contains diverse over-exposed image pairs for training, fine-tuning and evaluation. Both of them are created in both RAW and sRGB formats and contain paired over-exposed and properly-exposed images.

## Synthetic RAW Image Dataset
To simulate realistic multi-exposure errors, we render over-exposed RAW images by multiplying reference RAW images with 4 different digital ratios of 3, 5, 8, and 10. 1595 high-quality reference original images are finally retained in our SOF dataset. Each properly-exposed reference image corresponds to 4 overexposed images of different degrees.

## Real-World RAW Image Dataset
The collected Real-world Paired Over-exposure (RPO) dataset contains 650 indoor and outdoor scenarios. For each scene, we collect a sequence of overexposed RAW images with 4 pre-set over-exposure ratios to evaluate over-exposure correction methods. This yields a total of 2,600 over-exposure RAW images, with the corresponding sRGB images.
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
