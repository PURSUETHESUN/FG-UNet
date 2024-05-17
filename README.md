# FG-UNet: Rethinking Feature Guidance of UNet for medical image segmentation
This is an officially public repository of FG-UNet source code.
# Model Implementation
A PyTorch implementation of the FG-Unet can be found in /models directory
# Visualization Results
## Grad-CAM visualization
![Results of visualization of different blocks.](visualization/Grad_CAM_visualization.png)
## Qualitative comparison
![Results of qualitative comparison of different networks.](visualization/Qualitative_comparison.png)
# Experiments
## Recommended environment:
`conda create -n FGUNet python=3.10.9`
`conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia`
## Datasets
1. Kvasir-Seg [link](https://datasets.simula.no/kvasir-seg/).
2. ISIC2018 [link](https://challenge.isic-archive.com/data/#2018).
3. COVID-19 CT scan lesion segmentation dataset [link](https://www.kaggle.com/datasets/maedemaftouni/covid19-ct-scan-lesion-segmentation-dataset).
# Acknowledgment