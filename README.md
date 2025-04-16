
This repository contains materials related with article "Change Detection in Sentinel-2 Images Using Deep Learning Ensembles" Ewa Kopec, Agata M. Wijata, Jakub Nalepa.
The article has been sent to Elsevier: Remote Sensing Applications: Society and Environment (currently in second revision). 

# Implementation of UCDNet and ensemble learning script 
Fully Convolutional Neural Network model based on architecture proposed by K. S. Basavaraju - UCDNet (source https://ieeexplore.ieee.org/document/9740122). 

Ensemble learning - soft weighted voting based on results obtain from FC-EF, FC-Siam-conc, FC-Siam-diff and implemented UCDNet. 

Jupyter notebook and other models come from: 
https://github.com/rcdaudt/fully_convolutional_change_detection
[https://arxiv.org/pdf/1810.08462]

Models  such FC-EF, FC-Siam-conc and FC-Siam-diff can be downloaded and then train, test using **fully-convolutional-change-detection.ipynb**
Pretrained models for 50 epochs available here in catalogue 'Trained models'

# Brief guide 
1. OSCD dataset - official website with dataset -> https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection
2. After downloading it can be moved to GoogleDrive or it can be on local disk
3. Use this notebook and use not-trained networks from there https://github.com/rcdaudt/fully_convolutional_change_detection or use trained models
4. Before you run the notebook code, check if you have defined proper path for dataset and models
5. For UCDNet model you should add at the beginning the import:

```
# Models
from unet import Unet
from siamunet_conc import SiamUnet_conc
from siamunet_diff import SiamUnet_diff
from fresunet import FresUNet
from UCDNET import UCDNet
```
 and then 
```
# 0-RGB | 1-RGBIr | 2-All bands s.t. resulution <= 20m | 3-All bands

if TYPE == 0:
#     net, net_name = Unet(2*3, 2), 'FC-EF'
#     net, net_name = SiamUnet_conc(3, 2), 'FC-Siam-conc'
#     net, net_name = SiamUnet_diff(3, 2), 'FC-Siam-diff'
      net, net_name = UCDNet(3, 2), 'UCDNet'
#     net, net_name = FresUNet(2*3, 2), 'FresUNet'
elif TYPE == 1:
#     net, net_name = Unet(2*4, 2), 'FC-EF'
#     net, net_name = SiamUnet_conc(4, 2), 'FC-Siam-conc'
#     net, net_name = SiamUnet_diff(4, 2), 'FC-Siam-diff'
      net, net_name = UCDNet(4, 2), 'UCDNet'
#     net, net_name = FresUNet(2*4, 2), 'FresUNet'
elif TYPE == 2:
#     net, net_name = Unet(2*10, 2), 'FC-EF'
#     net, net_name = SiamUnet_conc(10, 2), 'FC-Siam-conc'
#     net, net_name = SiamUnet_diff(10, 2), 'FC-Siam-diff'
      net, net_name = UCDNet(10, 2), 'UCDNet'
#     net, net_name = FresUNet(2*10, 2), 'FresUNet'
elif TYPE == 3:
#     net, net_name = Unet(2*13, 2), 'FC-EF'
#     net, net_name = SiamUnet_conc(13, 2), 'FC-Siam-conc'
#     net, net_name = SiamUnet_diff(13, 2), 'FC-Siam-diff'
     net, net_name = UCDNet(13, 2), 'UCDNet'
#     net, net_name = FresUNet(2*13, 2), 'FresUNet'
```
# Requiremnets
- Colab Free or Pro version

# Authors

- Ewa Kopec
- Agata M. Wijata
- Jakub Nalepa

# License 
This repository is licensed under the MIT License. See LICENSE for more information.

# Contact
If you have any questions, please contact us at 
- ewakope946@student.polsl.pl
- Agata.Wijata@polsl.pl
- Jakub.Nalepa@polsl.pl


