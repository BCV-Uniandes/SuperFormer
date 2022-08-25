# SuperFormer: Volumetric Transformer Architectures for MRI Super-Resolution
This repository provides a Pytorch implementation of the paper SuperFormer: Volumetric Transformer Architectures for MRI Super-Resolution. Presented at the 7th [Simulation and Synthesis in Medical Imaging (SASHIMI) workshop](https://2022.sashimi-workshop.org/) in [MICCAI 2022](https://conferences.miccai.org/2022/en/). SuperFormer is a novel volumetric visual transformer for MRI Super-Resolution. Our method leverages the 3D and multi-domain information from volume and feature embeddings to reconstruct HR MRIs using a local self-attention mechanism. 
## Paper
SuperFormer: Volumetric Transformer Architectures for MRI Super-Resolution <br>
Cristhian Forigua $^1$ , [María Escobar](https://mc-escobar11.github.io/)$^1$ and [Pablo Arbeláez](https://scholar.google.com.co/citations?user=k0nZO90AAAAJ&hl=en)$^1$ <br>
$^1$ Center for Research and Formation in Artificial Intelligence ([CINFONIA](https://cinfonia.uniandes.edu.co/)) , Universidad de Los Andes, Bogotá, Colombia.
 
![OverviewSuperformer drawio (1)](https://user-images.githubusercontent.com/66923636/181068906-77dfbcb3-a373-4af0-9ea8-4af73a531961.png)

## Dependencies and installation 
1. Clone the repo
```
git clone https://github.com/BCV-Uniandes/SuperFormer.git
cd SuperFormer
```
2. Create environment from .yml file
```
conda env create -f environment.yml
conda activate superformer
```

## Human Connectome Project Dataset
Please refer to the [Human Connectome Project](https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release) to donwload the dataset.
Locate the files from ./split into the ./HCP folder before running the code.
### Low-resolution MRI generation
To generate the Low-resolution MRIs, we use the code in ./data/kspace.m. The factor_truncate variable controls the magnitude of subsampling the frequency domain space. Please change the "rootdir" path to your path where the data was downloaded. To run this code, you need MATLAB's "NIfTI_20140122" package.
```
./data/kspace.m
```
## Train
Training command:
```
sh train.sh
```
Make sure to change the paths to the HCP folder inside the options files. Please change the parameters "dataroot_H" and "dataroot_L". <br>
The default is to train SuperFormer from scratch. However, you can change the route of the options file inside train.sh to train either the swinIR 2D approach, 3D RRDBNet, or 3D EDSR. See the ./options folder.
## Pre-trained Model and Test
You can find our pe-trained models [here](https://drive.google.com/drive/folders/1o4p5JHO5hwfrS2G7HREKhdOZI0T7ZgWM) <br>
Before testing, make sure you change the paths of the pretrained models inside the ./options/test files. Change the attribute "pretrained_netG". Also, change the path to the HCP data.
Test 3D command:
```
sh test.sh
```
Test 2D command:
```
sh train.sh
```
## License and Acknowledgement
This project borrows heavily from [KAIR](https://github.com/cszn/KAIR), we thank the authors for their contributions to the community. <br>
More details about license in LICENSE.
## Contact
If you have any question, please email cd.forigua@uniandes.edu.co 
