# 3D Residual U-Net for Aortic Dissection Segmentation
Partial codebase of framework used for lumen segmentation in Type-B aortic dissection patients, utilizing a 3D residual symmetric U-net. Segmentation of three key labels is supported: background (BG), true lumen (TL), and false lumen (FL). Additionally, the framework offers the option to include a fourth label, false lumen thrombosis (FLT). The input data consists of computed tomography angiography (CTA) scans and their corresponding ground truth mask. The methodology is built upon the foundational work found in the [pytorch-3dunet repository](https://github.com/wolny/pytorch-3dunet). 

My work led to a publication on deep learning-based segmentation in aortic dissection ([Wobben et al., 2019](https://pubmed.ncbi.nlm.nih.gov/34892087/)). Note that, due to constraints, only the code for one of the three models (model 1, single-step multi-task) could (partially) be made publicly available. Nevertheless, this repository offers insights into the methodologies explored during my master's research and can hopefully serve as a source of inspiration.

## Usage instructions
1. Set up your environment with the packages from `requirements.txt`
2. Preprocess your data
    - Output format: .npy arrays
    - See `preprocessing.py` for an example script
3. Define all constants in Constants.py
4. Run `python3 -m visdom.server` for live imaging on Visdom
    - Follow instructions and access localhost
5. Run `python3 main.py` for training
6. Run `python3 test.py` for testing and saving predictions
