# FactorizePhys
FactorizePhys: Effective Spatial-Temporal Attention in Remote Photo-plethysmography through Factorization of Voxel Embeddings.

# :notebook: Algorithms
The repo currently supports the following algorithms:

* Supervised Neural Algorithms
  * FactorizePhys - proposed method
  * [Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks (PhysNet)](https://bmvc2019.org/wp-content/uploads/papers/0186-paper.pdf), by Yu *et al.*, 2019
  * [EfficientPhys: Enabling Simple, Fast and Accurate Camera-Based Cardiac Measurement (EfficientPhys)](https://openaccess.thecvf.com/content/WACV2023/papers/Liu_EfficientPhys_Enabling_Simple_Fast_and_Accurate_Camera-Based_Cardiac_Measurement_WACV_2023_paper.pdf), by Liu *et al.*, 2023
  * [PhysFormer: Facial Video-based Physiological Measurement with Temporal Difference Transformer (PhysFormer)](https://openaccess.thecvf.com/content/CVPR2022/papers/Yu_PhysFormer_Facial_Video-Based_Physiological_Measurement_With_Temporal_Difference_Transformer_CVPR_2022_paper.pdf), by Yu *et al.*, 2022

# :file_folder: Datasets

The repo supports four datasets, namely SCAMPS, UBFC-rPPG, PURE, and iBVP. **To use these datasets in a deep learning model, you should organize the files as follows.**

  * [SCAMPS](https://arxiv.org/abs/2206.04197)
    * D. McDuff, M. Wander, X. Liu, B. Hill, J. Hernandez, J. Lester, T. Baltrusaitis, "SCAMPS: Synthetics for Camera Measurement of Physiological Signals", NeurIPS, 2022
    -----------------
         data/SCAMPS/Train/
            |-- P00001.mat
            |-- P00002.mat
         |...
         data/SCAMPS/Val/
            |-- P00001.mat
            |-- P00002.mat
         |...
         data/SCAMPS/Test/
            |-- P00001.mat
            |-- P00002.mat
         |...
    -----------------

  * [UBFC-rPPG](https://sites.google.com/view/ybenezeth/ubfcrppg)
    * S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois, "Unsupervised skin tissue segmentation for remote photoplethysmography", Pattern Recognition Letters, 2017.
    -----------------
         data/UBFC-rPPG/
         |   |-- subject1/
         |       |-- vid.avi
         |       |-- ground_truth.txt
         |   |-- subject2/
         |       |-- vid.avi
         |       |-- ground_truth.txt
         |...
         |   |-- subjectn/
         |       |-- vid.avi
         |       |-- ground_truth.txt
    -----------------

  * [PURE](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure)
    * Stricker, R., Müller, S., Gross, H.-M.Non-contact "Video-based Pulse Rate Measurement on a Mobile Service Robot" in: Proc. 23st IEEE Int. Symposium on Robot and Human Interactive Communication (Ro-Man 2014), Edinburgh, Scotland, UK, pp. 1056 - 1062, IEEE 2014
    -----------------
         data/PURE/
         |   |-- 01-01/
         |      |-- 01-01/
         |      |-- 01-01.json
         |   |-- 01-02/
         |      |-- 01-02/
         |      |-- 01-02.json
         |...
         |   |-- ii-jj/
         |      |-- ii-jj/
         |      |-- ii-jj.json
    -----------------

  * [iBVP](https://github.com/PhysiologicAILab/iBVP-Dataset)
    * Joshi, J.; Cho, Y. iBVP Dataset: RGB-Thermal rPPG Dataset with High Resolution Signal Quality Labels. Electronics 2024, 13, 1334.
    -----------------
          iBVP_Dataset/
          |   |-- p01_a/
          |      |-- p01_a_rgb/
          |      |-- p01_a_t/
          |      |-- p01_a_bvp.csv
          |   |-- p01_b/
          |      |-- p01_b_rgb/
          |      |-- p01_b_t/
          |      |-- p01_b_bvp.csv
          |...
          |   |-- pii_x/
          |      |-- pii_x_rgb/
          |      |-- pii_x_t/
          |      |-- pii_x_bvp.csv
    -----------------

# :wrench: Setup

STEP 1: `bash setup.sh`

STEP 2: `conda activate fsam`

STEP 3: `pip install -r requirements.txt`

# :computer: Example of Using Pre-trained Models

Please use config files under `./configs/infer_configs`

For example, if you want to run The model trained on PURE and tested on UBFC-rPPG, use `python main.py --config_file configs/infer_configs/PURE_UBFC-rPPG_FactorizePhys_FSAM_Res.yaml`

# :computer: Examples of Neural Network Training

Please use config files under `./configs/train_configs`

## Training on PURE and Testing on iBVP With FactorizePhys

STEP 1: Download the PURE raw data by asking the [paper authors](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure).

STEP 2: Download the iBVP raw data by asking the [paper authors](https://github.com/PhysiologicAILab/iBVP-Dataset).

STEP 3: Modify `configs/train_configs/PURE_iBVP_FactorizePhys_FSAM_Res.yaml`

STEP 4: Run `python main.py --config_file configs/train_configs/PURE_iBVP_FactorizePhys_FSAM_Res.yaml`

Note 1: Preprocessing requires only once; thus turn it off on the yaml file when you train the network after the first time.

Note 2: The example yaml setting will allow 100% of PURE to train and and test on iBVP after training 10 for epochs. Alternatively, this can be changed to train using 80% of PURE, validate with 20% of PURE and use the best model(with the least validation loss) to test on iBVP.

## Cross-Dataset Generalization
|              Training   Dataset             |         Model        | Attention      Module |    MAE (HR) ↓   |    RMSE (HR) ↓   |    MAPE (HR)↓    |   Corr (HR) ↑   | SNR ( dB, BVP) ↑ |   MACC (BVP) ↑  |
|:-------------------------------------------:|:--------------------:|:---------------------:|:---------------:|:----------------:|:----------------:|:---------------:|:----------------:|:---------------:|
|                                             |                      |                       |                 |                  |                  |                 |                  |                 |
|   **Performance Evaluation on PURE**        |                      |                       |                 |                  |                  |                 |                  |                 |
|                                             |                      |                       |                 |                  |                  |                 |                  |                 |
|                     iBVP                    | PhysNet              | -                     |   7.78 ± 2.27   |   19.12 ± 3.93   |    8.94 ± 2.71   |   0.59 ± 0.11   |    9.90 ± 1.49   |   0.70 ± 0.03   |
|                                             | PhysFormer           | TD-MHSA*              |   6.58 ± 1.98   |   16.55 ± 3.60   |    6.93 ± 1.90   |   0.76 ± 0.09   |    9.75 ± 1.96   |   0.71 ± 0.03   |
|                                             | EfficientPhys        | SASN                  |   0.56 ± 0.17   |    1.40 ± 0.33   |    0.87 ± 0.28   | **1.00 ± 0.01** |   11.96 ± 0.84   |   0.73 ± 0.02   |
|                                             | EfficientPhys        | FSAM (Ours)           | **0.44 ± 0.14** |  **1.19 ± 0.30** |  **0.64 ± 0.22** | **1.00 ± 0.01** |   12.64 ± 0.78   |   0.75 ± 0.02   |
|                                             | FactorizePhys (Ours) | FSAM (Ours)           |   0.60 ± 0.21   |    1.70 ± 0.42   |    0.87 ± 0.30   | **1.00 ± 0.01** | **15.19 ± 0.91** | **0.77 ± 0.02** |
|                                             |                      |                       |                 |                  |                  |                 |                  |                 |
|                    SCAMPS                   | PhysNet              | -                     |   26.74 ± 3.17  |   36.19 ± 5.18   |   46.73 ± 5.66   |   0.45 ± 0.12   |   -2.21 ± 0.66   |   0.31 ± 0.02   |
|                                             | PhysFormer           | TD-MHSA*              |   16.64 ± 2.95  |   28.13 ± 5.00   |   30.58 ± 5.72   |   0.51 ± 0.11   |    0.84 ± 1.00   |   0.42 ± 0.02   |
|                                             | EfficientPhys        | SASN                  |   6.21 ± 2.26   |   18.45 ± 4.54   |   12.16 ± 4.57   |   0.74 ± 0.09   |    4.39 ± 0.78   |   0.51 ± 0.02   |
|                                             | EfficientPhys        | FSAM (Ours)           |   8.03 ± 2.25   |   19.09 ± 4.27   |   15.12 ± 4.44   |   0.73 ± 0.09   |    3.81 ± 0.79   |   0.48 ± 0.02   |
|                                             | FactorizePhys (Ours) | FSAM (Ours)           | **5.43 ± 1.93** | **15.80 ± 3.58** | **11.10 ± 4.05** | **0.80 ± 0.08** | **11.40 ± 0.76** | **0.67 ± 0.02** |
|                                             |                      |                       |                 |                  |                  |                 |                  |                 |
|                  UBFC-rPPG                  | PhysNet              | -                     |   10.38 ± 2.40  |   21.14 ± 3.90   |   20.91 ± 4.97   |   0.66 ± 0.10   |   11.01 ± 0.97   |   0.72 ± 0.02   |
|                                             | PhysFormer           | TD-MHSA*              |   8.90 ± 2.15   |   18.77 ± 3.67   |   17.68 ± 4.52   |   0.71 ± 0.09   |    8.73 ± 1.02   |   0.66 ± 0.02   |
|                                             | EfficientPhys        | SASN                  |   4.71 ± 1.79   |   14.52 ± 3.65   |    7.63 ± 2.97   |   0.80 ± 0.08   |    8.77 ± 1.00   |   0.66 ± 0.02   |
|                                             | EfficientPhys        | FSAM (Ours)           |   3.69 ± 1.66   |   13.27 ± 3.55   |    5.85 ± 2.63   |   0.83 ± 0.07   |    9.65 ± 0.90   |   0.68 ± 0.02   |
|                                             | FactorizePhys (Ours) | FSAM (Ours)           | **0.48 ± 0.17** |  **1.39 ± 0.35** |  **0.72 ± 0.28** | **1.00 ± 0.01** | **14.16 ± 0.83** | **0.78 ± 0.02** |
|                                             |                      |                       |                 |                  |                  |                 |                  |                 |
|  **Performance Evaluation on UBFC-rPPG**    |                      |                       |                 |                  |                  |                 |                  |                 |
|                                             |                      |                       |                 |                  |                  |                 |                  |                 |
|                     iBVP                    | PhysNet              | -                     |   3.09 ± 1.79   |   10.72 ± 4.24   |    2.83 ± 1.44   |   0.81 ± 0.10   |    7.13 ± 1.53   |   0.81 ± 0.02   |
|                                             | PhysFormer           | TD-MHSA*              |   9.88 ± 2.95   |   19.59 ± 5.35   |    8.72 ± 2.42   |   0.44 ± 0.16   |    2.80 ± 2.21   |   0.70 ± 0.03   |
|                                             | EfficientPhys        | SASN                  |   1.14 ± 0.45   |    2.85 ± 0.88   |    1.42 ± 0.58   | **0.99 ± 0.03** |    8.71 ± 1.23   |   0.84 ± 0.01   |
|                                             | EfficientPhys        | FSAM (Ours)           |   1.17 ± 0.46   |    2.87 ± 0.88   |    1.31 ± 0.53   | **0.99 ± 0.03** |    8.54 ± 1.26   |   0.85 ± 0.01   |
|                                             | FactorizePhys (Ours) | FSAM (Ours)           | **1.04 ± 0.38** |  **2.40 ± 0.69** |  **1.23 ± 0.48** | **0.99 ± 0.03** |  **8.84 ± 1.31** | **0.86 ± 0.01** |
|                                             |                      |                       |                 |                  |                  |                 |                  |                 |
|                     PURE                    | PhysNet              | -                     |   1.23 ± 0.41   |    2.65 ± 0.70   |    1.42 ± 0.50   | **0.99 ± 0.03** |    8.34 ± 1.22   |   0.85 ± 0.01   |
|                                             | PhysFormer           | TD-MHSA*              | **1.01 ± 0.38** |  **2.40 ± 0.69** |  **1.21 ± 0.48** | **0.99 ± 0.03** |    8.42 ± 1.24   |   0.85 ± 0.01   |
|                                             | EfficientPhys        | SASN                  |   1.41 ± 0.49   |    3.16 ± 0.93   |    1.68 ± 0.64   |   0.98 ± 0.03   |    6.87 ± 1.15   |   0.79 ± 0.02   |
|                                             | EfficientPhys        | FSAM (Ours)           |   1.20 ± 0.46   |    2.92 ± 0.92   |    1.50 ± 0.63   | **0.99 ± 0.03** |    7.37 ± 1.20   |   0.79 ± 0.01   |
|                                             | FactorizePhys (Ours) | FSAM (Ours)           |   1.04 ± 0.38   |    2.44 ± 0.69   |    1.23 ± 0.48   | **0.99 ± 0.03** |  **8.88 ± 1.30** | **0.87 ± 0.01** |
|                                             |                      |                       |                 |                  |                  |                 |                  |                 |
|                    SCAMPS                   | PhysNet              | -                     |   11.24 ± 2.63  |   18.81 ± 4.71   |   13.55 ± 3.81   |   0.38 ± 0.17   |   -0.09 ± 1.02   |   0.48 ± 0.03   |
|                                             | PhysFormer           | TD-MHSA*              |   8.42 ± 2.72   |   17.73 ± 5.09   |   11.27 ± 4.24   |   0.49 ± 0.16   |    2.29 ± 1.33   |   0.61 ± 0.03   |
|                                             | EfficientPhys        | SASN                  |   2.18 ± 0.75   |    4.82 ± 1.43   |    2.35 ± 0.76   |   0.96 ± 0.05   |    4.40 ± 1.03   |   0.67 ± 0.01   |
|                                             | EfficientPhys        | FSAM (Ours)           |   2.69 ± 0.77   |    5.20 ± 1.39   |    3.16 ± 0.95   |   0.95 ± 0.06   |    3.74 ± 1.16   |   0.63 ± 0.02   |
|                                             | FactorizePhys (Ours) | FSAM (Ours)           | **1.17 ± 0.40** |  **2.56 ± 0.70** |  **1.35 ± 0.49** | **0.99 ± 0.03** |  **8.41 ± 1.19** | **0.82 ± 0.01** |
|                                             |                      |                       |                 |                  |                  |                 |                  |                 |
|   **Performance Evaluation on iBVP**        |                      |                       |                 |                  |                  |                 |                  |                 |
|                                             |                      |                       |                 |                  |                  |                 |                  |                 |
|                     PURE                    | PhysNet              | -                     | **1.63 ± 0.33** |    3.77 ± 0.73   |  **2.17 ± 0.42** |   0.92 ± 0.04   |    6.08 ± 0.62   |   0.55 ± 0.01   |
|                                             | PhysFormer           | TD-MHSA*              |   2.50 ± 0.64   |    7.09 ± 1.50   |    3.39 ± 0.82   |   0.79 ± 0.06   |    5.21 ± 0.60   |   0.52 ± 0.01   |
|                                             | EfficientPhys        | SASN                  |   3.80 ± 1.38   |   14.82 ± 3.74   |    5.15 ± 1.87   |   0.56 ± 0.08   |    2.93 ± 0.48   |   0.45 ± 0.01   |
|                                             | EfficientPhys        | FSAM (Ours)           |   2.10 ± 0.33   |    4.00 ± 0.64   |    2.94 ± 0.49   |   0.91 ± 0.04   |    4.19 ± 0.54   |   0.49 ± 0.01   |
|                                             | FactorizePhys (Ours) | FSAM (Ours)           |   1.66 ± 0.30   |  **3.55 ± 0.65** |    2.31 ± 0.46   | **0.93 ± 0.04** |  **6.78 ± 0.57** | **0.58 ± 0.01** |
|                                             |                      |                       |                 |                  |                  |                 |                  |                 |
|                    SCAMPS                   | PhysNet              | -                     |   31.85 ± 1.89  |   37.40 ± 3.38   |   45.62 ± 2.96   |   -0.10 ± 0.10  |   -6.11 ± 0.22   |   0.16 ± 0.00   |
|                                             | PhysFormer           | TD-MHSA*              |   41.73 ± 1.31  |   43.89 ± 3.11   |   58.56 ± 2.36   |   0.15 ± 0.10   |   -9.13 ± 0.53   |   0.14 ± 0.00   |
|                                             | EfficientPhys        | SASN                  |   26.19 ± 3.47  |   44.55 ± 6.18   |   38.11 ± 5.21   |   -0.12 ± 0.10  |   -2.36 ± 0.38   |   0.30 ± 0.01   |
|                                             | EfficientPhys        | FSAM (Ours)           |   13.40 ± 1.69  |   22.10 ± 3.00   |   19.93 ± 2.67   |   0.05 ± 0.10   |   -3.46 ± 0.26   |   0.24 ± 0.01   |
|                                             | FactorizePhys (Ours) | FSAM (Ours)           | **2.71 ± 0.54** |  **6.22 ± 1.38** |  **3.87 ± 0.80** | **0.81 ± 0.06** |  **2.36 ± 0.47** | **0.43 ± 0.01** |
|                                             |                      |                       |                 |                  |                  |                 |                  |                 |
|                  UBFC-rPPG                  | PhysNet              | -                     |   3.18 ± 0.67   |    7.65 ± 1.46   |    4.84 ± 1.14   |   0.70 ± 0.07   |    5.54 ± 0.61   | **0.56 ± 0.01** |
|                                             | PhysFormer           | TD-MHSA*              |   7.86 ± 1.46   |   17.13 ± 2.69   |   11.44 ± 2.25   |   0.38 ± 0.09   |    1.71 ± 0.56   |   0.43 ± 0.01   |
|                                             | EfficientPhys        | SASN                  |   2.74 ± 0.63   |    7.07 ± 1.81   |    4.02 ± 1.08   |   0.74 ± 0.07   |    4.03 ± 0.55   |   0.49 ± 0.01   |
|                                             | EfficientPhys        | FSAM (Ours)           |   2.56 ± 0.54   |    6.13 ± 1.32   |    3.71 ± 0.92   |   0.79 ± 0.06   |    4.65 ± 0.56   |   0.50 ± 0.01   |
|                                             | FactorizePhys (Ours) | FSAM (Ours)           | **1.74 ± 0.39** |  **4.39 ± 1.06** |  **2.42 ± 0.57** | **0.90 ± 0.04** |  **6.59 ± 0.57** | **0.56 ± 0.01** |
|                                             |                      |                       |                 |                  |                  |                 |                  |                 |


## Reference:
This repo builds upon [rPPG-toolbox](https://github.com/ubicomplab/rPPG-Toolbox), that can be referred further for usage related instructions.
