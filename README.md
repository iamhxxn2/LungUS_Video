# LUV-Net: Multi-Pattern Lung Ultrasound Video Classification through Pattern-Specific Attention with Efficient Temporal Feature Extraction
![Image](https://github.com/user-attachments/assets/7716403c-8404-424f-ac1c-d82fe6ce3d4d)
## 1. Abstract
Lung ultrasound (LUS) has emerged as a crucial bedside imaging tool for critical care, offering advantages such as portability, cost-effectiveness, and the absence of radiation. However, LUS interpretation remains challenging due to its artifact-based nature and high dependency on operator expertise, which potentially limits its adoption in low- and middle- income countries (LMICs). To overcome these barriers, deep learning approaches offer promising solutions for LUS pattern analysis, yet existing methods have notable l imita- tions. Current approaches primarily focus on single-pattern recognition or disease-specific classification. Additionally, existing video-based models have not adequately addressed the temporal dynamics of LUS patterns, limiting their ability to capture the complex rela- tionships between consecutive frames. We propose the Lung Ultrasound Video Network (LUV-Net), a novel video-based deep learning model designed for the multi-label classifica- tion of LUS patterns. Our approach consists of two complementary modules: (i) a spatial feature extraction module utilizing pattern-specific attention mechanisms to independently process features for each LUS pattern (A-lines, B-lines, consolidation, and pleural effusion), and (ii) a temporal feature extraction module designed to capture sequential relationships between adjacent frames. We collected and labeled LUS videos from ICU patients for model development and validation, establishing two distinct datasets: a development set and a temporally separated validation set to evaluate model performance. Through 5-fold cross-validation results, our model demonstrated robust performance in identifying all four LUS patterns compared to both USVN and conventional video models. Additionally, the model’s interpretability was validated through the visualization of attention-based regions of interest corresponding to each LUS pattern.

## 2. Environment & Code
* Clone this repository and navigate to the LungUS_Video folder
  ```
  git clone https://github.com/iamhxxn2/LungUS_Video.git
  cd LungUS_Video
  ```
* Conda setting
  ```
  conda env create -f LUV_Net.yaml
  conda activate LUV_Net
  ```

## 3. Dataset

### 3.1. Additional Details on Dataset Structure & Annotation
#### 3.1.1. CSV Files

**Model Development Set**
- Original CSV: `original_labeling_sheet_20240216`
- **Version 1**
  - Labels finalized using **AND** logic: only regions where **both annotators agreed** are retained.
  - This version was used for model training and evaluation.
  - Path:
    ```text
    D:\Research\LUS\Dataset\csv_files\version1_v1.csv
    ```
- **Version 2**
  - Labels finalized using **OR** logic: union of labels from both annotators.
  - Path:
    ```text
    D:\Research\LUS\Dataset\csv_files\version2_v1.csv
    ```
- **5-Fold Validation Dataset**
  - Path:  
    ```text
    D:\Research\LUS\Dataset\csv_files\clip_multilabel_classification\model_development_set\version_1\5_artifacts\test_0.2
    ```
  - Contains split CSV files (`train`, `validation`, `test`) for clip-level classification after preprocessing.

**Temporally Separated Test Set**
- Original CSV Files:
  - Senior clinician (Jinwoo Lee): `Temporally_separated_sheet`
  - Two additional experts: `Temporally_separated_sheet_labeler1`, `Temporally_separated_sheet_labeler2`
- **Test Dataset CSV**
  - Path:  
    ```text
    D:\Research\LUS\Dataset\csv_files\clip_multilabel_classification\temporally_separated_test_set\clip\5_artifacts\temporally_separated_test.csv
    ```
  - Used for evaluating model generalizability (preprocessed clips only)

---

#### 3.1.2. Dataset Directory Structure

**Original DICOM Data**
- Raw files organized by acquisition date
- Path:  
    ```text
    D:\Research\LUS\Dataset\original_dataset
    ```
  *(includes both development and temporally separated sets)*

**Preprocessed DICOM (Anonymized)**
- Development set
    ```text
    D:\Research\LUS\Dataset\processed_dataset_dcm
    ``` 
- Temporally separated set
  ```text
    D:\Research\LUS\Dataset\processed_temporally_separated_dataset_dcm
  ```

**DICOM → AVI Conversion**
- Development set
  ```text
    D:\Research\LUS\Dataset\processed_dataset_avi
  ```  
- Temporally separated set
  ```text
    D:\Research\LUS\Dataset\processed_temporally_separated_dataset_avi
  ``` 

**Frame-Level PNG Extraction**
- Development set
  ```text
    D:\Research\LUS\Dataset\development_dcm_to_png
  ```  
- Temporally separated set
  ```text
    D:\Research\LUS\Dataset\temporally_separated_dcm_to_png
  ```
  
**Video to Clip (1s with 20% overlap)**
- Development set
  ```text
    D:\Research\LUS\Dataset\clip_avi_dataset
  ```
- Temporally separated set
  ```text
    D:\Research\LUS\Dataset\clip_avi_temporally_separated_dataset
  ```

### 3.2. Data Collection & Overview
![Image](https://github.com/user-attachments/assets/793b4996-15bf-4f2d-8566-31733d358afa)
We constructed two datasets from LUS videos acquired from ICU patients at Seoul National University Hospital (SNUH). Each video was annotated for four LUS patterns (A-lines, B-lines, consolidation, and pleural effusion) at the frame level. The development set was double-annotated with consensus adjudication, while the test set was labeled by a senior clinician.
* Model Development Set
   * Total raw videos collected: 370 videos from 36 patients
   * Final videos used after exclusion: 341 videos from 35 patients
   * Videos were acquired from anterior, lateral, and posterior lung zones using a convex probe (3–5 MHz), following the BLUE protocol.
   * Each video lasted approximately 5~8 seconds at 30 fps.

* Temporally Separated Test Set
   * Total videos collected: 56 videos from 11 patients
   * Used for external validation to assess model generalizability.

### 3.3. Preprocessing
We segmented each raw LUS video (5–8 seconds) into multiple 1-second clips (30 frames) using a 20% overlap (6 frames) strategy.
* Development Set: From 341 videos → 2,588 clips generated
* Temporally separated Set: From 56 videos → 366 clips generated
  
All frames were resized to 256×256 resolution. A strict patient-level split was used:
* 10% patients held out for internal test set
* 5-fold cross-validation conducted on the remaining 90%, ensuring no patient overlap across folds
  
**make CSV file from dicom information**
```
0_dicom_preprocessing_make_csv.ipynb
```
**DICOM → AVI Conversion**
* Dicom to Video (Model development set & Temporally separated set)
```
1_Development_data_dcm2avi.ipynb
1_Temporally_separated_data_dcm2avi.ipynb
```

**Frame-Level PNG Extraction**
```
2_Development_data_dcm2png.ipynb
2_Temporally_separated_data_dcm2png.ipynb
```

**Video to Clip (1s with 20% overlap)**
```
3_Development_data_preprocessing.ipynb
3_Temporally_separated_data_preprocessing.ipynb
```

## 4. LUV-Net Model train & Validation
### 4.1. Train
```
LUV-Net: python train_LUV_Net.py --model_name 'LUV_Net' --pooling_method 'attn_multilabel_conv' --num_heads 8 --kernel_size 13 --batch_size 4 --accumultation_steps 1
```
  
### 4.2. Effectiveness of Temporal Feature Extraction Study
```
LUV-Net: python train_LUV_Net.py --model_name 'LUV_Net' --pooling_method 'attn_multilabel_conv' --num_heads 8 --kernel_size 13 --batch_size 4 --accumultation_steps 1
LUV-Net_: python train.py --model_name 'LUV_Net_TFWO' --pooling_method 'attn_multilabel_conv' --num_heads 8 --batch_size 4 --accumultation_steps 1
```

### 4.3. Evaluation (Model development set / Temporally separated set)
```
LUV_Net_model_development_test.ipynb
LUV_Net_temporally_separated_test.ipynb
```

## 5. Baseline Models train & Validation
Baseline Models: USVN, C3D, R2Plus1D, CNN-LSTM, ViViT, MViT
 
### 5.1. Train
```
python train_baseline.py --model_name 'Model_name' --pooling_method 'attn' --num_heads 8 --batch_size 4 --accumultation_steps 1
```

### 5.2. Evaluation (Model development set / Temporally separated set)
```
baselines_model_development_test.ipynb
baselines_temporally_separated_test.ipynb
```

## 6. Attention graph (Model development set / Temporally separated set)
```
attention_graphs3_Internal.ipynb
attention_graphs3_TP.ipynb
```

## 7. Analysis

### 7.1. DeLong Test Implementation
```
delong_test.ipynb
```

### 7.2. Model parameter size / GFLOPS computation & Inference Time 

This analysis is included in the "**LUV_Net_model_development_test.ipynb**".

## Status
The model and results described in this repository have been accepted as a **full paper** at the **Medical Imaging with Deep Learning (MIDL) 2025** conference.



