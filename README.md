# LUV-Net: Multi-Pattern Lung Ultrasound Video Classification through Pattern-Specific Attention with Efficient Temporal Feature Extraction
![Image](https://github.com/user-attachments/assets/0fa3b756-3664-4aac-bcfa-573e6b846c26)
## Abstract
Lung ultrasound (LUS) has emerged as a crucial bedside imaging tool for critical care, offering advantages such as portability, cost-effectiveness, and the absence of radiation. However, LUS interpretation remains challenging due to its artifact-based nature and high dependency on operator expertise, which potentially limits its adoption in low- and middle- income countries (LMICs). To overcome these barriers, deep learning approaches offer promising solutions for LUS pattern analysis, yet existing methods have notable l imita- tions. Current approaches primarily focus on single-pattern recognition or disease-specific classification. Additionally, existing video-based models have not adequately addressed the temporal dynamics of LUS patterns, limiting their ability to capture the complex rela- tionships between consecutive frames. We propose the Lung Ultrasound Video Network (LUV-Net), a novel video-based deep learning model designed for the multi-label classifica- tion of LUS patterns. Our approach consists of two complementary modules: (i) a spatial feature extraction module utilizing pattern-specific attention mechanisms to independently process features for each LUS pattern (A-lines, B-lines, consolidation, and pleural effusion), and (ii) a temporal feature extraction module designed to capture sequential relationships between adjacent frames. We collected and labeled LUS videos from ICU patients for model development and validation, establishing two distinct datasets: a development set and a temporally separated validation set to evaluate model performance. Through 5-fold cross-validation results, our model demonstrated robust performance in identifying all four LUS patterns compared to both USVN and conventional video models. Additionally, the model’s interpretability was validated through the visualization of attention-based regions of interest corresponding to each LUS pattern.
## Dataset
![Image](https://github.com/user-attachments/assets/793b4996-15bf-4f2d-8566-31733d358afa)

## Environment & Code
- Clone this repository and navigate to the LungUS_Video folder
  ```python
  git clone https://github.com/iamhxxn2/LungUS_Video.git
  cd LungUS_Video

- Conda setting
  ```python
  conda env create -f LUV_Net.yaml
  conda activate LUV_Net

## Model train
- Main experiments
  ```python
    LUV-Net: python train.py --model_name 'LUV_Net' --num_heads 8 --kernel_size 13 --batch_size 4 --accumultation_steps 1
    LUV-Net: python train.py --model_name 'USVN' --pooling_method 'attn' --num_heads 8 --batch_size 4 --accumultation_steps 1
    LUV-Net: python train.py --model_name 'C3D' --batch_size 4 --accumultation_steps 1
    LUV-Net: python train.py --model_name 'R2Plus1D' --batch_size 4 --accumultation_steps 1

- Effectiveness of Temporal Feature Extraction Study

```python
  LUV-Net_: python train.py --model_name 'LUV-Net_' --pooling_method 'attn_multilabel_conv' --num_heads 8 --batch_size 4 --accumultation_steps 1


## Status

## Members



