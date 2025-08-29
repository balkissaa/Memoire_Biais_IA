# Bias Mitigation Methods  

This folder contains scripts implementing **bias mitigation strategies** on the **FairCVdb dataset**, using [IBM AI Fairness 360 (AIF360)](https://aif360.mybluemix.net/).  
Each method was applied to the recruitment scenario with **gender** as the protected attribute.  

## Categories  

### 1. Pre-processing  
Methods applied *before* training to reduce bias in the dataset.  

- **`preprocessing/ReweighingGender.py`**  
  Applies the **Reweighing** algorithm
  Adjusts instance weights in the training data to balance outcomes across protected groups.  
  Retrains the base hiring model with reweighted samples and evaluates fairness metrics (SPD, EOD, AOD, ERD, DI, Accuracy).  
  **Outputs:**  
  - `.npy` file with new prediction scores  
  - `.csv` file with fairness metrics across 10 random seeds  

### 2. In-processing  
Methods applied *during* training by modifying the learning algorithm.  

- **`inprocessing/AdversarialDebiasingGender.py`**  
  Implements **Adversarial Debiasing**  
  Trains a classifier while simultaneously minimizing an adversary’s ability to predict the protected attribute from the predictions.  
  Produces an “equitable” classifier whose predictions contain less discriminatory information.  
  **Outputs:**  
  - CSV with averaged fairness metrics across multiple runs  

### 3. Post-processing  
Methods applied *after* training to adjust model predictions.  

- **`postprocessing/CalibratedEqOdds.py`**  

- **`postprocessing/RejectOptionClassifier.py`**  
  Applies the **Reject Option Classification** technique  
  Relabels uncertain predictions in a way that improves fairness while maintaining accuracy.  

## Output  

For each method, the scripts generate:  
- **Fairness metrics**:  
  - Statistical Parity Difference (SPD)  
  - Equal Opportunity Difference (EOD)  
  - Average Odds Difference (AOD)  
  - Error Rate Difference (ERD)  
  - Disparate Impact (DI)  
  - Accuracy  
- **CSV files** with mean ± standard deviation across 10 runs  
- (Optional) `.npy` files with predicted scores after mitigation  
