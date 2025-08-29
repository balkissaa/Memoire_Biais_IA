import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.postprocessing import RejectOptionClassification
from aif360.metrics import ClassificationMetric
import os
from pathlib import Path


#   Chargement des données  
fairCV_path = Path(os.environ.get("FAIRCV_PATH", "data"))
fairCVBaseOriginal = "FairCVdb.npy"
fairCVBaseGender = "FairCVtest gender.npy"
faircvBaseproxyGenderproxy07 = "FairCVtest proxy0.7.npy"

fairCV_original = np.load(str(fairCV_path / fairCVBaseOriginal), allow_pickle=True).item()
fairCV_gender = np.load(str(fairCV_path / fairCVBaseGender), allow_pickle=True).item()
fairCV_genderproxy07  = np.load(str(fairCV_path / faircvBaseproxyGenderproxy07), allow_pickle=True).item()

#    Variables utiles  
gender = fairCV_genderproxy07['Gender'].reshape(-1, 1)
blind_labels_test = fairCV_genderproxy07['Blind Labels Test']
scores_biased = fairCV_genderproxy07['Scores']

#   Binarisation de la vérité terrain  
threshold_label = np.median(blind_labels_test)
y_true = (blind_labels_test >= threshold_label).astype(int)

#   Générer prédictions biaisées (top 10% sélectionnés)
score_threshold = np.percentile(scores_biased, 100 * (1 - y_true.mean()))
y_pred = (scores_biased >= score_threshold).astype(int)

#    Définir groupes protégés
privileged_groups = [{'gender': 0}]
unprivileged_groups = [{'gender': 1}]

#    Création des datasets AIF360  

# Dataset avec scores et vérité terrain (servira à évaluer et corriger)
df_pred = pd.DataFrame({
    'label': y_true.flatten(),
    'score': scores_biased.flatten(),
    'gender': gender.flatten()
})
dataset_pred = BinaryLabelDataset(
    df=df_pred,
    favorable_label=1,
    unfavorable_label=0,
    label_names=['label'],
    protected_attribute_names=['gender']
)

# Dataset avec scores et prédictions biaisées (servira à entraîner ROC)
df_biased = df_pred.copy()
df_biased['label'] = y_pred.flatten()
dataset_biased = BinaryLabelDataset(
    df=df_biased,
    favorable_label=1,
    unfavorable_label=0,
    label_names=['label'],
    protected_attribute_names=['gender']
)

#    Appliquer ROC  
roc = RejectOptionClassification(
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups,
    low_class_thresh=0.2,
    high_class_thresh=0.8,
    num_class_thresh=100,
    num_ROC_margin=50,
    metric_name="Statistical parity difference",
    metric_lb=-0.05,
    metric_ub=0.05
)

roc.fit(dataset_pred,dataset_biased)
dataset_pred.scores = scores_biased.reshape(-1, 1).astype(np.float32)
dataset_corrected = roc.predict(dataset_pred)

#    Évaluer avant/après  
metric_before = ClassificationMetric(dataset_pred, dataset_biased, unprivileged_groups, privileged_groups)
metric_after = ClassificationMetric(dataset_pred, dataset_corrected, unprivileged_groups, privileged_groups)

#    Affichage  
print("\n--- Taux de sélection AVANT ROC proxy 0,7 ---")
print(df_biased.groupby('gender')['label'].mean())

print("\n  Fairness Metrics AVANT ROC ===")
print(f"Statistical Parity Difference: {metric_before.statistical_parity_difference():.4f}")
print(f"Disparate Impact: {metric_before.disparate_impact():.4f}")
print(f"Equal Opportunity Difference: {metric_before.equal_opportunity_difference():.4f}")
print(f"Average Odds Difference: {metric_before.average_odds_difference():.4f}")
print(f"Error Rate Difference: {metric_before.error_rate_difference():.4f}")
print(f"Overall Accuracy: {metric_before.accuracy():.4f}")

print("\n=== Fairness Metrics APRÈS ROC ===")
print(f"Statistical Parity Difference: {metric_after.statistical_parity_difference():.4f}")
print(f"Disparate Impact: {metric_after.disparate_impact():.4f}")
print(f"Equal Opportunity Difference: {metric_after.equal_opportunity_difference():.4f}")
print(f"Average Odds Difference: {metric_after.average_odds_difference():.4f}")
print(f"Error Rate Difference: {metric_after.error_rate_difference():.4f}")
print(f"Overall Accuracy: {metric_after.accuracy():.4f}")

#    Vérifications  
print("\nSame values in y_pred and y_true?:", np.array_equal(y_pred.reshape(-1, 1), y_true.reshape(-1, 1)))
print("Proportion bons candidats (y_true):", y_true.mean())
print("Proportion sélectionnés (y_pred):", y_pred.mean())
