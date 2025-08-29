import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.postprocessing import RejectOptionClassification
from aif360.metrics import ClassificationMetric
from pathlib import Path
import os
import random
import tensorflow.compat.v1 as tf

#    Chargement des données  
fairCV_path = Path(os.environ.get("FAIRCV_PATH", "data"))
fairCVBaseOriginal = "FairCVdb.npy"
fairCVBaseNeutral = "FairCVtest neutral.npy"


fairCV_original =  np.load(str(fairCV_path / fairCVBaseOriginal), allow_pickle=True).item()
fairCV_neurtral =  np.load(str(fairCV_path / fairCVBaseNeutral), allow_pickle=True).item()

 
gender = fairCV_original['Profiles Test'][:, 1].astype(int).reshape(-1, 1)
blind_labels_test = fairCV_original['Blind Labels Test']
scores_neutral = fairCV_neurtral['Scores']

#    Binarisation de la vérité terrain  
threshold_label = np.median(blind_labels_test)
y_true = (blind_labels_test >= threshold_label).astype(int)

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)

results = []
for seed in range(1, 11):
    print(f"\n = Run avec seed {seed}  =")
    set_seeds(seed)

    #   Générer prédictions biaisées (top 10% sélectionnés)
    score_threshold = np.percentile(scores_neutral, 100 * (1 - y_true.mean()))
    y_pred = (scores_neutral >= score_threshold).astype(int)

    #    Définir groupes protégés
    privileged_groups = [{'gender': 0}]
    unprivileged_groups = [{'gender': 1}]

    #    Création des datasets AIF360  

    # Dataset avec scores et vérité terrain (servira à évaluer et corriger)
    df_pred = pd.DataFrame({
        'label': y_true.flatten(),
        'score': scores_neutral.flatten(),
        'gender': gender.flatten()
    })
    dataset_pred = BinaryLabelDataset(
        df=df_pred,
        favorable_label=1,
        unfavorable_label=0,
        label_names=['label'],
        protected_attribute_names=['gender']
    )

    # Dataset avec scores et prédictions binarisées (servira à entraîner ROC)
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
    dataset_pred.scores = scores_neutral.reshape(-1, 1).astype(np.float32)
    dataset_corrected = roc.predict(dataset_pred)

    #   Évaluer avant/après  
    metric_after = ClassificationMetric(dataset_pred, dataset_corrected, unprivileged_groups, privileged_groups)

    results.append({
            "seed": seed,
            "Statistical Parity Difference": metric_after.statistical_parity_difference(),
            "Equal Opportunity Difference": metric_after.equal_opportunity_difference(),
            "Error Rate Difference": metric_after.error_rate_difference(),
            "Average Odds Difference": metric_after.average_odds_difference(),
            "Disparate Impact": metric_after.disparate_impact(),
            "Accuracy": metric_after.accuracy()
        })

    print("\n  Fairness Metrics APRÈS ROC  ")
    print(f"Statistical Parity Difference: {metric_after.statistical_parity_difference():.4f}")
    print(f"Disparate Impact: {metric_after.disparate_impact():.4f}")
    print(f"Equal Opportunity Difference: {metric_after.equal_opportunity_difference():.4f}")
    print(f"Average Odds Difference: {metric_after.average_odds_difference():.4f}")
    print(f"Error Rate Difference: {metric_after.error_rate_difference():.4f}")
    print(f"Overall Accuracy: {metric_after.accuracy():.4f}")


#  Résumé des 10 runs  
df_results = pd.DataFrame(results)

print("\n  Moyenne des 10 runs  ")
print(df_results.mean(numeric_only=True).round(4))

print("\n  Écart-type des 10 runs  ")
print(df_results.std(numeric_only=True).round(4))



df_results.to_csv(fairCV_path / "FairCV_postprocessing_ROC_10seeds_neutral.csv", index=False)
