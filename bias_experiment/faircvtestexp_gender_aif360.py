import pandas as pd
import numpy as np
import os
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
import random
from pathlib import Path
import tensorflow.compat.v1 as tf

#chemin

DATA_DIR = Path(os.environ.get("FAIRCV_PATH", "data")) 
fairCVBaseOriginal= "FairCVdb.npy"
fairCVbaseNeutral= "FairCVtest neutral.npy"
fairCVBaseGender = "FairCVtest gender.npy"
fairCVbaseEhnicite = "FairCVtest ethnicity.npy"

# Dossier de sortie
OUT_DIR = Path("outputs/bias_experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Chargement des données
fairCV = np.load(DATA_DIR / fairCVBaseOriginal, allow_pickle=True).item()
fairCVtestNeutral = np.load(DATA_DIR / fairCVbaseNeutral, allow_pickle=True).item()
fairCVtestGender  = np.load(DATA_DIR / fairCVBaseGender, allow_pickle=True).item()
fairCVtestEthnic  = np.load(DATA_DIR / fairCVbaseEhnicite, allow_pickle=True).item()

# Load les infos du set
profiles_test = fairCV['Profiles Test']
blind_labels_test = fairCV['Blind Labels Test']

#extraction des variales 
scores = fairCVtestNeutral['Scores']
scoresGender = fairCVtestGender['Scores']
scoresEthnicite = fairCVtestEthnic['Scores']
gender_labels = profiles_test[:,1]

# Créer les colonnes et binairisation labels 
gender_labels = profiles_test[:, 1].astype(int) #colonne 1 genre
# Utiliser la médiane ou un seuil optimisé
threshold = np.median(blind_labels_test)
labels = (blind_labels_test >= threshold).astype(int) #verite terrain ( ground truth) au dessu de la merianne bon 


def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)

results = []

for seed in range(1, 11):
    set_seeds(seed)
        
    #labels = (blind_labels_test >= 0.9).astype(int) #label en binerisant (pas bianire dans faircv et garde les meilleurs candidats 0,9)
    #scores_bin = (scores > np.percentile(scores, 90)).astype(int)  # prédiction ( en gardant 90e percentile)
    score_threshold = np.percentile(scoresGender, 100*(1 - labels.mean())) #binariser score en gardant en gardant meme proportion bons que lables 
    scores_bin = (scoresGender >= score_threshold).astype(int)

    # Créer DataFrame
    df = pd.DataFrame({
        'label': labels.ravel(), #verité terrain (ground truth )
        'gender': gender_labels.ravel(), #label protégé
    })

    # Créer BinaryLabelDataset de vérité
    dataset = BinaryLabelDataset(
        df=df,
        label_names=["label"],
        protected_attribute_names=["gender"],
        favorable_label=1,
        unfavorable_label=0
    )

    # Créer un dataset avec les prédictions
    dataset_pred = dataset.copy(deepcopy=True)
    dataset_pred.scores = scoresGender.reshape(-1, 1) #ajout des scores originaux continus
    dataset_pred.labels = scores_bin.reshape(-1, 1) #les scores binarisés
    dataset_pred.predicted_labels = scores_bin.reshape(-1, 1)

    # Groupes
    privileged_groups = [{'gender': 0}]
    unprivileged_groups = [{'gender': 1}]

    # Calcul des métriques
    metric = ClassificationMetric(
        dataset, #realite cad ground truth
        dataset_pred, #ce que modele a predit
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )

    # Affichage
    results.append({
            "seed": seed,
            "Statistical Parity Difference": metric.statistical_parity_difference(),
            "Equal Opportunity Difference": metric.equal_opportunity_difference(),
            "Error Rate Difference": metric.error_rate_difference(),
            "Average Odds Difference": metric.average_odds_difference(),
            "Disparate Impact": metric.disparate_impact(),
            "Accuracy": metric.accuracy()
    })



# === Résultats ===
df_results = pd.DataFrame(results)

print("\n=== Moyenne des 10 runs ===")
print(df_results.mean(numeric_only=True).round(4))

print("\n=== Écart-type des 10 runs ===")
print(df_results.std(numeric_only=True).round(4))

# Export CSV (nom explicite et sans écrasement)
df_results.to_csv(OUT_DIR / "FairCV_gender_bias_metrics_10runs.csv", index=False)
print(f"\nRésultats exportés : {OUT_DIR / 'FairCV_gender_bias_metrics_10runs.csv'}")