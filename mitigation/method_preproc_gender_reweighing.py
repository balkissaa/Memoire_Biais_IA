import numpy as np
import os
import pickle
from pathlib import Path
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from aif360.metrics import ClassificationMetric
import pandas as pd
import random
import tensorflow as tf


# CONFIGURATION 
DATA_PATH = Path(os.environ.get("FAIRCV_PATH", "data"))  # changer via env 
DATABASE_FILE = "FairCVdb.npy"
TOKENIZER_FILE = DATA_PATH / "Tokenizer gender.pickle"
MODEL_BASE_FILE = DATA_PATH / "HiringTool gender.h5"

OUT_DIR = Path("outputs/preprocessing")
OUT_DIR.mkdir(parents=True, exist_ok=True)

WORD_EMB_FILE = 'crawl-300d-2M.vec'  # Pas utilisé ici directement, car le modèle est déjà entraîné
CONFIG = 'gender'
MAX_WORDS = 20000

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)  # Pour hash Python
    np.random.seed(seed)   #Pour NumPy
    random.seed(seed) #Pour random
    tf.random.set_seed(seed) #Pour TensorFlow



# Load dataset from gender configuration 
def load_gender_dataset():
    fairCV = np.load(DATA_PATH / DATABASE_FILE, allow_pickle=True).item()
    profiles_train = fairCV['Profiles Train'][:, 4:31]
    profiles_test = fairCV['Profiles Test'][:, 4:31]

    gender_train = fairCV['Profiles Train'][:, 1].astype(int)
    gender_test = fairCV['Profiles Test'][:, 1].astype(int)

    labels_train = fairCV['Biased Labels Train (Gender)']
    labels_test = fairCV['Blind Labels Test']

    return profiles_train, profiles_test, gender_train, gender_test, labels_train, labels_test


#  Load tokenizer and bios 
def load_bios_and_tokenizer():
    fairCV = np.load(DATA_PATH / DATABASE_FILE, allow_pickle=True).item()
    bios_train = fairCV['Bios Train'][:, 0]
    bios_test = fairCV['Bios Test'][:, 0]

    with open(TOKENIZER_FILE, 'rb') as f:
        tokenizer = pickle.load(f)

    return bios_train, bios_test, tokenizer


#  Tokenize bios 
def process_bios(bios_raw, tokenizer, max_len):
    sequences = tokenizer.texts_to_sequences(bios_raw)
    bios = pad_sequences(sequences, maxlen=max_len)
    return bios


#  Prepare AIF360 dataset and apply reweighing 
def apply_reweighing(X_train, y_train, gender_train):
    import pandas as pd
    df = pd.DataFrame(X_train)
    df['label'] = y_train
    df['gender'] = gender_train
    dataset = BinaryLabelDataset(
        df=df,
        label_names=['label'],
        protected_attribute_names=['gender'],
        favorable_label=1,
        unfavorable_label=0
    )

    privileged_groups = [{'gender': 0}]
    unprivileged_groups = [{'gender': 1}]
    RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    dataset_transf = RW.fit_transform(dataset)
    return dataset_transf.instance_weights


#  Train new model using reweighted samples 
def train_reweighed_model(model, X_train_structured, X_train_bios, y_train, sample_weights):
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanAbsoluteError())
    model.fit([X_train_structured, X_train_bios], y_train,
              sample_weight=sample_weights,
              batch_size=128, epochs=16)
    return model


#  Evaluate and return scores 
def evaluate_model(model, X_test_structured, X_test_bios):
    scores = model.predict([X_test_structured, X_test_bios])
    return scores


# RUN EVERYTHING 
profiles_train, profiles_test, gender_train, gender_test, labels_train_raw, labels_test_raw = load_gender_dataset()
bios_train_raw, bios_test_raw, tokenizer = load_bios_and_tokenizer()

# Binarize labels
threshold = np.median(labels_train_raw)
y_train = (labels_train_raw >= threshold).astype(int)
y_test = (labels_test_raw >= threshold).astype(int)

# Load tokenizer and model length
model = load_model(MODEL_BASE_FILE)
max_len = model.input[1].shape[1]

# Process bios
bios_train = process_bios(bios_train_raw, tokenizer, max_len)
bios_test = process_bios(bios_test_raw, tokenizer, max_len)



results = []
for seed in range(1, 11):  # Pour 10 répétitions
    print(f"\n= Run avec seed {seed} ")
    set_seeds(seed)

    # Reweighing
    weights = apply_reweighing(profiles_train, y_train, gender_train)

    # Re-train model with weights
    model_rw = load_model(MODEL_BASE_FILE)  # recharge modèle de base
    model_rw = train_reweighed_model(model_rw, profiles_train, bios_train, y_train, weights)

    # Predict
    scores_rw = evaluate_model(model_rw, profiles_test, bios_test)


    # Préparer les labels et attributs protégés pour le test
    df_test = pd.DataFrame(profiles_test)
    df_test['label'] = y_test
    df_test['gender'] = gender_test

    # Dataset réel
    dataset_test = BinaryLabelDataset(
        df=df_test,
        label_names=['label'],
        protected_attribute_names=['gender'],
        favorable_label=1,
        unfavorable_label=0
    )

    # Évaluation AVANT reweighing (modèle original) 
    # Charger modèle original et prédire
    model_orig = load_model(MODEL_BASE_FILE)
    scores_orig = model_orig.predict([profiles_test, bios_test])
    scores_bin_orig = (scores_orig >= np.percentile(scores_orig, 100*(1 - y_test.mean()))).astype(int)

    dataset_pred_orig = dataset_test.copy()
    dataset_pred_orig.labels = scores_bin_orig.reshape(-1, 1)

    metric_orig = ClassificationMetric(
        dataset_test, dataset_pred_orig,
        unprivileged_groups=[{'gender': 1}],
        privileged_groups=[{'gender': 0}]
    )

    # print("=== Fairness metrics AVANT reweighing ===")
    # print(f"Statistical Parity Difference: {metric_orig.statistical_parity_difference():.4f}")
    # print(f"Equal Opportunity Difference: {metric_orig.equal_opportunity_difference():.4f}")
    # print(f"Error Rate Difference: {metric_orig.error_rate_difference():.4f}")
    # print(f"Average Odds Difference: {metric_orig.average_odds_difference():.4f}")
    # print(f"Disparate Impact: {metric_orig.disparate_impact():.4f}")
    # print(f"Accuracy: {metric_orig.accuracy():.4f}")


    #  Évaluation APRÈS reweighing ===
    scores_bin_rw = (scores_rw >= np.percentile(scores_rw, 100*(1 - y_test.mean()))).astype(int)

    dataset_pred_rw = dataset_test.copy()
    dataset_pred_rw.labels = scores_bin_rw.reshape(-1, 1)

    metric_rw = ClassificationMetric(
        dataset_test, dataset_pred_rw,
        unprivileged_groups=[{'gender': 1}],
        privileged_groups=[{'gender': 0}]
    )

    print("=== Fairness metrics APRÈS reweighing ===")
    print(f"Statistical Parity Difference: {metric_rw.statistical_parity_difference():.4f}")
    print(f"Equal Opportunity Difference: {metric_rw.equal_opportunity_difference():.4f}")
    print(f"Error Rate Difference: {metric_rw.error_rate_difference():.4f}")
    print(f"Average Odds Difference: {metric_rw.average_odds_difference():.4f}")
    print(f"Disparate Impact: {metric_rw.disparate_impact():.4f}")
    print(f"Accuracy: {metric_rw.accuracy():.4f}")

    results.append({
        "seed": seed,
        "Statistical Parity Difference": metric_rw.statistical_parity_difference(),
        "Equal Opportunity Difference": metric_rw.equal_opportunity_difference(),
        "Average Odds Difference": metric_rw.average_odds_difference(),
        "Disparate Impact": metric_rw.disparate_impact(),
        "Error Rate Difference": metric_rw.error_rate_difference(),
        "Accuracy": metric_rw.accuracy()
    })

df = pd.DataFrame(results)
print("\n=== Moyenne des 10 runs ===")
print(df.mean(numeric_only=True).round(4))

print("\n=== Écart-type des 10 runs ===")
print(df.std(numeric_only=True).round(4))

np.save(OUT_DIR / 'FairCVtest_gender_reweighed.npy', {'Scores': scores_rw})
df.to_csv(OUT_DIR / "FairCV_preprocessGender_metrics_reweighing_10seeds.csv", index=False)