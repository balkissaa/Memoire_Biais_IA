import numpy as np
import os
import pickle
from pathlib import Path
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.metrics import ClassificationMetric
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras as keras
import random
import tensorflow as tf


#CONFIGURATION 
DATA_PATH = Path(os.environ.get("FAIRCV_PATH", "data"))  # changer via env 
DATABASE_FILE = "FairCVdb.npy"
TOKENIZER_FILE = DATA_PATH / "Tokenizer gender.pickle"
MODEL_BASE_FILE = DATA_PATH / "HiringTool gender.h5"

OUT_DIR = Path("outputs/preprocessing")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOKENIZER_FILE = os.path.join(DATA_PATH, "Tokenizer gender.pickle")
MODEL_BASE_FILE = os.path.join(DATA_PATH, "HiringTool gender.h5")
NEUTRAL_SCORES_FILE = os.path.join(DATA_PATH, "FairCVtest neutral.npy")
MAX_WORDS = 20000

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)  # Pour hash Python
    np.random.seed(seed)   #Pour NumPy
    random.seed(seed) #Pour random
    tf.random.set_seed(seed) #Pour TensorFlow

# LOAD DATA
def load_dataset():
    fairCV = np.load(DATA_PATH / DATABASE_FILE, allow_pickle=True).item()
    profiles_train = fairCV['Profiles Train'][:, 4:31]
    profiles_test = fairCV['Profiles Test'][:, 4:31]
    gender_train = fairCV['Profiles Train'][:, 1].astype(int)
    gender_test = fairCV['Profiles Test'][:, 1].astype(int)
    labels_train = fairCV['Blind Labels Train']
    labels_test = fairCV['Blind Labels Test']
    return profiles_train, profiles_test, gender_train, gender_test, labels_train, labels_test


def load_bios_and_tokenizer():
    fairCV = np.load(DATA_PATH / DATABASE_FILE, allow_pickle=True).item()
    bios_train = fairCV['Bios Train'][:, 0]
    bios_test = fairCV['Bios Test'][:, 0]
    with open(TOKENIZER_FILE, 'rb') as f:
        tokenizer = pickle.load(f)
    return bios_train, bios_test, tokenizer

#pretraitement bios 
def process_bios(bios_raw, tokenizer, max_len):
    sequences = tokenizer.texts_to_sequences(bios_raw)
    bios = pad_sequences(sequences, maxlen=max_len)
    return bios

#crées un BinaryLabelDataset pour le training, on appliques la méthode Reweighing et récupères les poids par instance. Ces poids corrigent les déséquilibres dans les données
# REWEIGHING 
def apply_reweighing(X_train, y_train, gender_train):
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
    RW = Reweighing(unprivileged_groups=[{'gender': 1}], privileged_groups=[{'gender': 0}])
    dataset_transf = RW.fit_transform(dataset)
    return dataset_transf.instance_weights


# TRAIN MODEL 
def train_reweighed_model(model, X_struct, X_bios, y_train, weights):
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanAbsoluteError())
    model.fit([X_struct, X_bios], y_train, sample_weight=weights, batch_size=128, epochs=16)
    return model


# EVALUATION 
def binarize_scores(scores, y_true):
    threshold = np.percentile(scores, 100 * (1 - y_true.mean()))
    return (scores >= threshold).astype(int)


# RUN 
profiles_train, profiles_test, gender_train, gender_test, labels_train_raw, labels_test_raw = load_dataset()
bios_train_raw, bios_test_raw, tokenizer = load_bios_and_tokenizer()
model = load_model(MODEL_BASE_FILE)
max_len = model.input[1].shape[1]

# BINARIZE LABELS 
threshold = np.median(labels_train_raw)
y_train = (labels_train_raw >= threshold).astype(int)
y_test = (labels_test_raw >= threshold).astype(int)

#  PROCESS BIOS 
bios_train = process_bios(bios_train_raw, tokenizer, max_len)
bios_test = process_bios(bios_test_raw, tokenizer, max_len)

# LOAD NEUTRAL SCORES 
neutral_scores = np.load(NEUTRAL_SCORES_FILE, allow_pickle=True).item()['Scores']
neutral_preds = binarize_scores(neutral_scores, y_test)

# =CREATE AIF360 TEST DATASET 
df_test = pd.DataFrame(profiles_test)
df_test['label'] = y_test
df_test['gender'] = gender_test

dataset_test = BinaryLabelDataset(
    df=df_test,
    label_names=['label'],
    protected_attribute_names=['gender'],
    favorable_label=1,
    unfavorable_label=0
)

#  FAIRNESS METRICS AVANT 
dataset_pred_orig = dataset_test.copy()
dataset_pred_orig.labels = neutral_preds.reshape(-1, 1)

metric_orig = ClassificationMetric(
    dataset_test, dataset_pred_orig,
    unprivileged_groups=[{'gender': 1}],
    privileged_groups=[{'gender': 0}]
)

print("=== Fairness metrics AVANT reweighing ===")
print(f"Statistical Parity Difference: {metric_orig.statistical_parity_difference():.4f}")
print(f"Equal Opportunity Difference: {metric_orig.equal_opportunity_difference():.4f}")
print(f"Error Rate Difference: {metric_orig.error_rate_difference():.4f}")
print(f"Average Odds Difference: {metric_orig.average_odds_difference():.4f}")
print(f"Disparate Impact: {metric_orig.disparate_impact():.4f}")
print(f"Accuracy: {metric_orig.accuracy():.4f}")


results = []
for seed in range(1, 11):  # Pour 10 répétitions
    print(f"\n==== Run avec seed {seed} ====")
    set_seeds(seed)

    # === APPLY REWEIGHING + RE-TRAIN ===
    weights = apply_reweighing(profiles_train, y_train, gender_train)
    model_rw = load_model(MODEL_BASE_FILE)
    model_rw = train_reweighed_model(model_rw, profiles_train, bios_train, y_train, weights)

    # === EVALUATE REWEIGHED MODEL ===
    scores_rw = model_rw.predict([profiles_test, bios_test])
    rw_preds = binarize_scores(scores_rw, y_test)

    dataset_pred_rw = dataset_test.copy()
    dataset_pred_rw.labels = rw_preds.reshape(-1, 1)

    metric_rw = ClassificationMetric(
        dataset_test, dataset_pred_rw,
        unprivileged_groups=[{'gender': 1}],
        privileged_groups=[{'gender': 0}]
    )

    print(f"SPD: {metric_rw.statistical_parity_difference():.4f} | "
      f"EOD: {metric_rw.equal_opportunity_difference():.4f} | "
      f"AOD: {metric_rw.average_odds_difference():.4f} | "
      f"DI: {metric_rw.disparate_impact():.4f} | "
      f"ERD: {metric_rw.error_rate_difference():.4f} | "
      f"Acc: {metric_rw.accuracy():.4f}")

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

np.save(OUT_DIR / 'FairCVtest_neutral_reweighed.npy', {'Scores': scores_rw})
df.to_csv(OUT_DIR / "FairCV_preprocessneutral_metrics_reweighing_10seeds.csv", index=False)