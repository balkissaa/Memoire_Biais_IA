import numpy as np
import os
import pickle
import pandas as pd
import random
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.metrics import ClassificationMetric
from pathlib import Path
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM, Bidirectional, Concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


# CONFIGURATION 
DATA_PATH = Path(os.environ.get("FAIRCV_PATH", "data"))
DATABASE_FILE = "FairCVdb.npy"
TOKENIZER_FILE = DATA_PATH / "Tokenizer gender.pickle"
MAX_WORDS = 20000
PROXY_CORRELATION = 0.9 # Corrélation voulue avec le genre
EMBED_DIM = 128
EPOCHS = 16

OUT_DIR = Path("outputs/preprocessing")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)  # Pour hash Python
    np.random.seed(seed)   #Pour NumPy
    random.seed(seed) #Pour random
    tf.random.set_seed(seed) #Pour TensorFlow


#   BUILD MODEL FUNCTION  
def build_model(input_dim_structured, max_len_bios, vocab_size=MAX_WORDS, emb_dim=EMBED_DIM):
    input_struct = Input(shape=(input_dim_structured,), name="structured_input")
    x_struct = Dense(64, activation='relu')(input_struct)
    x_struct = Dropout(0.3)(x_struct)

    input_text = Input(shape=(max_len_bios,), name="bio_input")
    x_text = Embedding(input_dim=vocab_size, output_dim=emb_dim, input_length=max_len_bios)(input_text)
    x_text = Bidirectional(LSTM(64))(x_text)
    x_text = Dropout(0.3)(x_text)

    x = Concatenate()([x_struct, x_text])
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input_struct, input_text], outputs=output)
    return model

#   LOAD DATA FUNCTIONS  
def load_dataset():
    fairCV = np.load(DATA_PATH / DATABASE_FILE, allow_pickle=True).item()
    profiles_train = fairCV['Profiles Train'][:, 4:31]
    profiles_test = fairCV['Profiles Test'][:, 4:31]
    gender_train = fairCV['Profiles Train'][:, 1].astype(int)
    gender_test = fairCV['Profiles Test'][:, 1].astype(int)
    labels_train = fairCV['Biased Labels Train (Gender)']
    labels_test = fairCV['Blind Labels Test']

    return profiles_train, profiles_test, gender_train, gender_test, labels_train, labels_test

def add_proxy_variable(X, gender_array, rho):
    gender_scaled = StandardScaler().fit_transform(gender_array.reshape(-1, 1)).flatten()
    np.random.seed(42)
    proxy = rho * gender_scaled + np.sqrt(1 - rho**2) * np.random.randn(len(gender_scaled))
    return np.concatenate((X, proxy.reshape(-1, 1)), axis=1)

def load_bios_and_tokenizer():
    fairCV = np.load(DATA_PATH / DATABASE_FILE, allow_pickle=True).item()
    bios_train = fairCV['Bios Train'][:, 0]
    bios_test = fairCV['Bios Test'][:, 0]
    with open(TOKENIZER_FILE, 'rb') as f:
        tokenizer = pickle.load(f)
    return bios_train, bios_test, tokenizer

def process_bios(bios_raw, tokenizer, max_len):
    sequences = tokenizer.texts_to_sequences(bios_raw)
    return pad_sequences(sequences, maxlen=max_len)

results = []
for seed in range(1, 11):  # Pour 10 répétitions
    print(f"\n = Run avec seed {seed}  =")
    set_seeds(seed)
    #   REWEIGHING  
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

    #   TRAINING  
    def train_model(model, X_structured, X_bios, y_train, weights=None):
        model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanAbsoluteError())
        model.fit([X_structured, X_bios], y_train, sample_weight=weights, batch_size=128, epochs=EPOCHS)
        return model

    #   EVALUATION  
    def evaluate_fairness(model, X_structured, X_bios, y_test, gender_test, label_name="AVANT"):
        scores = model.predict([X_structured, X_bios])
        threshold = np.percentile(scores, 100 * (1 - y_test.mean()))
        preds = (scores >= threshold).astype(int)

        df_test = pd.DataFrame(X_structured)
        df_test['label'] = y_test
        df_test['gender'] = gender_test

        dataset_test = BinaryLabelDataset(
            df=df_test,
            label_names=['label'],
            protected_attribute_names=['gender'],
            favorable_label=1,
            unfavorable_label=0
        )

        dataset_pred = dataset_test.copy()
        dataset_pred.labels = preds.reshape(-1, 1)

        metric = ClassificationMetric(
            dataset_test, dataset_pred,
            unprivileged_groups=[{'gender': 1}],
            privileged_groups=[{'gender': 0}]
        )

        print(f"=== Fairness metrics {label_name} Reweighing (proxy corr={PROXY_CORRELATION}) ===")
        print(f"Statistical Parity Difference: {metric.statistical_parity_difference():.4f}")
        print(f"Equal Opportunity Difference: {metric.equal_opportunity_difference():.4f}")
        print(f"Error Rate Difference: {metric.error_rate_difference():.4f}")
        print(f"Average Odds Difference: {metric.average_odds_difference():.4f}")
        print(f"Disparate Impact: {metric.disparate_impact():.4f}")
        print(f"Accuracy: {metric.accuracy():.4f}")

        return metric    

    #   RUN  
    profiles_train, profiles_test, gender_train, gender_test, labels_train_raw, labels_test_raw = load_dataset()
    bios_train_raw, bios_test_raw, tokenizer = load_bios_and_tokenizer()

    # Binarize labels
    threshold = np.median(labels_train_raw)
    y_train = (labels_train_raw >= threshold).astype(int)
    y_test = (labels_test_raw >= np.median(labels_test_raw)).astype(int)

    # Add proxy variable
    profiles_train_proxy = add_proxy_variable(profiles_train, gender_train, PROXY_CORRELATION)
    profiles_test_proxy = add_proxy_variable(profiles_test, gender_test, PROXY_CORRELATION)

    # Prepare bios
    max_len = 300
    bios_train = process_bios(bios_train_raw, tokenizer, max_len)
    bios_test = process_bios(bios_test_raw, tokenizer, max_len)

    # Build and train initial model
    model_orig = build_model(input_dim_structured=profiles_train_proxy.shape[1], max_len_bios=max_len)
    model_orig = train_model(model_orig, profiles_train_proxy, bios_train, y_train)
    #evaluate_fairness(model_orig, profiles_test_proxy, bios_test, y_test, gender_test, label_name="AVANT")

    # Reweighing and retrain
    weights = apply_reweighing(profiles_train_proxy, y_train, gender_train)
    model_rw = build_model(input_dim_structured=profiles_train_proxy.shape[1], max_len_bios=max_len)
    model_rw = train_model(model_rw, profiles_train_proxy, bios_train, y_train, weights=weights)
    metric = evaluate_fairness(model_rw, profiles_test_proxy, bios_test, y_test, gender_test, label_name="APRÈS")

    results.append({
        "seed": seed,
        "type": "APRÈS",
        "Statistical Parity Difference": metric.statistical_parity_difference(),
        "Equal Opportunity Difference": metric.equal_opportunity_difference(),
        "Error Rate Difference": metric.error_rate_difference(),
        "Average Odds Difference": metric.average_odds_difference(),
        "Disparate Impact": metric.disparate_impact(),
        "Accuracy": metric.accuracy()
        })
    

df = pd.DataFrame(results)
print("\n=== Moyenne des 10 runs ===")
print(df.mean(numeric_only=True).round(4))

print("\n=== Écart-type des 10 runs ===")
print(df.std(numeric_only=True).round(4))

out_csv = OUT_DIR / f"FairCV_preprocessGender_metrics_reweighing_10seedsproxy{PROXY_CORRELATION}.csv"
df.to_csv(out_csv, index=False)
print(f"Résultats exportés : {out_csv}")