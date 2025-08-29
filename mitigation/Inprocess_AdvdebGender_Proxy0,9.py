import numpy as np
import pandas as pd
import os
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from pathlib import Path


from sklearn.preprocessing import StandardScaler
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.metrics import ClassificationMetric

#   Configuration  
DATA_PATH = Path(os.environ.get("FAIRCV_PATH", "data"))
DATABASE_FILE = "FairCVdb.npy"
PROXY_CORRELATION = 0.9

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)


#   Load Dataset  
fairCV = np.load(str(DATA_PATH / DATABASE_FILE), allow_pickle=True).item()
X_train = fairCV['Profiles Train'][:, 4:31]
X_test = fairCV['Profiles Test'][:, 4:31]
gender_train = fairCV['Profiles Train'][:, 1].astype(int)
gender_test = fairCV['Profiles Test'][:, 1].astype(int)
labels_train = fairCV['Biased Labels Train (Gender)']
labels_test = fairCV['Blind Labels Test']

#   Add proxy variable  
def add_proxy(X, gender_array, rho):
    gender_scaled = StandardScaler().fit_transform(gender_array.reshape(-1, 1)).flatten()
    proxy = rho * gender_scaled + np.sqrt(1 - rho**2) * np.random.randn(len(gender_scaled))
    return np.concatenate((X, proxy.reshape(-1, 1)), axis=1)

X_train_proxy = add_proxy(X_train, gender_train, PROXY_CORRELATION)
X_test_proxy = add_proxy(X_test, gender_test, PROXY_CORRELATION)

#   Binarize Labels  
threshold = np.median(labels_train)
y_train = (labels_train >= threshold).astype(int)
y_test = (labels_test >= np.median(labels_test)).astype(int)

#   Create AIF360 Datasets  
df_train = pd.DataFrame(X_train_proxy)
df_train['label'] = y_train
df_train['gender'] = gender_train

df_test = pd.DataFrame(X_test_proxy)
df_test['label'] = y_test
df_test['gender'] = gender_test

dataset_train = BinaryLabelDataset(df=df_train, label_names=['label'], protected_attribute_names=['gender'])
dataset_test = BinaryLabelDataset(df=df_test, label_names=['label'], protected_attribute_names=['gender'])

#   Baseline Model: Logistic Regression  
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_proxy, y_train)
baseline_preds = clf.predict(X_test_proxy)

df_baseline_pred = df_test.copy()
df_baseline_pred['label'] = baseline_preds
dataset_baseline_pred = BinaryLabelDataset(df=df_baseline_pred, label_names=['label'], protected_attribute_names=['gender'])


results = []
for seed in range(1, 11):
    print(f"\n = Run avec seed {seed}  =")
    set_seeds(seed)
#   Adversarial Debiasing Model  
    sess = tf.Session()
    debiased_model = AdversarialDebiasing(
        privileged_groups=[{'gender': 0}],
        unprivileged_groups=[{'gender': 1}],
        scope_name=f'adv_debias_{seed}',
        debias=True,
        sess=sess,
        num_epochs=50,
        batch_size=128
    )

    debiased_model.fit(dataset_train)
    debiased_preds = debiased_model.predict(dataset_test)


    metric_after = ClassificationMetric(dataset_test, debiased_preds,
                                        unprivileged_groups=[{'gender': 1}],
                                        privileged_groups=[{'gender': 0}])
    

    results.append({
        "seed" : seed, 
        "Statistical Parity Difference": metric_after.statistical_parity_difference(),
        "Equal Opportunity Difference": metric_after.equal_opportunity_difference(),
        "Average Odds Difference": metric_after.average_odds_difference(),
        "Disparate Impact": metric_after.disparate_impact(),
        "Error Rate Difference": metric_after.error_rate_difference(),
        "Accuracy": metric_after.accuracy()
    })
    print(f"\n  Fairness Metrics APRÈS Adversarial Debiasing   Proxy {PROXY_CORRELATION}")
    print(f"Statistical Parity Difference: {metric_after.statistical_parity_difference():.4f}")
    print(f"Equal Opportunity Difference: {metric_after.equal_opportunity_difference():.4f}")
    print(f"Average Odds Difference: {metric_after.average_odds_difference():.4f}")
    print(f"Disparate Impact: {metric_after.disparate_impact():.4f}")
    print(f"Accuracy: {metric_after.accuracy():.4f}")



#   Résumé des 10 runs  
df = pd.DataFrame(results)
print("\n=== Moyenne des 10 runs ===")
print(df.mean(numeric_only=True).round(4))

print("\n=== Écart-type des 10 runs ===")
print(df.std(numeric_only=True).round(4))

out_csv = DATA_PATH / "FairCV_inprocessinggender_metrics_10seeds_0.9.csv"
df.to_csv(str(out_csv), index=False)
print(f"Résultats exportés : {out_csv}")