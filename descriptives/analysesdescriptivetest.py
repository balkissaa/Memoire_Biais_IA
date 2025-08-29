import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Chemins et dossiers de sortie
#Remplacez le chemin par l'emplacement où se trouve votre dossier "data"
DATA_DIR = Path(os.environ.get("FAIRCV_PATH", "data"))
FAIRCV_FILE = DATA_DIR / "FairCVdb.npy"
OUT_DIR = Path("outputs_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Charger la base FairCV (split test)
fairCV = np.load(FAIRCV_FILE, allow_pickle=True).item()
profiles_test = fairCV['Profiles Test']
blind_labels_test = fairCV['Blind Labels Test']
biased_labels_gender_test = fairCV['Biased Labels Test (Gender)']
biased_labels_ethnicity_test = fairCV['Biased Labels Test (Ethnicity)']

# fonction utilitaire pour obtenir des tableaux de proportions
def pct_table(series, sort_index=False):
    s = series.value_counts(normalize=True, dropna=False) * 100
    s = s.sort_index() if sort_index else s.sort_values(ascending=False)
    return s.round(2).to_frame("Pourcentage (%)")

# Export CSV + LaTeX
def export_table(df, name, caption=None, label=None):
    df.to_csv(OUT_DIR / f"{name}.csv")
    if caption is None:
        caption = f"Répartition de {name}"
    if label is None:
        label = f"tab:{name}"
    with open(OUT_DIR / f"{name}.tex", "w", encoding="utf-8") as f:
        f.write(df.to_latex(escape=True, caption=caption, label=label))

#Pour les graphes en barres
def bar_plot_from_table(df_pct, title, fname, rotation=0):
    plt.figure(figsize=(8, 5))
    df_pct.iloc[:, 0].plot(kind="bar")
    plt.ylabel("Pourcentage (%)")
    plt.title(title)
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{fname}.png", dpi=200)
    plt.close()

#Construction du DataFrame test
df_test = pd.DataFrame({
    'ethnicity':    profiles_test[:, 0].astype(int),
    'gender':       profiles_test[:, 1].astype(int),
    'occupation':   profiles_test[:, 2].astype(int),
    'job_fit':      profiles_test[:, 3].astype(float),
    'education':    profiles_test[:, 4].astype(float),
    'experience':   profiles_test[:, 5].astype(float),
    'rec_letter':   profiles_test[:, 6].astype(int),
    'availability': profiles_test[:, 7].astype(float),
    'lang1':        profiles_test[:, 8].astype(float),
    'lang2':        profiles_test[:, 9].astype(float),
    'lang3':        profiles_test[:,10].astype(float),
})

# Remplacer les codes par des libelles plus lisibles
eth_map = {0: 'G1', 1: 'G2', 2: 'G3'}
gender_map = {0: 'Homme', 1: 'Femme'}
occup_map = {
    0:'Infirmier', 1:'Chirurgien', 2:'Médecin', 3:'Journaliste', 4:'Photographe',
    5:'Cinéaste', 6:'Enseignant', 7:'Professeur', 8:'Avocat', 9:'Comptable'
}

df_test_named = df_test.copy()
df_test_named['ethnicity'] = df_test_named['ethnicity'].map(eth_map)
df_test_named['gender'] = df_test_named['gender'].map(gender_map)
df_test_named['occupation'] = df_test_named['occupation'].map(occup_map)

# Taille
print("Taille test :", len(df_test_named))

# Genre
tab_genre = pct_table(df_test_named['gender'])
print("\n Répartition Genre (test, %)")
print(tab_genre)
export_table(tab_genre, "test_genre", caption="Répartition du genre (test)", label="tab:test_genre")
bar_plot_from_table(tab_genre, "Répartition Genre (test)", "test_genre")

# Ethnicité
tab_eth = pct_table(df_test_named['ethnicity'])
print("\n Répartition Ethnicité (test, %)")
print(tab_eth)
export_table(tab_eth, "test_ethnicite", caption="Répartition de l’ethnicité (test)", label="tab:test_ethnicite")
bar_plot_from_table(tab_eth, "Répartition Ethnicité (test)", "test_ethnicite")

# Occupations
tab_occ = pct_table(df_test_named['occupation'])
print("\nOccupations (test, %)")
print(tab_occ)
export_table(tab_occ, "test_occupations", caption="Répartition des occupations (test)", label="tab:test_occupations")
bar_plot_from_table(tab_occ, "Répartition Occupations (test)", "test_occupations", rotation=45)

# Variables ordinales et binaires
tab_jobfit = pct_table(df_test_named['job_fit'], sort_index=True)
tab_edu = pct_table(df_test_named['education'], sort_index=True)
tab_exp = pct_table(df_test_named['experience'], sort_index=True)
tab_rec = pct_table(df_test_named['rec_letter'])
tab_av = pct_table(df_test_named['availability'], sort_index=True)

print("\nAdéquation poste (test, %)")
print(tab_jobfit)
print("\nEducation (test, %)")
print(tab_edu)
print("\nExpérience (test, %)")
print(tab_exp)
print("\nLettre de reco (test, %)")
print(tab_rec)
print("\nDisponibilité (test, %)")
print(tab_av)

export_table(tab_jobfit, "test_job_fit", caption="Répartition de l’adéquation au poste (test)", label="tab:test_job_fit")
export_table(tab_edu, "test_education", caption="Répartition du niveau d’éducation (test)", label="tab:test_education")
export_table(tab_exp, "test_experience", caption="Répartition de l’expérience professionnelle (test)", label="tab:test_experience")
export_table(tab_rec, "test_rec_letter", caption="Répartition des lettres de recommandation (test)", label="tab:test_rec_letter")
export_table(tab_av, "test_availability", caption="Répartition de la disponibilité (test)", label="tab:test_availability")

#Langues
lang_means = df_test_named[['lang1','lang2','lang3']].mean().to_frame("Moyenne").round(3)
print("\nCompétences linguistiques (moyennes)")
print(lang_means)
export_table(lang_means, "test_lang_means", caption="Moyennes des compétences linguistiques (test)", label="tab:test_lang_means")

for col in ['lang1','lang2','lang3']:
    tab_lang = pct_table(df_test_named[col], sort_index=True)
    export_table(tab_lang, f"test_{col}_distribution", caption=f"Répartition {col} (test)", label=f"tab:test_{col}")

#Labels "Blind" continus + binarisation
blind_series = pd.Series(blind_labels_test.ravel())
desc = blind_series.describe()
print("\nLabels 'Blind' (test, continu) : résumé")
print(desc)

threshold = float(np.median(blind_labels_test))
labels_bin = (blind_labels_test.ravel() >= threshold).astype(int)
tab_labels_bin = pct_table(pd.Series(labels_bin))

print(f"\nSeuil (médiane) = {threshold:.4f}")
print("Distribution labels binarisés (test, %)")
print(tab_labels_bin)

export_table(tab_labels_bin, "test_labels_bin", caption="Distribution des labels binarisés (test)", label="tab:test_labels_bin")
desc_df = pd.DataFrame(desc)
export_table(desc_df, "test_blind_labels_summary", caption="Résumé statistique des labels 'blind' (test)", label="tab:test_blind_summary")

# Corrélations entre variables numériques
print("\nCorrélations entre variables continues")
print(df_test[['job_fit','education','experience','availability']].corr())

print(f"\nExports générés dans {OUT_DIR}")
