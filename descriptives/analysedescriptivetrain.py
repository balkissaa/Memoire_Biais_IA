import os
from pathlib import Path
import numpy as np
import pandas as pd 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


#config
# Chemins et dossiers de sortie
#Remplacez le chemin par l'emplacement où se trouve votre dossier "data"
DATA_DIR = Path(os.environ.get("FAIRCV_PATH", "data"))
FAIRCV_FILE = DATA_DIR / "FairCVdb.npy"
OUT_DIR = Path("outputs_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)

#chargement donnees
fairCV = np.load(FAIRCV_FILE, allow_pickle=True).item()

profiles_train = fairCV['Profiles Train']
blind_labels_train = fairCV['Blind Labels Train']            
biased_labels_gender_train = fairCV['Biased Labels Train (Gender)']     
biased_labels_ethnicity_train = fairCV['Biased Labels Train (Ethnicity)']


def pct_table(series, sort_index=False):
    """Retourne DataFrame de pourcentages trié prêt à exporter."""
    s = series.value_counts(normalize=True, dropna=False) * 100
    s = s.sort_index() if sort_index else s.sort_values(ascending=False)
    return s.round(2).to_frame("Pourcentage (%)")

def export_table(df, name, caption=None, label=None):
    """Exporte en CSV et LaTeX."""
    df.to_csv(OUT_DIR / f"{name}.csv")
    if caption is None:
        caption = f"Répartition de {name}"
    if label is None:
        label = f"tab:{name}"
    with open(OUT_DIR / f"{name}.tex", "w") as f:
        f.write(df.to_latex(escape=True, caption=caption, label=label))

def bar_plot_from_table(df_pct, title, fname):
    """Crée un diagramme en barres PNG à partir d’un tableau de pourcentages (index=catégories)."""
    plt.figure(figsize=(8, 5))
    df_pct.iloc[:, 0].plot(kind="bar")
    plt.ylabel("Pourcentage (%)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{fname}.png", dpi=200)
    plt.close()

# Colonnes (structure  FairCVdb)
# 0: ethnicité (0=G1, 1=G2, 2=G3)
# 1: genre (0=Homme, 1=Femme)
# 2: occupation (0..9)
# 3: adéquation poste (0.25, 0.5, 0.75, 1)
# 4: éducation (0.2, 0.4, 0.6, 0.8, 1)
# 5: expérience (0, 0.2, 0.4, 0.6, 0.8, 1)
# 6: lettre reco (0/1)
# 7: disponibilité (0.2..1)
# 8:11 langues (3 colonnes)
# 11:31 face embeddings (20 dims)
# 31:51 blind face embeddings (20 dims)

df_train = pd.DataFrame({
    'ethnicity':   profiles_train[:, 0].astype(int),
    'gender':      profiles_train[:, 1].astype(int),
    'occupation':  profiles_train[:, 2].astype(int),
    'job_fit':     profiles_train[:, 3].astype(float),
    'education':   profiles_train[:, 4].astype(float),
    'experience':  profiles_train[:, 5].astype(float),
    'rec_letter':  profiles_train[:, 6].astype(int),
    'availability':profiles_train[:, 7].astype(float),
    'lang1':       profiles_train[:, 8].astype(float),
    'lang2':       profiles_train[:, 9].astype(float),
    'lang3':       profiles_train[:,10].astype(float),
})

# Mappings lisibles
eth_map = {0: 'G1', 1: 'G2', 2: 'G3'}
gender_map = {0: 'Homme', 1: 'Femme'}
occup_map = {
    0:'Infirmier', 1:'Chirurgien', 2:'Médecin', 3:'Journaliste', 4:'Photographe',
    5:'Cinéaste', 6:'Enseignant', 7:'Professeur', 8:'Avocat', 9:'Comptable'
}

df_train_named = df_train.copy()
df_train_named['ethnicity'] = df_train_named['ethnicity'].map(eth_map)
df_train_named['gender'] = df_train_named['gender'].map(gender_map)
df_train_named['occupation'] = df_train_named['occupation'].map(occup_map)

#DESCRIPTIVES(%)

print("=== Taille train ===")
print(len(df_train_named))

#Genre
tab_genre = pct_table(df_train_named['gender'])
print("\n Répartition Genre (train, %)")
print(tab_genre)
export_table(tab_genre, "train_genre", caption="Répartition du genre (train)", label="tab:train_genre")
bar_plot_from_table(tab_genre, "Répartition Genre (train)", "train_genre")

#Ethnicite
tab_eth = pct_table(df_train_named['ethnicity'])
print("\n Répartition Ethnicité (train, %) ")
print(tab_eth)
export_table(tab_eth, "train_ethnicite", caption="Répartition de l’ethnicité (train)", label="tab:train_ethnicite")
bar_plot_from_table(tab_eth, "Répartition Ethnicité (train)", "train_ethnicite")

#Occupations
tab_occ = pct_table(df_train_named['occupation'])
print("\nOccupations (train, %) ")
print(tab_occ)
export_table(tab_occ, "train_occupations", caption="Répartition des occupations (train)", label="tab:train_occupations")
bar_plot_from_table(tab_occ, "Répartition Occupations (train)", "train_occupations")

#Variables ordinales / binaires
tab_jobfit = pct_table(df_train_named['job_fit'], sort_index=True)
export_table(tab_jobfit, "train_job_fit", caption="Répartition de l’adéquation au poste (train)", label="tab:train_job_fit")

tab_edu = pct_table(df_train_named['education'], sort_index=True)
export_table(tab_edu, "train_education", caption="Répartition du niveau d’éducation (train)", label="tab:train_education")

tab_exp = pct_table(df_train_named['experience'], sort_index=True)
export_table(tab_exp, "train_experience", caption="Répartition de l’expérience professionnelle (train)", label="tab:train_experience")

tab_rec = pct_table(df_train_named['rec_letter'])
export_table(tab_rec, "train_rec_letter", caption="Répartition des lettres de recommandation (train)", label="tab:train_rec_letter")

tab_av = pct_table(df_train_named['availability'], sort_index=True)
export_table(tab_av, "train_availability", caption="Répartition de la disponibilité (train)", label="tab:train_availability")

print("\n Adéquation poste (train, %) ")
print(tab_jobfit)
print("\n Education (train, %) ")
print(tab_edu)
print("\n Expérience (train, %) ")
print(tab_exp)
print("\n Lettre de reco (train, %)")
print(tab_rec)
print("\n Disponibilité (train, %) ")
print(tab_av)

# LANGUES (3 colonnes)
# Moyennes
lang_means = df_train_named[['lang1','lang2','lang3']].mean().to_frame("Moyenne").round(3)
print("\n Compétences linguistiques (moyennes par langue)")
print(lang_means)
export_table(lang_means, "train_lang_means", caption="Moyennes des compétences linguistiques (train)", label="tab:train_lang_means")

#Distributions
for col in ['lang1','lang2','lang3']:
    tab_lang = pct_table(df_train_named[col], sort_index=True)
    export_table(tab_lang, f"train_{col}_distribution", caption=f"Répartition {col} (train)", label=f"tab:train_{col}")

#LABELS BLIND (CONTINUS) + BINARISATION

blind_series = pd.Series(blind_labels_train.ravel())
desc = blind_series.describe()
print("\n--- Labels 'Blind' (train, continu) : résumé ")
print(desc)

#Binarisation a la médiane
threshold = float(np.median(blind_labels_train))
labels_bin = (blind_labels_train.ravel() >= threshold).astype(int)
tab_labels_bin = pct_table(pd.Series(labels_bin))
print(f"\nSeuil (médiane) = {threshold:.4f}")
print("Distribution des labels binarisés (train, %)")
print(tab_labels_bin)
export_table(tab_labels_bin, "train_labels_bin", caption="Distribution des labels binarisés (train)", label="tab:train_labels_bin")


#Export resumecontinu
desc_df = pd.DataFrame(desc)
export_table(desc_df, "train_blind_labels_summary", caption="Résumé statistique des labels 'blind' (train)", label="tab:train_blind_summary")

print(f"\nExports disponibles dans : {OUT_DIR}")
print(" CSV + LaTeX pour chaque tableau")
print(" PNG pour genre/ethnicité/occupations")

print(df_train[['job_fit','education','experience','availability']].corr())
