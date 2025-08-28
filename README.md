# Memoire_Biais_IA
Ce dépôt contient le code développé dans le cadre de mon mémoire de maîtrise consacré à l’évaluation et la mitigation des biais algorithmiques dans les systèmes d’intelligence artificielle appliqués au recrutement intitué : **L’équité dans les systèmes d’intelligence artificielle pour la prise de décision : atténuer les biais algorithmiques genrés et raciaux en apprentissage automatique**
L’objectif de ce projet est d’analyser la robustesse de différentes méthodes de mitigation (pré-traitement, in-processing, post-traitement) à l’aide du dataset **FairCVdb** (https://github.com/BiDAlab/FairCVtest/tree/master/data) et de bibliothèques spécialisées telles qu’[IBM AI Fairness 360] (https://github.com/Trusted-AI/AIF360).

---

##  Organisation du dépôt

- `descriptives/`  
  Scripts d’analyse descriptive des données (statistiques de base sur le dataset FairCVdb, répartition par genre/origine, etc.).
  - `analysedescriptivetrain.py`  
  - `analysesdescriptivetest.py`

- `bias_experiments/`  
  Expériences avec AIF360 pour calculer différentes métriques d’équité (SPD, EOD, AOD, DI, ERD, etc.) sur les modèles neutres et biaisés.  
  - `faircvtestexp_gender_aif360.py`  
  - `faircvtestexp_neutral_aif360.py`

- `inprocessing/`  
  Implémentations de **l’Adversarial Debiasing** (méthode in-processing) avec TensorFlow et AIF360.  
  Inclut des expériences avec variables proxy corrélées au genre (corrélations 0.3, 0.5, 0.7, 0.9).  
  - `InprocessAdvdebGender.py`  
  - `InprocessAdvdebiNeutral.py`  
  - `InprocessAdvdebGenderProxy0_3.py`  
  - `InprocessAdvdebGenderProxy0_5.py`  
  - `InprocessAdvdebGenderProxy0_7.py`  
  - `InprocessAdvdebGenderProxy0_9.py`

- `requirements.txt`  
  Liste des dépendances Python nécessaires à l’exécution des scripts.

---

## ⚙️ Installation

1. **Cloner le dépôt :**
   ```bash
   git clone https://github.com/<ton-profil>/memoire-biais-ia.git
   cd memoire-biais-ia


Technologies principales
Python 3.9+
NumPy
Pandas
Matplotlib
scikit-learn
TensorFlow (compat.v1)
IBM AI Fairness 360
