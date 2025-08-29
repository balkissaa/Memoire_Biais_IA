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

- `preprocessing/`  
  Implémentations de la méthode **Reweighing** (pré-traitement) avec AIF360.  
  Inclut également les expériences avec variables proxy corrélées au genre (ρ = 0.3, 0.5, 0.7, 0.9).  
  - `MethodepreprocessingNeutralRNoriginal.py`  
  - `MethodepreprocessingGenderRNoriginal.py`  
  - `method_preproc_reweighing_Proxy_0.3.py`  
  - `method_preproc_reweighing_Proxy_0.5.py`  
  - `method_preproc_reweighing_Proxy_0.7.py`  
  - `method_preproc_reweighing_Proxy_0.9.py`  

- `inprocessing/`  
  Implémentations de **l’Adversarial Debiasing** (in-processing) avec TensorFlow et AIF360.  
  Expériences menées sur données neutres, genrées et proxy (ρ = 0.3, 0.5, 0.7, 0.9).  
  - `Inprocess_Advdeb_Gender.py`  
  - `Inprocess_Advdeb_Gender_Proxy0.3.py`  
  - `Inprocess_Advdeb_Gender_Proxy0.5.py`  
  - `Inprocess_Advdeb_Gender_Proxy0.7.py`  
  - `Inprocess_Advdeb_Gender_Proxy0.9.py`  

- `postprocessing/`  
  Implémentations de la méthode **Reject Option Classification (ROC)** (post-traitement).  
  Appliquée aux données neutres, genrées et proxy.  
  - `Postprocessing_Neutral.py`  
  - `Postprocessing_Gender.py`  
  - `Postprocess_Proxy0.3.py`  
  - `Postprocess_Proxy0.5.py`  
  - `Postprocess_Proxy0.7.py`  
  - `Postprocess_Proxy0.9.py`  

---
Le code utilise des bibliothèques courantes en science des données, telles que :  
Python 3.9+
NumPy
Pandas
Matplotlib
seaborn
scikit-learn
TensorFlow (compat.v1)
IBM AI Fairness 360
