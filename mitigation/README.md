# Méthodes de mitigation des biais  

Ce dossier contient des scripts implémentant des **stratégies de mitigation des biais** sur le **jeu de données FairCVdb**, en utilisant [IBM AI Fairness 360 (AIF360)](https://aif360.readthedocs.io/en/stable/).  
Chaque méthode a été appliquée au scénario de recrutement avec le **genre** comme attribut protégé.  

## Catégories  

### 1. Pré-traitement  
Méthodes appliquées *avant* l’entraînement afin de réduire les biais dans le jeu de données.  

- **`preprocessing/ReweighingGender.py`**  
  Applique l’algorithme **Reweighing**  
  Ajuste les poids des instances dans les données d’entraînement afin d’équilibrer les résultats entre groupes protégés.  
  Réentraîne le modèle de recrutement de base avec les échantillons rééquilibrés et évalue les métriques d’équité (SPD, EOD, AOD, ERD, DI, Accuracy).  
  **Sorties :**  
  - Fichier `.npy` avec les nouveaux scores de prédiction  
  - Fichier `.csv` avec les métriques d’équité sur 10 tirages aléatoires  

### 2. En-traitement  
Méthodes appliquées *pendant* l’entraînement en modifiant l’algorithme d’apprentissage.  

- **`inprocessing/AdversarialDebiasingGender.py`**  
  Implémente le **Débiasage Adversarial**  
  Entraîne un classificateur tout en minimisant simultanément la capacité d’un adversaire à prédire l’attribut protégé à partir des prédictions.  
  Produit un classificateur « équitable » dont les prédictions contiennent moins d’informations discriminatoires.  
  **Sorties :**  
- **Fichiers CSV** avec la moyenne ± écart-type sur 10 exécutions
-  Fichiers `.npy` avec les scores de prédiction après mitigation  

### 3. Post-traitement  
Méthodes appliquées *après* l’entraînement pour ajuster les prédictions du modèle.  

- **`postprocessing/RejectOptionClassifier.py`**  
  Applique la technique du **Reject Option Classification**  
  Relabelise les prédictions incertaines de façon à améliorer l’équité tout en maintenant la précision.  

## Sorties  

Pour chaque méthode, les scripts génèrent :  
- **Métriques d’équité :**  
  - Statistical Parity Difference (SPD)  
  - Equal Opportunity Difference (EOD)  
  - Average Odds Difference (AOD)  
  - Error Rate Difference (ERD)  
  - Disparate Impact (DI)  
  - Accuracy  
- **Fichiers CSV**  
-  Fichiers `.npy` avec les scores de prédiction après mitigation  
