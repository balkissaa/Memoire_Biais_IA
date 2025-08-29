Ce dossier contient les scripts utilisés pour évaluer quantitativement les biais dans le jeu de données **FairCVdb** à l’aide de la librairie IBM AI Fairness 360.  

### Scripts  

- **`faircvtestexp_gender_aif360.py`**  
  Calcule plusieurs métriques d’équité (Statistical Parity Difference, Equal Opportunity Difference, Average Odds Difference, Error Rate Difference, Disparate Impact, Accuracy) à partir des **scores prédits biaisés par le genre**.  
  Produit un fichier CSV résumant les résultats.  

- **`faircvtestexp_neutral_aif360.py`**  
  Réalise la même évaluation sur des **scores prédits neutres (sans biais explicite)**.  
  Sert de référence comparative par rapport au scénario biaisé.  
  Produit également un fichier CSV résumant les résultats.  
