# New Repository Game Predictor 

**Projet de Master 2 – Prédire les tendences de création sur github**

Ce projet à pour objectif d'utiliser l'API Github afin de **prédire le nombre de nouveau répertoires créer par semaines** pour un jeux spécifié. Il utilise des modèles de machine larning qui sont réentrainé par semaine, automatique en choisissant le modèle pour effectuer une prédiction en se basant sur des données des semaines passées. 

---

## Features

- **Prediction hebdomadaire**
  Predire combien de nouveaux repertoires Github seront créer la semaine prochaine pour un jeu précis.

- **Plusieurs modèles de Machine Learning**
  Réentraine et compare plusieurs modèles chaques semaines puis séléctionne le meilleur pour effectuer les prédictions.

- **Entrainement et récolte de données automatiques**
  Toute la chaîn de traitement depui la collecte des données jusqu'au réentrainement et la prédictions sont automatisés

- **Endpoints API**
  Inclut des requêtes API pour :
    - Entraîné manuellement : 
    - Collecter les données : 
    - Faire des prédictions : 

---

## Technologies utilisés
- **Python** - pour les modèles et le projet de manière général
- **Docker** & **docker-compose** - containerisation et mise en place
- Librairie de Machine Learning (ex cikit-learn, pandas, etc) pour l'entrainement

---

##  Pour le reproduire

### Cloner le répertoire

```bash
git clone https://github.com/MasterPNJ/new-repository-game-predictor.git
cd new-repository-game-predictor
