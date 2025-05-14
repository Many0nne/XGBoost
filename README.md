# MSPR.AI - Prédiction de l’évolution des cas ou décès d’une maladie

Ce projet implémente une Intelligence Artificielle permettant de prédire l’évolution des cas ou des décès liés à une maladie dans un pays donné, à partir de données historiques stockées dans une base MariaDB/MySQL. Il utilise des techniques de machine learning (XGBoost) et propose une visualisation des résultats.

## Fonctionnalités

- Chargement des données depuis une base de données relationnelle.
- Création automatique de features temporelles (moyennes mobiles, décalages, etc.).
- Entraînement d’un modèle de prédiction (XGBoost) pour estimer les valeurs futures.
- Génération et sauvegarde de graphiques comparant les données historiques et les prédictions.
- Modularité du code (séparé en plusieurs fichiers pour la base, le traitement, le modèle, la visualisation).

## Structure du projet

MSPR.AI/
│
├── predictor/
│   ├── database.py
│   ├── data_processing.py
│   ├── model.py
│   └── visualization.py
├── docs/
│   └── database.sql
├── visualization/
├── venv/
├── models/
├── main.py
├── .gitignore
├── requirements.txt
└── README.md

## Installation

1. Cloner le dépôt  
   git clone <url_du_repo>
   cd MSPR.AI

2. Créer un environnement virtuel  
   python -m venv venv

3. Activer l’environnement virtuel  
   - Windows :
     .\venv\Scripts\activate
   - Linux/Mac :
     source venv/bin/activate

4. Installer les dépendances  
   pip install -r requirements.txt

## Configuration de la base de données

- Le schéma SQL est disponible dans docs/database.sql.
- Adaptez les identifiants de connexion dans main.py si besoin.

## Utilisation

1. Lancer la prédiction et la génération du graphique  
   python main.py

2. Résultat  
   - Un graphique de prédiction sera généré dans le dossier visualization/.
   - Le modèle entraîné est sauvegardé dans le dossier models/ pour réutilisation.

## Personnalisation

- Modifiez les paramètres dans main.py pour changer le pays, la cible (new_cases ou new_deaths), ou le nombre de jours à prédire.
- Le code est modulaire : chaque étape (chargement, features, modèle, visualisation) peut être adaptée ou réutilisée.

## Tests

Tests unitaires disponibles
- pytests Tests/

## Dépendances principales

- Python 3.10+
- pandas
- SQLAlchemy
- PyMySQL (ou MariaDB connector)
- scikit-learn
- xgboost
- matplotlib
- joblib

## Remarques

- Le projet fonctionne avec MariaDB ou MySQL (le connecteur PyMySQL est compatible MariaDB).
- Le dossier venv/, __pycache__/ et visualization/ sont ignorés par Git (voir .gitignore).

## Auteur

Terry Barillon