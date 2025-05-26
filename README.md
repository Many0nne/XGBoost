# MSPR.AI - Prédiction de l’évolution des cas ou décès d’une maladie

Ce projet implémente une Intelligence Artificielle permettant de prédire l’évolution des cas ou des décès liés à une maladie dans un pays donné, à partir de données historiques stockées dans une base MariaDB/MySQL. Il utilise des techniques de machine learning (XGBoost) et propose une visualisation des résultats.

## Fonctionnalités

- Prédiction simultanée de plusieurs cibles (`new_cases`, `new_deaths`, `new_recovered`) avec un seul appel.
- Chargement dynamique des données depuis la base, seules les colonnes nécessaires sont récupérées selon les cibles choisies.
- Création automatique de features temporelles (moyennes mobiles, décalages, etc.), avec ajout conditionnel des features selon la disponibilité des colonnes (ex : `cases_per_100k` uniquement si `new_cases` et `population` sont présents).
- Entraînement d’un modèle de prédiction (XGBoost) pour chaque cible, avec possibilité de tuning des hyperparamètres.
- Génération et sauvegarde de graphiques individuels et combinés pour toutes les cibles sélectionnées.
- Visualisation avancée : taux de mortalité (si `new_cases` et `new_deaths` sont prédits), graphiques de résidus, etc.
- Sauvegarde automatique des métriques d’évaluation (R², RMSE, MAE) pour chaque cible.
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

1. Lancer la prédiction et la génération des graphiques :
   ```bash
   python main.py --country "votre_pays" --days <nbr_jours_a_predire> --targets new_cases new_deaths
   ```
   - Les cibles sont dynamiques, tu peux en choisir une ou plusieurs parmi : `new_cases`, `new_deaths`, `new_recovered`.
   - Par défaut, toutes les cibles sont prédites si vous ne spécifiez pas de paramètre.

2. Résultat  
   - Un graphique de prédiction sera généré dans le dossier visualization/.
   - Le modèle entraîné est sauvegardé dans le dossier models/ pour réutilisation.
   - Les métriques (R², RMSE, MAE) seront sauvegardées dans visualization/.
   - Un graphique des résidus sera généré dans visualization/.
   - Les prédictions seront sauvegardées dans un fichier CSV dans visualization/.

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

## Benchmark / Présentation de XGBoost

### Pourquoi avoir choisi XGBoost ?

Le choix du modèle s’est porté sur **XGBoost** plutôt que sur un algorithme comme Random Forest pour plusieurs raisons :

- **Performance et scalabilité** : XGBoost est reconnu pour sa rapidité d’entraînement et sa capacité à gérer de grands volumes de données. Même si le projet actuel ne traite pas encore de très gros jeux de données, XGBoost offre une solution robuste et évolutive pour l’avenir.
- **Gestion du surapprentissage** : Grâce à la régularisation intégrée (L1 et L2), XGBoost limite le risque de surapprentissage, ce qui est un avantage par rapport à Random Forest.
- **Flexibilité** : XGBoost permet de personnaliser de nombreux paramètres (profondeur des arbres, taux d’apprentissage, sous-échantillonnage, etc.), ce qui le rend adaptable à différents types de problèmes.
- **Communauté et documentation** : XGBoost est largement utilisé dans la communauté data science, ce qui facilite l’accès à des ressources, des exemples et du support.

> **Remarque :** Pour ce projet, XGBoost est sans doute un peu « overkill » étant donné la taille modeste des données, mais il garantit une solution performante et facilement extensible si le volume de données augmente.

---

### Comment fonctionne le gradient boosting ?

Le **gradient boosting** est une méthode d’ensemble qui construit un modèle prédictif en combinant plusieurs modèles faibles (généralement des arbres de décision).  
Le principe :

1. On commence par entraîner un premier arbre sur les données.
2. À chaque itération suivante, un nouvel arbre est entraîné pour corriger les erreurs (résidus) du modèle précédent.
3. Chaque nouvel arbre est ajouté à l’ensemble, et la prédiction finale est la somme pondérée des prédictions de tous les arbres.
4. L’optimisation se fait en suivant le gradient de la fonction de perte (d’où le nom « gradient boosting »).

Ce processus permet d’obtenir un modèle global très performant, capable de capturer des relations complexes dans les données.

---

### Comment fonctionne XGBoost pour la prédiction ?

- **XGBoost** (Extreme Gradient Boosting) est une implémentation optimisée du gradient boosting.
- Lors de l’entraînement, XGBoost construit séquentiellement des arbres de décision, chaque arbre cherchant à corriger les erreurs des arbres précédents.
- Chaque arbre est construit en minimisant une fonction de perte (par exemple, l’erreur quadratique pour la régression) et en appliquant une régularisation pour éviter le surapprentissage.
- Lors de la prédiction, chaque arbre dans XGBoost apporte une correction additive à la prédiction précédente. La prédiction finale est obtenue par la somme des contributions de tous les arbres, pondérée éventuellement par un taux d'apprentissage.
- XGBoost utilise des techniques avancées comme le sous-échantillonnage des colonnes et des lignes, la gestion efficace des valeurs manquantes, et l’optimisation parallèle pour accélérer l’entraînement.

**En résumé :**  
XGBoost est un algorithme puissant, flexible et rapide, particulièrement adapté aux problèmes de régression ou de classification sur des données structurées. Il est souvent le choix privilégié dans les compétitions de data science pour ses performances et sa capacité à gérer des jeux de données volumineux et complexes.

---

### Exemple illustratif du fonctionnement du gradient boosting

Imaginons que l’on souhaite prédire la température à quatre moments d’une journée : matin, midi, après-midi, soir.  
Voici les températures réelles :  
- Matin : 10°C  
- Midi : 16°C  
- Après-midi : 15°C  
- Soir : 11°C  

#### Étape 1 : Premier arbre (modèle faible)
Le premier arbre prédit la moyenne pour tous les moments :  
Prédiction = 13°C partout

Erreurs (résidus) :  
- Matin : 10 - 13 = **-3**
- Midi : 16 - 13 = **+3**
- Après-midi : 15 - 13 = **+2**
- Soir : 11 - 13 = **-2**

On calcule la racine de l’erreur quadratique moyenne (RMSE) :  
RMSE = sqrt(((−3)² + 3² + 2² + (−2)²) / 4) = sqrt((9 + 9 + 4 + 4) / 4) = sqrt(26 / 4) ≈ **2.55**

#### Étape 2 : Deuxième arbre (apprend sur les résidus)
Le deuxième arbre apprend à corriger les erreurs du premier.  
Supposons qu’il prédit :  
- Matin : -2  
- Midi : +2  
- Après-midi : +1  
- Soir : -1  

On additionne les prédictions du premier et du deuxième arbre (avec un taux d’apprentissage, par exemple 0.5) :  
Nouvelle prédiction = Prédiction arbre 1 + 0.5 × Prédiction arbre 2

- Matin : 13 + 0.5×(−2) = 12  
- Midi : 13 + 0.5×2 = 14  
- Après-midi : 13 + 0.5×1 = 13.5  
- Soir : 13 + 0.5×(−1) = 12.5  

Nouveaux résidus :  
- Matin : 10 - 12 = **-2**
- Midi : 16 - 14 = **+2**
- Après-midi : 15 - 13.5 = **+1.5**
- Soir : 11 - 12.5 = **-1.5**

Nouveau RMSE = sqrt((4 + 4 + 2.25 + 2.25) / 4) = sqrt(12.5 / 4) ≈ **1.77**

#### Étape 3 : Troisième arbre (apprend sur les nouveaux résidus)
On répète le processus : un nouvel arbre apprend sur les nouveaux résidus, et on additionne ses corrections aux prédictions précédentes.

---

**En résumé :**  
À chaque étape, un nouvel arbre corrige les erreurs des arbres précédents.  
La prédiction finale est la somme pondérée des prédictions de tous les arbres.  
Le RMSE diminue à chaque itération, ce qui montre que le modèle s’améliore progressivement.

Ce principe est exactement celui utilisé par XGBoost, mais à grande échelle et avec des optimisations avancées.

## Auteur

Terry Barillon
