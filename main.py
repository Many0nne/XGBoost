from predictor.database import create_db_engine, load_data
from predictor.data_processing import create_features
from predictor.model import PandemicModel
from predictor.visualization import plot_predictions

"""
Ce projet implémente une IA pour prédire l'évolution des cas ou des décès liés à une maladie dans un pays donné.
L'objectif principal est de fournir des prédictions basées sur des données historiques, en utilisant des modèles
d'apprentissage automatique (XGBoost). Les étapes incluent :

1. Chargement des données depuis une base de données MySQL.
2. Préparation des données avec des features temporelles (décalages, moyennes mobiles, etc.).
3. Entraînement d'un modèle de prédiction pour estimer les valeurs futures.
4. Génération de graphiques comparant les données historiques et les prédictions.

Ce système peut être utilisé pour anticiper les tendances et aider à la prise de décision dans des contextes
sanitaires ou épidémiologiques.
"""
if __name__ == "__main__":
    # Configuration de la base de données
    DB_USER = "root"
    DB_PASSWORD = "root"
    DB_HOST = "localhost"
    DB_NAME = "pandemia"
    
    # Initialisation
    engine = create_db_engine(DB_USER, DB_PASSWORD, DB_HOST, DB_NAME)
    country_name = "France"
    target = "new_cases"
    
    # Chargement des données
    df = load_data(engine, country_name)
    
    # Création des features
    df = create_features(df, target)
    
    # Initialisation du gestionnaire de modèles
    model_manager = PandemicModel()
    
    # Prédictions futures
    predictions = model_manager.predict_future(df, target)
    
    # Enregistrement du graphique
    plot_predictions(df, predictions, target, country_name)