from predictor.database import create_db_engine, load_data
from predictor.data_processing import create_features
from predictor.model import PandemicModel
from predictor.visualization import plot_predictions, save_metrics, plot_residuals

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
    
    # Définition unique de la liste des features
    feature_names = [col for col in df.columns if col.startswith('lag_') or 
                     col.startswith('rolling_') or 
                     col in ['day_of_week', 'day_of_month', 'month', 'cases_per_100k', 'deaths_per_100k']]
    
    # Initialisation du gestionnaire de modèles
    model_manager = PandemicModel()
    
    # Entraînement du modèle
    model, metrics = model_manager.train_model(df, target, feature_names=feature_names)
    
    # Prédictions futures
    predictions = model_manager.predict_future(df, target, feature_names=feature_names)

    # On clippe les valeurs prédites pour éviter les valeurs négatives
    predictions[f"predicted_{target}"] = predictions[f"predicted_{target}"].clip(lower=0)
    
    # Sauvegarde des prédictions dans un CSV
    predictions.to_csv(f"visualization/{country_name}_{target}_predictions.csv")
    print(f"Prédictions : {predictions.head()}")

    # Enregistrement du graphique réel vs prédictions
    plot_predictions(df, predictions, target, country_name)
    
    # Sauvegarde des métriques
    save_metrics(metrics, country_name, target)
    
    # Graphique des résidus (optionnel)
    if len(df) >= len(predictions):
        y_true = df[target].iloc[-len(predictions):].values
        y_pred = predictions[f"predicted_{target}"].values
        plot_residuals(y_true, y_pred, country_name, target)