from predictor.database import create_db_engine, load_data
from predictor.data_processing import create_features
from predictor.model import PandemicModel
from predictor.visualization import plot_predictions, plot_residuals, save_metrics
import argparse
import os

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
    parser = argparse.ArgumentParser(description="Prédiction de cas/décès par IA")
    parser.add_argument("--country", type=str, default="France", help="Nom du pays à prédire")
    parser.add_argument("--days", type=int, default=7, help="Nombre de jours à prédire")
    args = parser.parse_args()

    # Configuration de la base de données
    DB_USER = "root"
    DB_PASSWORD = "root"
    DB_HOST = "localhost"
    DB_NAME = "pandemia"

    # S'assurer que le dossier visualization existe
    if not os.path.exists("visualization"):
        os.makedirs("visualization")

    # Initialisation
    engine = create_db_engine(DB_USER, DB_PASSWORD, DB_HOST, DB_NAME)
    country_name = args.country
    days_ahead = args.days

    # Chargement des données
    df = load_data(engine, country_name)

    # Liste des cibles à prédire
    targets = ["new_cases", "new_deaths"]

    for target in targets:
        # Vérifier que la colonne cible existe bien
        if target not in df.columns:
            print(f"Colonne {target} absente, passage.")
            continue

        # Création des features pour chaque cible
        df_features = create_features(df.copy(), target, look_back=30, use_lags=True, use_rolling=True, use_calendar=True)

        # Définition unique de la liste des features
        feature_names = [col for col in df_features.columns if col.startswith('lag_') or 
                         col.startswith('rolling_') or 
                         col in ['day_of_week', 'day_of_month', 'month', 'cases_per_100k', 'deaths_per_100k']]

        # Initialisation du gestionnaire de modèles
        model_manager = PandemicModel()

        # Entraînement du modèle
        model, metrics = model_manager.train_model(df_features, target, feature_names=feature_names)

        # Enregistrement des métriques (si tu veux les sauvegarder)
        save_metrics(metrics, country_name, target)

        # Prédictions futures
        predictions = model_manager.predict_future(df_features, target, feature_names=feature_names, days_ahead=days_ahead)

        # On clippe les valeurs prédites pour éviter les valeurs négatives
        predictions[f"predicted_{target}"] = predictions[f"predicted_{target}"].clip(lower=0)

        # Sauvegarde des prédictions dans un CSV
        predictions.to_csv(f"visualization/{country_name}_{target}_predictions.csv")

        # Enregistrement du graphique réel vs prédictions
        plot_predictions(df_features, predictions, target, country_name)

        # Graphique des résidus (optionnel)
        if len(df_features) >= len(predictions):
            y_true = df_features[target].iloc[-len(predictions):].values
            y_pred = predictions[f"predicted_{target}"].values
            plot_residuals(y_true, y_pred, country_name, target)