from predictor.database import create_db_engine, load_data
from predictor.data_processing import create_features
from predictor.model import PandemicModel
from predictor.visualization import (plot_predictions, plot_residuals, save_metrics, plot_combined_predictions)
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

def main():
    parser = argparse.ArgumentParser(description="Prédiction de cas/décès par IA")
    parser.add_argument("--country", type=str, default="France", help="Nom du pays à prédire")
    parser.add_argument("--days", type=int, default=7, help="Nombre de jours à prédire")
    parser.add_argument("--no-train", action="store_true", help="Utiliser les modèles existants sans ré-entraînement")
    parser.add_argument("--tune", action="store_true", help="Effectuer un tuning des hyperparamètres")
    args = parser.parse_args()

    # Configuration de la base de données
    DB_USER = "root"
    DB_PASSWORD = "root"
    DB_HOST = "localhost"
    DB_NAME = "pandemia"

    # S'assurer que les dossiers existent
    os.makedirs("visualization", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Initialisation
    engine = create_db_engine(DB_USER, DB_PASSWORD, DB_HOST, DB_NAME)
    country_name = args.country
    days_ahead = args.days

    # Chargement des données
    df = load_data(engine, country_name)

    # Liste des cibles à prédire
    targets = ["new_cases", "new_deaths"]
    predictions = {}

    model_manager = PandemicModel()
    
    for target in targets:
        # Vérifier que la colonne cible existe bien
        if target not in df.columns:
            print(f"Colonne {target} absente, passage.")
            continue

        # Création des features pour chaque cible
        df_features = create_features(df.copy(), target, look_back=30, use_lags=True, use_rolling=True, use_calendar=True)

        # Définition unique de la liste des features
        feature_names = [col for col in df_features.columns if col.startswith('lag_') or col.startswith('rolling_') or col in ['day_of_week', 'day_of_month', 'month', 'cases_per_100k', 'deaths_per_100k']]

        if not args.no_train:
            # Entraînement du modèle avec option de tuning
            model, metrics = model_manager.train_model(df_features, target, feature_names=feature_names,tune_hyperparams=args.tune)
            # Sauvegarde du modèle
            model_manager.save_model(model, target)
        else:
            # Chargement du modèle existant
            model = model_manager.load_model(target)
            if model is None:
                print(f"Aucun modèle trouvé pour {target}, entraînement d'un nouveau modèle.")
                model, metrics = model_manager.train_model(df_features, target, feature_names=feature_names,tune_hyperparams=args.tune)
                model_manager.save_model(model, target)

        # Prédictions futures
        preds = model_manager.predict_future(df_features, target, feature_names=feature_names, days_ahead=days_ahead)
        
        # On clippe les valeurs prédites pour éviter les valeurs négatives
        preds[f"predicted_{target}"] = preds[f"predicted_{target}"].clip(lower=0)
        predictions[target] = preds

        # Enregistrement des métriques
        if not args.no_train:
            save_metrics(metrics, country_name, target)

        # Sauvegarde des prédictions dans un CSV
        preds.to_csv(f"visualization/{country_name}_{target}_predictions.csv")

        # Enregistrement du graphique réel vs prédictions
        plot_predictions(df_features, preds, target, country_name)

        # Graphique des résidus (si on a assez de données et qu'on a entraîné)
        if not args.no_train and len(df_features) >= len(preds):
            y_true = df_features[target].iloc[-len(preds):].values
            y_pred = preds[f"predicted_{target}"].values
            plot_residuals(y_true, y_pred, country_name, target)

    # Graphique combiné si les deux cibles sont disponibles
    if all(t in predictions for t in targets):
        plot_combined_predictions(df_features, predictions["new_cases"], predictions["new_deaths"], country_name)

if __name__ == "__main__":
    main()