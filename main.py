from predictor.database import create_db_engine, load_data
from predictor.data_processing import create_features
from predictor.model import PandemicModel
from predictor.visualization import visualize_all_results
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Prédiction de cas/décès par IA")
    parser.add_argument("--country", type=str, default="France", help="Nom du pays à prédire")
    parser.add_argument("--days", type=int, default=7, help="Nombre de jours à prédire")
    parser.add_argument("--no-train", action="store_true", help="Utiliser les modèles existants sans ré-entraînement")
    parser.add_argument("--tune", action="store_true", help="Effectuer un tuning des hyperparamètres")
    parser.add_argument("--targets", nargs="+", 
                       choices=["new_cases", "new_deaths", "new_recovered"],
                       default=["new_cases", "new_deaths", "new_recovered"],
                       help="Cibles à prédire (par défaut: toutes)")
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
    targets = args.targets

    # Chargement des données (en passant les targets)
    df = load_data(engine, country_name, targets=targets)

    model_manager = PandemicModel()
    predictions = {}
    metrics = {}

    for target in targets:
        # Vérifier que la colonne cible existe bien dans les données chargées
        if target not in df.columns:
            print(f"Attention: La colonne {target} n'est pas présente dans les données chargées.")
            continue

        # Création des features pour chaque cible
        df_features = create_features(df.copy(), target, look_back=30, 
                                    use_lags=True, use_rolling=True, use_calendar=True)

        # Définition de la liste des features
        feature_names = [col for col in df_features.columns 
            if col.startswith('lag_') or 
            col.startswith('rolling_') or 
            col in ['day_of_week', 'day_of_month', 'month', 
                    'cases_per_100k', 'deaths_per_100k', 'recovered_per_100k']
            and col in df_features.columns]

        if not args.no_train:
            # Entraînement du modèle avec option de tuning
            print(f"Entraînement du modèle pour {target}...")
            model, target_metrics = model_manager.train_model(
                df_features, target, feature_names=feature_names, 
                tune_hyperparams=args.tune
            )
            # Sauvegarde du modèle
            model_manager.save_model(model, target)
            metrics[target] = target_metrics
            print(f"Modèle entraîné pour {target} - Métriques: {target_metrics}")
        else:
            # Chargement du modèle existant
            print(f"Chargement du modèle existant pour {target}...")
            model = model_manager.load_model(target)
            if model is None:
                print(f"Aucun modèle trouvé pour {target}, entraînement d'un nouveau modèle.")
                model, target_metrics = model_manager.train_model(
                    df_features, target, feature_names=feature_names, 
                    tune_hyperparams=args.tune
                )
                model_manager.save_model(model, target)
                metrics[target] = target_metrics

        # Prédictions futures
        print(f"Génération des prédictions pour {target}...")
        preds = model_manager.predict_future(
            df_features, target, feature_names=feature_names, 
            days_ahead=days_ahead
        )
        
        # On clippe les valeurs prédites pour éviter les valeurs négatives
        preds[f"predicted_{target}"] = preds[f"predicted_{target}"].clip(lower=0)
        predictions[target] = preds

        # Sauvegarde des prédictions dans un CSV
        preds.to_csv(f"visualization/{country_name}_{target}_predictions.csv")
        print(f"Prédictions sauvegardées pour {target}")

    # Génération de toutes les visualisations
    if predictions:  # Vérifie si le dictionnaire n'est pas vide
        print("Génération des visualisations...")
        visualize_all_results(df, predictions, country_name)
        
        # Sauvegarde des métriques
        for target, target_metrics in metrics.items():
            with open(f"visualization/{country_name}_{target}_metrics.txt", "w") as f:
                for key, value in target_metrics.items():
                    f.write(f"{key}: {value}\n")
        print("Traitement terminé avec succès!")
    else:
        print("Aucune prédiction générée. Vérifiez les données chargées et les targets spécifiées.")

if __name__ == "__main__":
    main()