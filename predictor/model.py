from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import timedelta
import logging
import numpy as np
import joblib
import os
import pandas as pd

logger = logging.getLogger(__name__)

class PandemicModel:
    def __init__(self, model_dir="models"):
        self.models = {}
        self.model_dir = model_dir

    def train_model(self, df, target, feature_names, look_back=30, test_size=0.2):
        """
        Entraîne un modèle XGBoost pour prédire les nouveaux cas ou décès et le sauvegarde.
        """
        # Création des features
        X = df[feature_names]
        y = df[target]
        
        # Séparation train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        
        # Configuration du modèle
        import xgboost as xgb
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.8,
            early_stopping_rounds=50,
            random_state=42
        )
        
        # Entraînement
        logger.info("Début de l'entraînement du modèle XGBoost.")
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=10
        )
        
        # Évaluation
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        logger.info(f"Modèle entraîné pour {target} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")
        
        # Vérification et création du répertoire `models/` si nécessaire
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            logger.info(f"Répertoire {self.model_dir} créé.")

        # Sauvegarde du modèle
        model_path = f"{self.model_dir}/{target}_model.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Modèle sauvegardé dans {model_path}")
        
        return model, {'MAE': mae, 'RMSE': rmse, 'R2': r2}

    def load_model(self, target):
        """
        Charge un modèle sauvegardé pour une cible donnée.
        """
        model_path = f"{self.model_dir}/{target}_model.pkl"
        try:
            model = joblib.load(model_path)
            logger.info(f"Modèle chargé depuis {model_path}")
            return model
        except FileNotFoundError:
            logger.warning(f"Aucun modèle trouvé pour {target} dans {model_path}.")
            return None

    def predict_future(self, df, target, feature_names, days_ahead=7, look_back=30):
        """
        Prédit les valeurs futures pour la cible spécifiée.

        Args:
            - df (pd.DataFrame): DataFrame contenant les données historiques.
            - target (str): Nom de la colonne cible pour la prédiction.
            - feature_names (list): Liste des features utilisées pour l'entraînement et la prédiction.
            - days_ahead (int): Nombre de jours à prédire dans le futur.
            - look_back (int): Nombre de jours pour les features de décalage temporel.

        Returns:
            - predictions (pd.DataFrame): DataFrame contenant les dates et les valeurs prédites.
        """
        # Charger un modèle existant ou entraîner un nouveau modèle
        model = self.load_model(target)
        if model is None:
            logger.info(f"Entraînement d'un nouveau modèle pour {target}.")
            model, _ = self.train_model(df, target, feature_names, look_back)
        
        # Générer les prédictions
        predictions = []
        current_date = df.index.max()
        last_data = df.iloc[-1:].copy()

        for i in range(1, days_ahead + 1):
            current_date = current_date + timedelta(days=1)
            X_pred = last_data[feature_names]
            pred = model.predict(X_pred)[0]
            predictions.append({
                'date': current_date,
                f'predicted_{target}': pred
            })
            # Mettre à jour last_data pour la prochaine prédiction
            new_row = last_data.copy()
            new_row[target] = pred
            # Décaler les lags
            for lag in range(look_back, 1, -1):
                new_row[f'lag_{lag}'] = new_row[f'lag_{lag-1}']
            if 'lag_1' in new_row.columns:
                new_row['lag_1'] = pred
            # Mettre à jour les moyennes mobiles si besoin
            if 'rolling_7_mean' in new_row.columns:
                vals = list(last_data[target].values[-6:]) + [pred]
                new_row['rolling_7_mean'] = np.mean(vals)
            if 'rolling_30_mean' in new_row.columns:
                vals = list(last_data[target].values[-29:]) + [pred]
                new_row['rolling_30_mean'] = np.mean(vals)
            # Mettre à jour la date
            new_row.index = [current_date]
            last_data = pd.concat([last_data, new_row]).iloc[1:]
            current_date = new_row.index[0]

        return pd.DataFrame(predictions).set_index('date')