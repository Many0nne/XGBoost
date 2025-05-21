import matplotlib.pyplot as plt
import os
import numpy as np

def ensure_dir_exists(output_dir):
    """
    Vérifie si le répertoire de sortie existe, sinon le crée.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def plot_predictions(df, preds, target, country_name, output_dir="visualization"):
    """
    Enregistre un graphique comparant les données historiques et les prédictions dans un fichier.

    Args:
        df (pd.DataFrame): DataFrame contenant les données historiques.
        preds (pd.DataFrame): DataFrame contenant les prédictions.
        target (str): Nom de la colonne cible à prédire.
        country_name (str): Nom du pays pour lequel les prédictions sont faites.
        output_dir (str): Répertoire de sortie pour sauvegarder le graphique.

    Returns:
        None
    """
    ensure_dir_exists(output_dir)
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[target], label='Données historiques', color='blue')
    plt.plot(preds.index, preds[f'predicted_{target}'], label='Prédictions', color='red', linestyle='--')
    plt.title(f"Prédiction des {target} pour {country_name}")
    plt.xlabel('Date')
    plt.ylabel(f"Nombre de {target}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{country_name}_{target}_predictions.png")
    plt.savefig(output_path)
    plt.close()

def save_metrics(metrics: dict, country_name: str, target: str, output_dir="visualization"):
    """
    Sauvegarde les métriques d'évaluation du modèle dans un fichier texte.

    Args:
        metrics (dict): Dictionnaire contenant les métriques d'évaluation.
        country_name (str): Nom du pays pour lequel les métriques sont calculées.
        target (str): Nom de la colonne cible à prédire.
        output_dir (str): Répertoire de sortie pour sauvegarder le fichier texte.

    Returns:
        None
    """
    ensure_dir_exists(output_dir)
    output_path = os.path.join(output_dir, f"{country_name}_{target}_metrics.txt")
    with open(output_path, "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

def plot_residuals(y_true, y_pred, country_name, target, output_dir="visualization"):
    """
    Génère et sauvegarde le graphique des résidus (erreurs) du modèle.

    Args:
        y_true (np.ndarray): Valeurs réelles.
        y_pred (np.ndarray): Valeurs prédites.
        country_name (str): Nom du pays pour lequel les résidus sont calculés.
        target (str): Nom de la colonne cible à prédire.
        output_dir (str): Répertoire de sortie pour sauvegarder le graphique.

    Returns:
        None
    """
    ensure_dir_exists(output_dir)
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 5))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Valeurs prédites")
    plt.ylabel("Résidus (y_true - y_pred)")
    plt.title(f"Résidus du modèle pour {country_name} ({target})")
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{country_name}_{target}_residuals.png")
    plt.savefig(output_path)
    plt.close()