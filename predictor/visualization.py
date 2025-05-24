import matplotlib.pyplot as plt
import os
import numpy as np

def ensure_dir_exists(output_dir):
    """Vérifie si le répertoire de sortie existe, sinon le crée."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def plot_predictions(df, preds, target, country_name, output_dir="visualization"):
    """Enregistre un graphique comparant les données historiques et les prédictions."""
    ensure_dir_exists(output_dir)
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[target], label='Données historiques', color='blue')
    
    last_date = df.index[-1]
    last_value = df[target].iloc[-1]
    all_pred_dates = [last_date] + list(preds.index)
    all_pred_values = [last_value] + list(preds[f'predicted_{target}'])
    
    plt.plot(all_pred_dates, all_pred_values, label='Prédictions', color='red')
    plt.title(f"Prédiction des {target} pour {country_name}")
    plt.xlabel('Date')
    plt.ylabel(f"Nombre de {target}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"{country_name}_{target}_predictions.png")
    plt.savefig(output_path)
    plt.close()

def plot_combined_predictions(df, cases_preds, deaths_preds, country_name, output_dir="visualization"):
    """Enregistre un graphique combinant les prédictions de cas et décès."""
    ensure_dir_exists(output_dir)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Graphique des cas
    ax1.plot(df.index, df['new_cases'], label='Cas historiques', color='blue')
    last_date = df.index[-1]
    last_value = df['new_cases'].iloc[-1]
    all_pred_dates = [last_date] + list(cases_preds.index)
    all_pred_values = [last_value] + list(cases_preds['predicted_new_cases'])
    ax1.plot(all_pred_dates, all_pred_values, label='Prédictions cas', color='red')
    ax1.set_ylabel("Nombre de nouveaux cas")
    ax1.legend()
    ax1.grid(True)
    
    # Graphique des décès
    ax2.plot(df.index, df['new_deaths'], label='Décès historiques', color='green')
    last_value = df['new_deaths'].iloc[-1]
    all_pred_values = [last_value] + list(deaths_preds['predicted_new_deaths'])
    ax2.plot(all_pred_dates, all_pred_values, label='Prédictions décès', color='orange')
    ax2.set_ylabel("Nombre de nouveaux décès")
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle(f"Prédictions COVID-19 pour {country_name}")
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"{country_name}_combined_predictions.png")
    plt.savefig(output_path)
    plt.close()

def save_metrics(metrics: dict, country_name: str, target: str, output_dir="visualization"):
    """Sauvegarde les métriques d'évaluation du modèle."""
    ensure_dir_exists(output_dir)
    output_path = os.path.join(output_dir, f"{country_name}_{target}_metrics.txt")
    with open(output_path, "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

def plot_residuals(y_true, y_pred, country_name, target, output_dir="visualization"):
    """Génère et sauvegarde le graphique des résidus."""
    ensure_dir_exists(output_dir)
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 5))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='red')
    plt.xlabel("Valeurs prédites")
    plt.ylabel("Résidus (y_true - y_pred)")
    plt.title(f"Résidus du modèle pour {country_name} ({target})")
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"{country_name}_{target}_residuals.png")
    plt.savefig(output_path)
    plt.close()