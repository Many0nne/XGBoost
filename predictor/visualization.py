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

def plot_combined_predictions(df, cases_preds, deaths_preds, recovered_preds, country_name, output_dir="visualization"):
    """Enregistre un graphique combinant les prédictions de cas et décès."""
    ensure_dir_exists(output_dir)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
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

    # Graphique des guérisons
    ax3.plot(df.index, df['new_recovered'], label='Guérisons historiques', color='purple')
    last_value = df['new_recovered'].iloc[-1]
    all_pred_values = [last_value] + list(recovered_preds['predicted_new_recovered'])
    ax3.plot(all_pred_dates, all_pred_values, label='Prédictions guérisons', color='pink')
    ax3.set_ylabel("Nouvelles guérisons")
    ax3.set_xlabel('Date')
    ax3.legend()
    ax3.grid(True)
    
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

def plot_mortality_rate(df, cases_preds, deaths_preds, country_name, output_dir="visualization"):
    """Calcule et affiche le taux de mortalité (new_deaths / new_cases) historique et prédit."""
    ensure_dir_exists(output_dir)
    
    # Calcul du taux de mortalité historique
    historical_rate = (df['new_deaths'] / df['new_cases']).replace([np.inf, -np.inf], np.nan).fillna(0) * 100
    
    # Calcul du taux de mortalité prédit
    pred_rate = (deaths_preds['predicted_new_deaths'] / cases_preds['predicted_new_cases']).replace([np.inf, -np.inf], np.nan).fillna(0) * 100
    
    # Création du graphique
    plt.figure(figsize=(12, 6))
    
    # Taux historique
    plt.plot(df.index, historical_rate, label='Taux de mortalité historique', color='blue')
    
    # Taux prédit
    last_date = df.index[-1]
    last_value = historical_rate.iloc[-1]
    all_pred_dates = [last_date] + list(deaths_preds.index)
    all_pred_values = [last_value] + list(pred_rate)
    
    plt.plot(all_pred_dates, all_pred_values, label='Taux de mortalité prédit', color='red')
    
    # Calcul et affichage de la variation
    if len(pred_rate) > 0:
        initial_rate = historical_rate.iloc[-1]
        final_rate = pred_rate.iloc[-1]
        variation = final_rate - initial_rate
        variation_text = f"Variation prévue: {variation:.2f}%"
        plt.annotate(variation_text, xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.title(f"Taux de mortalité (décès/cas) pour {country_name}")
    plt.xlabel('Date')
    plt.ylabel('Taux de mortalité (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"{country_name}_mortality_rate.png")
    plt.savefig(output_path)
    plt.close()

def save_mortality_stats(cases_preds, deaths_preds, country_name, output_dir="visualization"):
    """Sauvegarde les statistiques de mortalité dans un fichier texte."""
    ensure_dir_exists(output_dir)
    
    # Calcul des taux de mortalité
    mortality_rate = (deaths_preds['predicted_new_deaths'] / cases_preds['predicted_new_cases']).replace([np.inf, -np.inf], np.nan).fillna(0) * 100
    
    # Calcul de la variation
    if len(mortality_rate) > 1:
        initial_rate = mortality_rate.iloc[0]
        final_rate = mortality_rate.iloc[-1]
        variation = final_rate - initial_rate
        variation_pct = (variation / initial_rate) * 100 if initial_rate != 0 else 0
    else:
        variation = 0
        variation_pct = 0
    
    # Sauvegarde dans un fichier
    output_path = os.path.join(output_dir, f"{country_name}_mortality_stats.txt")
    with open(output_path, "w") as f:
        f.write(f"Taux de mortalité initial: {initial_rate:.2f}%\n")
        f.write(f"Taux de mortalité final: {final_rate:.2f}%\n")
        f.write(f"Variation absolue: {variation:.2f} points\n")
        f.write(f"Variation relative: {variation_pct:.2f}%\n")
        f.write("\nDétails par jour:\n")
        for date, rate in mortality_rate.items():
            f.write(f"{date.date()}: {rate:.2f}%\n")