import pandas as pd

def create_features(df: pd.DataFrame, target: str, look_back: int = 30,
                    use_lags: bool = True, use_rolling: bool = True, use_calendar: bool = True) -> pd.DataFrame:
    """
    Fonction pour créer des features à partir des données historiques.

    Args:
        df (pd.DataFrame): DataFrame contenant les données historiques.
        target (str): Nom de la colonne cible à prédire.
        look_back (int): Nombre de jours pour les features de décalage temporel.
        use_lags (bool): Indique si les features de décalage temporel doivent être créées.
        use_rolling (bool): Indique si les moyennes mobiles doivent être créées.
        use_calendar (bool): Indique si les features temporelles doivent être créées.

    Returns:
        pd.DataFrame: DataFrame avec les nouvelles features ajoutées.
    """
    # Supprimer les anciennes colonnes de features si elles existent
    for i in range(1, look_back + 1):
        col = f'lag_{i}'
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    if 'rolling_7_mean' in df.columns:
        df.drop(columns=['rolling_7_mean'], inplace=True)
    if 'rolling_30_mean' in df.columns:
        df.drop(columns=['rolling_30_mean'], inplace=True)
    if 'day_of_week' in df.columns:
        df.drop(columns=['day_of_week'], inplace=True)
    if 'day_of_month' in df.columns:
        df.drop(columns=['day_of_month'], inplace=True)
    if 'month' in df.columns:
        df.drop(columns=['month'], inplace=True)
    if 'cases_per_100k' in df.columns:
        df.drop(columns=['cases_per_100k'], inplace=True)
    if 'deaths_per_100k' in df.columns:
        df.drop(columns=['deaths_per_100k'], inplace=True)

    # Features de décalage temporel
    if use_lags:
        for i in range(1, look_back + 1):
            df[f'lag_{i}'] = df[target].shift(i)
    # Moyennes mobiles
    if use_rolling:
        df['rolling_7_mean'] = df[target].rolling(7).mean()
        df['rolling_30_mean'] = df[target].rolling(30).mean()
    # Features temporelles
    if use_calendar:
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
    # Ratio cas/population
    if 'population' in df.columns:
        df['cases_per_100k'] = (pd.to_numeric(df['new_cases'], errors='coerce') / (pd.to_numeric(df['population'], errors='coerce') / 100000)).fillna(0)
        df['deaths_per_100k'] = (pd.to_numeric(df['new_deaths'], errors='coerce') / (pd.to_numeric(df['population'], errors='coerce') / 100000)).fillna(0)
    
    # Suppression des lignes avec valeurs manquantes
    cols_to_check = [target]
    if use_lags:
        cols_to_check += [f'lag_{i}' for i in range(1, look_back + 1)]
    if use_rolling:
        cols_to_check += ['rolling_7_mean', 'rolling_30_mean']
    if use_calendar:
        cols_to_check += ['day_of_week', 'day_of_month', 'month']
    if 'population' in df.columns:
        cols_to_check += ['cases_per_100k', 'deaths_per_100k']

    df = df.dropna(subset=cols_to_check)
    return df