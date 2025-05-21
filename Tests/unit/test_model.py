import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import pandas as pd
from predictor.model import PandemicModel

def test_train_model():
    df = pd.DataFrame({
        "new_cases": [1,2,3,4,5],
        "new_deaths": [0,1,0,1,0],
        "population": [1000]*5,
        "lag_1": [0,1,2,3,4],
        "rolling_7_mean": [1,1.5,2,2.5,3],
        "cases_per_100k": [0.1]*5,
        "day_of_week": [0,1,2,3,4],
        "day_of_month": [1,2,3,4,5],
        "month": [1,1,1,1,1],
    }, index=pd.date_range("2023-01-01", periods=5))
    model_manager = PandemicModel()
    # Génération de la liste des features comme dans main.py
    feature_names = [col for col in df.columns if col.startswith('lag_') or 
                     col.startswith('rolling_') or 
                     col in ['day_of_week', 'day_of_month', 'month', 'cases_per_100k', 'deaths_per_100k']]
    model, metrics = model_manager.train_model(df, "new_cases", feature_names=feature_names)
    assert model is not None
    assert "MAE" in metrics