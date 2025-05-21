import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import pandas as pd
from predictor.data_processing import create_features

def test_create_features():
    df = pd.DataFrame({"new_cases": [1,2,3,4,5], "new_deaths": [0,1,0,1,0], "population": [1000]*5}, index=pd.date_range("2023-01-01", periods=5))
    result = create_features(df, "new_cases", look_back=2)
    assert "lag_1" in result.columns
    assert "rolling_7_mean" in result.columns

def test_create_features_no_lags():
    df = pd.DataFrame({"new_cases": [1,2,3,4,5], "new_deaths": [0,1,0,1,0], "population": [1000]*5}, index=pd.date_range("2023-01-01", periods=5))
    result = create_features(df, "new_cases", look_back=2, use_lags=False)
    assert "lag_1" not in result.columns
    assert "rolling_7_mean" in result.columns