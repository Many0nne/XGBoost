import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import pandas as pd
from predictor.visualization import plot_predictions

def test_plot_predictions(tmp_path):
    df = pd.DataFrame({"new_cases": [1,2,3,4,5]}, index=pd.date_range("2023-01-01", periods=5))
    preds = pd.DataFrame({"predicted_new_cases": [2,3,4,5,6,7,8]}, index=pd.date_range("2023-01-06", periods=7))
    plot_predictions(df, preds, "new_cases", "France", output_dir=str(tmp_path))
    output_file = tmp_path / "France_new_cases_predictions.png"
    assert output_file.exists()