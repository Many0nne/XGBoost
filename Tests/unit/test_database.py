import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from predictor.database import create_db_engine

def test_create_db_engine():
    engine = create_db_engine("user", "password", "localhost", "testdb")
    assert engine is not None