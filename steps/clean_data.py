import logging
from zenml import step
import pandas as pd

@step
def clean_data(df: pd.DataFrame)->None:
    pass