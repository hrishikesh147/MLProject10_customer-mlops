import logging
from zenml import pipeline
import pandas as pd
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import train_data
from steps.evaluation import evaluate_model

@pipeline(enable_cache=True)
def trainingpipeline(data_path:str):
    df=ingest_data(data_path)
    clean_data(df)
    train_data(df)
    evaluate_model(df)


