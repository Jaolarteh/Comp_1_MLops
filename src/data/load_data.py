import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# Testing2
import argparse
import wandb
import os

parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")

def load(train_size=.8):
    """
    # Load the data
    """
      
    # the data, split between train and test sets

    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path.split('src')[0],'src', 'raw_data/loanpred_train.csv')
    train = pd.read_csv(file_path)

    train_set, val_set= train_test_split(
        train,
        test_size= 1 - train_size, random_state=42,
        stratify= train['Loan_Status'].map({'Y': 1, 'N': 0})
        )
    file_path = os.path.join(base_path.split('src')[0],'src', 'raw_data/loanpred_test.csv')
    test_set = pd.read_csv(file_path)

    datasets = [train_set, val_set, test_set]

    return datasets

def load_and_log():
    # 🚀 start a run, with a type to label it and a project it can call home
    with wandb.init(
        project="MLOps-ClassAssigment",
        name=f"Load Raw Data ExecId-{args.IdExecution}", job_type="load-data") as run:
        
        datasets = load()  # separate code for loading the datasets
        names = ["Train","Val", "Test"]

        # 🏺 create our Artifact
        raw_data = wandb.Artifact(
            "data-", type="dataset",
            description="raw Loan prediction dataset, split into train/val/test",
            metadata={"source": "Assigned in class proyect",
                      "sizes": [len(dataset) for dataset in datasets]})


        os.makedirs("temp_data", exist_ok=True)

        for name, df in zip(names, datasets):
            file_path = f"temp_data/{name}.csv"
            df.to_csv(file_path, index=False)
            raw_data.add_file(file_path, name=f"{name}.csv")

        run.log_artifact(raw_data)

        # Limpieza opcional
        for name in names:
            os.remove(f"temp_data/{name}.csv")
        os.rmdir("temp_data")


# testing
load_and_log()
