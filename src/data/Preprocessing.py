import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
import pickle
import io
import tempfile

#testing
#
import os
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")
else:
    args.IdExecution = "testing console"

categorical_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
target = 'Loan_Status'

def recover_dataframe(dataframe):

    feature_names = numerical_features + categorical_features
    return(pd.DataFrame(dataframe, columns=feature_names))


def pipeline_training(Train_dataset):    

    data_pipeline = ColumnTransformer([
        ('num_imputer', SimpleImputer(strategy='median'), numerical_features),
        ('cat_imputer', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder())
        ]), categorical_features)
    ])

    return(data_pipeline.fit(Train_dataset.drop(columns=[target])))


def preprocessing_and_log():

    with wandb.init(project="MLOps-ClassAssigment",name=f"Preprocess Data ExecId-{args.IdExecution}", job_type="preprocess-data") as run:
        
        artifact = run.use_artifact('MLOps-ClassAssigment/data-:latest', type='dataset')
        artifact_dir = artifact.download()


        train_df = pd.read_csv(f"{artifact_dir}/Train.csv")
        val_df = pd.read_csv(f"{artifact_dir}/Val.csv")
        test_df = pd.read_csv(f"{artifact_dir}/Test.csv")

        Preprocess = pipeline_training(train_df)



        X_train = Preprocess.transform(train_df.drop(columns=[target]))
        X_train[target] = np.where(train_df[target] == 'Y', 1, 0)
        X_val = Preprocess.transform(val_df.drop(columns=[target]))
        X_val[target] = np.where(val_df[target] == 'Y', 1, 0)
        X_test = Preprocess.transform(test_df)

        print(X_train.head(5))

        feature_names = numerical_features + categorical_features
        artifact = wandb.Artifact(
            "preprocessed-data", type="dataset",
            description="Train/Val/Test preprocesados",
            metadata={"features": feature_names, "target": target}
        )

        # Crear datasets y nombres
        datasets = [X_train, X_val, X_test]
        names = ["X_train", "X_val", "X_test"]

        # Crear carpeta temporal
        os.makedirs("temp_data", exist_ok=True)

        for name, df in zip(names, datasets):
            file_path = f"temp_data/{name}.csv"
            df.to_csv(file_path, index=False)
            artifact.add_file(file_path, name=f"{name}.csv")

        # Loggear artefacto
        run.log_artifact(artifact)

        # Limpieza opcional
        for name in names:
            os.remove(f"temp_data/{name}.csv")
        os.rmdir("temp_data")
