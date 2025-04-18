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

# Preprocessing.py

# ... (todo tu código anterior igual)

def preprocessing_and_log():
    with wandb.init(project="MLOps-ClassAssigment", name=f"Preprocess Data ExecId-{args.IdExecution}", job_type="preprocess-data") as run:
        try:
            print("Buscando artifact de datos crudos...")
            artifact = run.use_artifact('jaolarteh-universidad-eafit/MLOps-ClassAssigment/data-:v0', type='dataset')
            artifact_dir = artifact.download()
            print(f"Artifact descargado en: {artifact_dir}")
        except Exception as e:
            print(f"Error al usar artifact: {e}")
            return  # Termina si falla

        try:
            train_df = pd.read_csv(f"{artifact_dir}/Train.csv")
            val_df = pd.read_csv(f"{artifact_dir}/Val.csv")
            test_df = pd.read_csv(f"{artifact_dir}/Test.csv")
            print("Datos cargados exitosamente.")
        except Exception as e:
            print(f"Error al cargar datasets: {e}")
            return

        Preprocess = pipeline_training(train_df)
        print("Pipeline de preprocesamiento entrenado.")

        # Transformación
        X_train = Preprocess.transform(train_df.drop(columns=[target]))
        X_train = recover_dataframe(X_train)
        X_train[target] = np.where(train_df[target] == 'Y', 1, 0)

        X_val = Preprocess.transform(val_df.drop(columns=[target]))
        X_val = recover_dataframe(X_val)
        X_val[target] = np.where(val_df[target] == 'Y', 1, 0)

        X_test = Preprocess.transform(test_df)
        X_test = recover_dataframe(X_test)

        print("Datos transformados.")

        # Crear artifact
        artifact = wandb.Artifact(
            "preprocessed-data", type="dataset",
            description="Train/Val/Test preprocesados",
            metadata={"features": numerical_features + categorical_features, "target": target}
        )

        os.makedirs("temp_data", exist_ok=True)

        datasets = [X_train, X_val, X_test]
        names = ["X_train", "X_val", "X_test"]

        for name, df in zip(names, datasets):
            file_path = f"temp_data/{name}.csv"
            df.to_csv(file_path, index=False)
            artifact.add_file(file_path, name=f"{name}.csv")

        run.log_artifact(artifact)
        print("Artifact loggeado exitosamente.")

        # Limpieza
        for name in names:
            os.remove(f"temp_data/{name}.csv")
        os.rmdir("temp_data")
        print("Archivos temporales eliminados.")

if __name__ == "__main__":
    preprocessing_and_log()
