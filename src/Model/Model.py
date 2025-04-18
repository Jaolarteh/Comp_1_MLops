import pandas as pd
import numpy as np
import argparse
import wandb
import os
import joblib
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")

# Función de optimización
def objective(trial):
    global X_train, y_train
    model_name = trial.suggest_categorical("model", ["xgb", "rf"])
    
    if model_name == "xgb":
        model = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            gamma=trial.suggest_float("gamma", 0, 5),
            reg_alpha=trial.suggest_float("reg_alpha", 0, 10),
            reg_lambda=trial.suggest_float("reg_lambda", 0, 10),
            use_label_encoder=False,
            eval_metric='logloss'
        )
    else:
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            bootstrap=trial.suggest_categorical("bootstrap", [True, False])
        )
    
    # Validación cruzada
    score = cross_val_score(model, X_train, y_train, scoring=make_scorer(f1_score), cv=5).mean()

    # Loggear a wandb
    wandb.log({
        "trial": trial.number,
        "model": model_name,
        "f1_score": score,
        **trial.params  # Para ver los hiperparámetros de cada trial
    })

    return score


def search_and_train():
    global X_train, y_train  

    with wandb.init(
        project="MLOps-ClassAssigment",
        name=f"Model training ExecId-{args.IdExecution}", job_type="model-selection") as run:  

        # Descargar artifact de datos preprocesados
        artifact = run.use_artifact('preprocessed-data:latest', type='dataset')
        artifact_dir = artifact.download()

        # Cargar datasets
        X_train = pd.read_csv(f"{artifact_dir}/X_train.csv")
        X_val = pd.read_csv(f"{artifact_dir}/X_val.csv")
        X_test = pd.read_csv(f"{artifact_dir}/X_test.csv")

        # Separar target
        y_train = X_train.pop('Loan_Status')
        y_val = X_val.pop('Loan_Status')

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)

        print("Best trial:")
        print(study.best_trial.params)


        # Reconstruir el mejor modelo
        best_params = study.best_params
        model_name = best_params.pop("model")

        if model_name == "xgb":
            best_model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
        else:
            best_model = RandomForestClassifier(**best_params)

        # Entrenar
        best_model.fit(X_train, y_train)

        # Validar
        val_preds = best_model.predict(X_val)
        val_f1 = f1_score(y_val, val_preds)
        print(f"Validation F1 Score: {val_f1}")

        # Guardar modelo en archivo
        os.makedirs("models", exist_ok=True)
        model_path = "models/best_model.pkl"
        joblib.dump(best_model, model_path)

        # Crear y subir artifact
        model_artifact = wandb.Artifact(
            "best-model", type="model",
            description="Best model found by Optuna",
            metadata=study.best_trial.params
        )
        model_artifact.add_file(model_path)

        run.log_artifact(model_artifact)

        # Limpieza
        os.remove(model_path)
        os.rmdir("models")