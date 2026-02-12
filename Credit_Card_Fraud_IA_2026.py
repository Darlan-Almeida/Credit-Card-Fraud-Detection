
import numpy as np
import pandas as pd

from datetime import datetime

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = "creditcard.csv"

df = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "mlg-ulb/creditcardfraud",
  file_path,
)

print(df.head())

X = df.drop("Class", axis=1).values
y = df["Class"].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)


models = {
    "RandomForest": (
        RandomForestClassifier(class_weight="balanced", random_state=42),
        {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
        },
    ),

    "ExtraTrees": (
        ExtraTreesClassifier(class_weight="balanced", random_state=42),
        {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
        },
    ),

    "GradientBoosting": (
        GradientBoostingClassifier(random_state=42),
        {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5],
        },
    ),

    "SVM": (
        SVC(class_weight="balanced"),
        {
            "C": [0.1, 1, 10],
            "kernel": ["rbf"],
            "gamma": ["scale", "auto"],
        },
    ),

    "LogisticRegression": (
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        {
            "C": [0.1, 1, 10],
        },
    ),

    "KNN": (
        KNeighborsClassifier(),
        {
            "n_neighbors": [3, 5, 7],
        },
    ),

    "MLP": (
        MLPClassifier(max_iter=50, random_state=42),
        {
            "hidden_layer_sizes": [(64,), (128, 64)],
            "alpha": [0.0001, 0.001],
        },
    ),
}


def evaluate_model(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    return classification_report(y_test, pred, output_dict=True)


print("\n================ BASELINE ================\n")

classic_results = []

for name, (model, params) in models.items():
    for p in ParameterGrid(params):
        clf = model.set_params(**p)
        metrics = evaluate_model(clf, X_train, X_test, y_train, y_test)

        classic_results.append({
            "modelo": name,
            "params": str(p),
            **metrics
        })

        print(name, p, metrics)

classic_results_df = pd.DataFrame(classic_results)

# Salvando resultados
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

classic_results_df.to_csv(f"baseline_results_{timestamp}.csv", index=False)


input_dim = X_train.shape[1]

print("\n================ AUTOENCODER ================\n")


architectures = [
    {
        "enc1": 128, "enc2": 64, "bottleneck": 8,
        "fit": {"epochs": 20, "batch_size": 256, "validation_split": 0.1, "verbose": 0}
    },
    {
        "enc1": 64, "enc2": 32, "bottleneck": 8,
        "fit": {"epochs": 20, "batch_size": 256, "validation_split": 0.1, "verbose": 0}
    },
    {
        "enc1": 128, "enc2": 64, "bottleneck": 16,
        "fit": {"epochs": 20, "batch_size": 256, "validation_split": 0.1, "verbose": 0}
    },
]

ae_results = []


for arch in architectures:
    print(f"\nTreinando arquitetura: enc1={arch['enc1']} "
          f"enc2={arch['enc2']} bottleneck={arch['bottleneck']}")

    input_layer = keras.Input(shape=(input_dim,))

    # Encoder
    encoded = layers.Dense(arch["enc1"], activation="relu")(input_layer)
    encoded = layers.Dense(arch["enc2"], activation="relu")(encoded)
    bottleneck = layers.Dense(arch["bottleneck"], activation="relu")(encoded)

    # Decoder
    decoded = layers.Dense(arch["enc2"], activation="relu")(bottleneck)
    decoded = layers.Dense(arch["enc1"], activation="relu")(decoded)
    output_layer = layers.Dense(input_dim, activation="linear")(decoded)

    autoencoder = keras.Model(input_layer, output_layer)
    autoencoder.compile(optimizer="adam", loss="mse")

    
    history = autoencoder.fit(
        X_train, X_train,
        **arch["fit"]
    )

    best_val_loss = min(history.history["val_loss"])
    last_train_loss = history.history["loss"][-1]

    encoder = keras.Model(input_layer, bottleneck)

    X_train_enc = encoder.predict(X_train, verbose=0)
    X_test_enc = encoder.predict(X_test, verbose=0)


    for name, (model, params) in models.items():
        for p in ParameterGrid(params):
            clf = model.set_params(**p)

            metrics = evaluate_model(
                clf, X_train_enc, X_test_enc, y_train, y_test
            )

            ae_results.append({
                "enc1": arch["enc1"],
                "enc2": arch["enc2"],
                "bottleneck": arch["bottleneck"],
                "fit_params": str(arch["fit"]),
                "best_val_loss": best_val_loss,
                "last_train_loss": last_train_loss,
                "modelo": name,
                "params": str(p),
                **metrics
            })

            print(name, p, metrics)



ae_results_df = pd.DataFrame(ae_results)


# Salvando resultados
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
ae_results_df.to_csv(f"autoencoder_results_{timestamp}.csv", index=False)

print("\nArquivos salvos:")
print(f"baseline_results_{timestamp}.csv")
print(f"autoencoder_results_{timestamp}.csv")
