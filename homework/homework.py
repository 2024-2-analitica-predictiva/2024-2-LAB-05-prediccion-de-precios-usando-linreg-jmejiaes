#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#



import pandas as pd
import numpy as np
import os
import json
import gzip
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error, mean_absolute_error

# Paso 1: Preprocesar los datos
def preprocess_data(train, test, reference_year=2021):
    train['Age'] = reference_year - train['Year']
    test['Age'] = reference_year - test['Year']
    
    train.drop(columns=["Year", "Car_Name"], inplace=True)
    test.drop(columns=["Year", "Car_Name"], inplace=True)
    
    x_train = train.drop(columns=['Present_Price'])
    y_train = train['Present_Price']
    
    x_test = test.drop(columns=['Present_Price'])
    y_test = test['Present_Price']
    
    return x_train, y_train, x_test, y_test

# Paso 2: Definir y entrenar el modelo
def create_pipeline():
    categoricas = ['Fuel_Type', 'Selling_type', 'Transmission']
    numericas = ['Selling_Price', 'Driven_kms', 'Owner', 'Age']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'), categoricas),
            ('scaler', MinMaxScaler(), numericas),
        ]
    )
    
    selectkbest = SelectKBest(score_func=f_regression, k=10)
    LinR = LinearRegression()

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', selectkbest),
        ('classifier', LinR)
    ])
    
    return pipeline

# Paso 3: Ajuste de hiperparámetros
def optimize_model(pipeline, x_train, y_train):
    CVkf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    param_grid = {
        'feature_selection__k': [5, 10, 'all'],
        'classifier__fit_intercept': [True, False]
    }

    model = GridSearchCV(
        pipeline,
        param_grid,
        cv=CVkf,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        refit=True
    )
    
    model.fit(x_train, y_train)
    return model

# Paso 4: Evaluación del modelo
def evaluate_model(model, x_train, y_train, x_test, y_test):
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Métricas entrenamiento
    r2_train = r2_score(y_train, y_train_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mad_train = median_absolute_error(y_train, y_train_pred)

    # Métricas prueba
    r2_test = r2_score(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    mad_test = median_absolute_error(y_test, y_test_pred)
    
    return r2_train, mse_train, mad_train, r2_test, mse_test, mad_test

# Paso 5: Guardar el modelo
def save_model(model, filepath='../files/models/model.pkl.gz'):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with gzip.open(filepath, "wb") as file:
        pickle.dump(model, file)

# Paso 6: Guardar las métricas
def save_metrics(metrics, filepath='../files/output/metrics.json'):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        for metric in metrics:
            f.write(json.dumps(metric, ensure_ascii=False))
            f.write('\n')

def main():
    # Leer los datasets
    train = pd.read_csv("../files/input/train_data.csv.zip", index_col=False, compression="zip")
    test = pd.read_csv("../files/input/test_data.csv.zip", index_col=False, compression="zip")
    
    # Paso 1: Preprocesar los datos
    x_train, y_train, x_test, y_test = preprocess_data(train, test)

    # Paso 2: Crear el pipeline
    pipeline = create_pipeline()

    # Paso 3: Optimizar el modelo
    model = optimize_model(pipeline, x_train, y_train)

    # Paso 4: Evaluar el modelo
    r2_train, mse_train, mad_train, r2_test, mse_test, mad_test = evaluate_model(model, x_train, y_train, x_test, y_test)

    # Mostrar resultados
    print(f"Mejores parámetros: {model.best_params_}")
    print(f"Mejor MAE (validación cruzada): {-model.best_score_:.4f}")
    print(f"R2 (Entrenamiento): {r2_train}")
    print(f"MSE (Entrenamiento): {mse_train}")
    print(f"MAD (Entrenamiento): {mad_train}")
    print(f"R2 (Prueba): {r2_test}")
    print(f"MSE (Prueba): {mse_test}")
    print(f"MAD (Prueba): {mad_test}")

    # Paso 5: Guardar el modelo
    save_model(model)

    # Paso 6: Guardar las métricas
    metrics = [
        {'type': 'metrics', 'dataset': 'train', 'r2': r2_train, 'mse': mse_train, 'mad': mad_train},
        {'type': 'metrics', 'dataset': 'test', 'r2': r2_test, 'mse': mse_test, 'mad': mad_test},
    ]
    save_metrics(metrics)

if __name__ == '__main__':
    main()
