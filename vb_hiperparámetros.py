import tensorflow as tf
import pandas as pd
from keras import layers, models, Input
import matplotlib.pyplot as plt
import itertools

# Lectura y normalización de datos
dataset = pd.read_csv("winequality-white.csv", sep=";")
max_val = dataset.max(axis=0) # Se obtiene el máximo de cada columna
min_val = dataset.min(axis=0)# Se obtiene el mínimo de cada columna
difference = max_val - min_val # Se obtiene la diferencia de los dos
new_dataset = (dataset - min_val) / difference # Y se utiliza para normalizarlas

# Variables de entrada
features = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
]

# Hiperparámetros (Genera 36 combinaciones: 2 x 3 x 2 x 3)
units_list = [5, 2]
lr_list = [0.01, 0.001, 0.5]
train_pct_list = [0.80, 0.55]
epochs_list = [10, 6, 2]

combinaciones = list(itertools.product(units_list, lr_list, train_pct_list, epochs_list))
resultados = []

# Ciclo for para evaluar cada combinación
for idx, (units, lr, train_pct, epochs) in enumerate(combinaciones):

    # División en entrenamiento y validación
    trainset = new_dataset.sample(frac=train_pct, random_state=42) ## Se extraen datos al azar del conjunto para el entrenamiento
    testset = new_dataset.drop(trainset.index) #= # Y se le quitan esos mismos
# al dataset para crear los datos de prueba

    # Inicialización del modelo
    network = models.Sequential()

    # Capa de entrada: 11 variables simultáneamente
    network.add(Input(shape=(11,)))
    
    # 2 Capas ocultas (el original n=3 iteraba n-1 veces)
    for _ in range(2):
        network.add(layers.Dense(
            units=units,
            activation="relu"   #no hay datos negativos, por lo que no se necesita la función de activación
                                #tangente hiperbólica
            )
        ) 

    # Capa de salida: una única variable (quality)
    network.add(layers.Dense(
        units=1,
        activation=None
        )
    )

    # Compilación
    network.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=lr),
        loss="mse",
        metrics=["mae"] #error absoluto promedio sobre entrenamiento
    )

    # Entrenamiento (verbose=0 para limpiar la terminal)

    print(f"Entrenando config {idx+1}/36: Neuronas={units}, LR={lr}, Train={train_pct*100}%, Epochs={epochs}")
    history = network.fit(
        x=trainset[features],
        y=trainset["quality"],
        validation_data=(
            testset[features],
            testset["quality"]
        ),
        batch_size=250,
        epochs=epochs,
        verbose=0
    )
    
    # Almacenamiento de resultados
    resultados.append({
        'idx': idx + 1,
        'label': f"N:{units}|LR:{lr}|Tr:{train_pct}|Ep:{epochs}",
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss']
    })

# Graficación de resultados en matrices 4x4
plots_per_image = 16
for i in range(0, len(resultados), plots_per_image):
    chunk = resultados[i:i + plots_per_image]
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    fig.canvas.manager.set_window_title(f'Gráficas {i+1} a {i+len(chunk)}')
    axes = axes.flatten()
    
    for j, res in enumerate(chunk):
        ax = axes[j]
        ax.plot(res['loss'], label='Entrenamiento (loss)')
        ax.plot(res['val_loss'], label='Validación (val_loss)')
        ax.set_title(f"#{res['idx']} {res['label']}", fontsize=9)
        ax.set_xlabel("Epochs", fontsize=7)
        ax.set_ylabel("MSE", fontsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, linestyle='--', alpha=0.5)
    
    # Eliminar recuadros vacíos si la matriz no se llena
    for j in range(len(chunk), 16):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plt.show()