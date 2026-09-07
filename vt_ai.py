import tensorflow as tf
import pandas
from keras import layers, models, Input
import matplotlib.pyplot as plt

# Hiperparámetros
train_percentage = 0.8  # % de datos a apartar para el entrenamiento
                        #esto genera un subconjunto de 80% para entrenar
                        #y 20% para probar el modelo
n = 3  # Número de capas ocultas del modelo
units = 2  # Número de neuronas por capa
activation = "relu"  # Función de activación= ignora valorse negativos
learning_rate = 0.5  # Taza de aprendizaje
epochs = 6  # Número de iteraciones de entrenamiento


loss = "mse"  # Función de pérdida
batch_size = 250  # Tamaño de lote = cuántos ejemplos del conjunto de
                  #entrenamiento utiliza la red neuronal antes de actualizar sus pesos.


# Lectura de los datos
dataset = pandas.read_csv("winequality-red.csv", sep=";")
print(dataset)
print(dataset.isnull().sum())

# Calcular los cuartiles
Q1 = dataset.quantile(0.25)
Q3 = dataset.quantile(0.75)
IQR = Q3 - Q1

# Definir los límites estadísticos
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

# Contar cuántos datos reales se salen de esos límites
outliers_reales = ((dataset < limite_inferior) | (dataset > limite_superior)).sum()

print("\n--- Conteo real de Outliers por IQR ---")
print(outliers_reales)

#Eliminando los outliers
# 1. Crear una condición que busque si ALGUNA columna de la fila (.any) es un outlier
condicion_outliers = ((dataset < limite_inferior) | (dataset > limite_superior)).any(axis=1)

# 2. Extraer los números de fila (índices) que cumplen esa condición
indices_outliers = dataset[condicion_outliers].index

# 3. Aplicar la función .drop usando los índices encontrados
dataset = dataset.drop(index=indices_outliers)

# Normalizado
max_val = dataset.max(axis=0)  # Se obtiene el máximo de cada columna
min_val = dataset.min(axis=0)  # Se obtiene el mínimo de cada columna
difference = max_val - min_val  # Se obtiene la diferencia de los dos
new_dataset = (dataset - min_val)/(difference)  # Y se utiliza para normalizarlas

# División en entrenamiento y validación.
trainset = new_dataset.sample(frac=train_percentage)  # ATENCIÓN: HIPERPARÁMETRO
# Se extraen datos para el entrenamiento

testset = new_dataset.drop(trainset.index)  # Y se le quitan esos mismos
# al dataset para crear los datos de prueba
print(trainset) #==>imprime el 80% de los datos del dataset ya normalizados, que se utilizarán para entrenar el modelo

import tensorflow as tf


# Variables de entrada
features = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]

# Inicialización del modelo
network = models.Sequential()

# Capa de entrada: 11 variables simultáneamente
network.add(Input(shape=(11,)))

# Capas intermedias
for i in range(n - 1):
    network.add(layers.Dense(
            units=units,
            activation=activation
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
        learning_rate=learning_rate
    ),
    loss=loss,
    metrics=["mae"] #error absoluto promedio sobre entrenamiento
)

# Entrenamiento
losses = network.fit(x=trainset[features],
                    y=trainset["quality"],
                    validation_data=(
                        testset[features],
                        testset["quality"]
                    ),
                    batch_size=batch_size,
                    epochs=epochs
                    )


# Se extrae el historial de error contra iteraciones de la clase
loss_df = pandas.DataFrame(losses.history)

loss_df.loc[:, ['loss', 'val_loss']].plot() # Se crea la curva a graficar

# Y se llama a la ventana que se muestre la grafica
plt.show()

# --- EVALUACIÓN DE LAS 6 MUESTRAS DEL LABORATORIO ---

# 1. Se crea un diccionario con los datos exactos de las 6 muestras de la tabla
datos_evaluacion = {
    "fixed acidity": [5.9, 8.7, 7.2, 6.3, 9.9, 5.9],
    "volatile acidity": [0.66, 0.16, 0.64, 0.56, 0.26, 0.77],
    "citric acid": [0.58, 0.26, 0.42, 0.18, 0.02, 0.12],
    "residual sugar": [2.3, 9.3, 6.3, 5.3, 6.9, 7.2],
    "chlorides": [0.077, 0.025, 0.056, 1.77, 0.024, 0.112],
    "free sulfur dioxide": [12.0, 34.0, 22.0, 23.0, 51.0, 67.0],
    "total sulfur dioxide": [97.0, 186.0, 97.0, 132.0, 186.0, 170.0],
    "density": [0.9733, 0.9933, 0.9018, 0.8933, 0.8733, 0.8841],
    "pH": [4.2, 3.2, 1.2, 8.2, 2.3, 6.5],
    "sulphates": [0.45, 0.49, 0.44, 0.40, 0.40, 0.44],
    "alcohol": [10.2, 12.2, 8.2, 11.7, 9.2, 14.5]
}

# 2. Se convierte en un DataFrame de Pandas
nombres_muestras = ["Muestra 1 (VIB)", "Muestra 2 (VIB)", "Muestra 3 (VIB)", 
                    "Muestra 4 (VIT)", "Muestra 5 (VIT)", "Muestra 6 (VIT)"]
muestras_df = pandas.DataFrame(datos_evaluacion, index=nombres_muestras)

# 3. Se normalizan las muestras usando los valores min y max del dataset de entrenamiento (VIB)
muestras_norm = (muestras_df - min_val[features]) / difference[features]
muestras_norm = muestras_norm.astype(float)

# 4. Se realiza la predicción con el modelo entrenado
prediccion_norm = network.predict(muestras_norm)

# 5. Se desnormaliza el resultado para obtener la escala de calidad original
calidad_estimada = prediccion_norm.flatten() * difference["quality"] + min_val["quality"]

# 6. Se integran los resultados y se muestran en la terminal
muestras_df["Calidad Estimada"] = calidad_estimada.round(2)

print("\n--- RESULTADOS DE EVALUACIÓN (MODELO VINO TINTO) ---")
print(muestras_df[["Calidad Estimada"]])


#Sensibilidad de variables (Ceteris Paribus)
print("\nIniciando estudio de sensibilidad Ceteris Paribus...")

# 1. Definir el "Vino Estándar" (Mediana), Q1 y Q3 con el dataset limpio
mediana = dataset[features].median()
q1 = dataset[features].quantile(0.25)
q3 = dataset[features].quantile(0.75)

#esta definicion anterior Calcula la mediana exacta de las 11 variables
#químicas de tu dataset ya limpio, por lo que en lugar de tener 11 columnas
#y 50000 filas, ahora tienes 11 columnas y 1 fila (la mediana de cada variable)


# 2. Crear las 22 muestras sintéticas (11 variables x 2 extremos)
variaciones = []
nombres_variaciones = []

for feature in features:
    # Ceteris paribus: Copiar el vino estándar (10 variables constantes)
    vino_q1 = mediana.copy()
    vino_q3 = mediana.copy()
    
    # Modificar SOLO la variable actual a sus extremos
    vino_q1[feature] = q1[feature]
    vino_q3[feature] = q3[feature]
    
    variaciones.append(vino_q1)
    variaciones.append(vino_q3)
    nombres_variaciones.extend([f"{feature} (Q1)", f"{feature} (Q3)"])

# Convertir a DataFrame de pandas
df_ceteris = pandas.DataFrame(variaciones, index=nombres_variaciones)

# 3. Normalizar, predecir y desnormalizar usando los hiperparámetros ya entrenados
df_ceteris_norm = (df_ceteris - min_val[features]) / difference[features]
df_ceteris_norm = df_ceteris_norm.astype(float)

# Predecir usando la red ya entrenada
preds_norm = network.predict(df_ceteris_norm, verbose=0)
preds_reales = preds_norm.flatten() * difference["quality"] + min_val["quality"]

# 4. Calcular el impacto de cada variable
resultados_sensibilidad = []

# Iteramos de 2 en 2 porque guardamos Q1 y Q3 seguidos
for i, feature in enumerate(features):
    pred_q1 = preds_reales[2*i]
    pred_q3 = preds_reales[2*i + 1]
    
    # El impacto absoluto mide cuánto cambió la calidad al mover la variable
    impacto = abs(pred_q3 - pred_q1)
    
    resultados_sensibilidad.append({
        "Variable": feature, 
        "Calidad_Q1": round(pred_q1, 3),
        "Calidad_Q3": round(pred_q3, 3),
        "Impacto_Absoluto": round(impacto, 3)
    })

df_sensibilidad = pandas.DataFrame(resultados_sensibilidad)

# 5. Clasificar en "Muy Sensible" y "Poco Sensible"

umbral_absoluto = 2.5

df_sensibilidad["Clasificación"] = df_sensibilidad["Impacto_Absoluto"].apply(
    lambda x: "Muy Sensible" if x >= umbral_absoluto else "Poco Sensible"
)

# Ordenar de mayor a menor impacto para visualizar mejor en la terminal
df_sensibilidad = df_sensibilidad.sort_values(by="Impacto_Absoluto", ascending=False).reset_index(drop=True)

print("\n--- RESULTADOS: SENSIBILIDAD DE VARIABLES ---")
print(df_sensibilidad.to_string())
