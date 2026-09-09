#librerias a utilizar
import tensorflow as tf
import pandas
import os
from keras import layers, models, Input
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

#semilla para obtener resultados reproducibles
tf.keras.utils.set_random_seed(42)

#hiperparametros
##train_percentage = 0.75  # % de datos a apartar para el entrenamiento
train_percentage = 0.60
validation_percentage = 0.20
test_percentage = 0.20
n = 2  # Número de capas ocultas del modelo 
units = 10  # Número de neuronas por capa 
activation = "tanh"  # Función de activación 
learning_rate = 0.01  # Taza de aprendizaje
optimizer = "Adam" #optimizador
loss = "categorical_crossentropy"  # Función de pérdida 
batch_size = 32  # Tamaño de lote 
epochs = 200  # Número de iteraciones de entrenamiento

# Lectura de los datos
dataset = pandas.read_csv("milknew.csv") 
print(dataset)
##print("\nTotal de filas:")
##print(len(dataset))
##
##print("\nFilas unicas considerando todas las columnas:")
##print(len(dataset.drop_duplicates()))
##
##print("\nFilas unicas considerando solamente las entradas:")
##print(len(dataset.drop_duplicates(
##    subset=[
##        "pH",
##        "Temprature",
##        "Taste",
##        "Odor",
##        "Fat ",
##        "Turbidity",
##        "Colour"
##    ]
##)))
##grade_por_entrada = dataset.groupby(
##    [
##        "pH",
##        "Temprature",
##        "Taste",
##        "Odor",
##        "Fat ",
##        "Turbidity",
##        "Colour"
##    ]
##)["Grade"].nunique()
##
##print("\nEntradas que aparecen asociadas a mas de un Grade:")
##print((grade_por_entrada > 1).sum())
##
##dataset_unico = dataset.drop_duplicates()
##
##print("\nCantidad de muestras unicas:")
##print(len(dataset_unico))
##
##print("\nDistribucion de Grade en las muestras unicas:")
##print(dataset_unico["Grade"].value_counts())
##
##print("\nPorcentaje de Grade en las muestras unicas:")
##print(dataset_unico["Grade"].value_counts(normalize=True) * 100)


print("\nCantidad de valores faltantes por columna:")
print(dataset.isnull().sum())
print("\nCantidad de filas duplicadas:")
print(dataset.duplicated().sum())

#separar entradas x de salidas y
x = dataset.drop(columns="Grade")
y = dataset["Grade"]
print(dataset.columns.tolist())
print("\nEstadisticas del dataset:")
print(x.describe())

###Analisis de posibles outliers
##columnas_continuas = [
##    "pH",
##    "Temprature",
##    "Colour"
##]
##
##for columna in columnas_continuas:
##
##    Q1 = dataset[columna].quantile(0.25)
##    Q3 = dataset[columna].quantile(0.75)
##
##    IQR = Q3 - Q1
##
##    limite_inferior = Q1 - 1.5 * IQR
##    limite_superior = Q3 + 1.5 * IQR
##
##    outliers = dataset[
##        (dataset[columna] < limite_inferior) |
##        (dataset[columna] > limite_superior)
##    ]
##
##    print("\n------------------------------")
##    print("Analisis de:", columna)
##    print("------------------------------")
##    print("Q1:", Q1)
##    print("Q3:", Q3)
##    print("IQR:", IQR)
##    print("Limite inferior:", limite_inferior)
##    print("Limite superior:", limite_superior)
##    print("Cantidad de posibles outliers:", len(outliers))

##observar cuantos valores hay de cada uno de los grados
#print(y.value_counts())
#print(x)
#print(y)

#convertir los grados de la leche de high, medium, low a numeros
y = y.replace({
    "low": 0,
    "medium": 1,
    "high": 2,
})
#print(y)

#agregamos nuevamente la salida al dataset
dataset["Grade"] = y

###Normalizamos las entradas
##max_val = x.max(axis=0)
##min_val = x.min(axis=0)
##difference = max_val - min_val
##
##new_dataset = (x - min_val)/ difference
##new_dataset = new_dataset.astype(float)
##
###print(new_dataset)
##
###entrenamiento y prueba
##trainset = new_dataset.sample(frac=train_percentage, random_state=42)
##testset = new_dataset.drop(trainset.index)
##
##trainset["Grade"] = y.loc[trainset.index]
##testset["Grade"] = y.loc[testset.index]

#print("Datos de entrenamiento:")
#print(trainset)

#print("Datos de prueba:")
#print(testset)

##observar la distribucion de clases en entrenamiento y prueba
#print("\nDistribucion de clases en entrenamiento:")
#print(trainset["Grade"].value_counts())

#print("\nDistribucion de clases en prueba:")
#print(testset["Grade"].value_counts())

##observar la distribucion en porcentaje
#print("\nPorcentaje de clases en entrenamiento:")
#print(trainset["Grade"].value_counts(normalize=True) * 100)

#print("\nPorcentaje de clases en prueba:")
#print(testset["Grade"].value_counts(normalize=True) * 100)

####Preparacion de entradas y salidas
##train_x = trainset.drop(columns="Grade")
##test_x = testset.drop(columns="Grade")
##
##train_y = to_categorical(
##    trainset["Grade"],
##    num_classes=3
##)
##test_y = to_categorical(
##    testset["Grade"],
##    num_classes=3
##)

#Columnas utilizadas como entradas
columnas_entrada = [
    "pH",
    "Temprature",
    "Taste",
    "Odor",
    "Fat ",
    "Turbidity",
    "Colour"
]

#Obtener solamente las combinaciones unicas
dataset_unico = dataset.drop_duplicates(
    subset=columnas_entrada
).copy()

print("\nCantidad de muestras unicas:")
print(len(dataset_unico))

##print("\n================================")
##print("OUTLIERS EN LAS MUESTRAS UNICAS")
##print("================================")
##
##columnas_continuas = [
##    "pH",
##    "Temprature",
##    "Colour"
##]
##
##for columna in columnas_continuas:
##
##    Q1 = dataset_unico[columna].quantile(0.25)
##    Q3 = dataset_unico[columna].quantile(0.75)
##
##    IQR = Q3 - Q1
##
##    limite_inferior = Q1 - 1.5 * IQR
##    limite_superior = Q3 + 1.5 * IQR
##
##    outliers_unicos = dataset_unico[
##        (dataset_unico[columna] < limite_inferior) |
##        (dataset_unico[columna] > limite_superior)
##    ]
##
##    print("\nAnalisis de:", columna)
##    print("Q1:", Q1)
##    print("Q3:", Q3)
##    print("IQR:", IQR)
##    print("Limite inferior:", limite_inferior)
##    print("Limite superior:", limite_superior)
##    print(
##        "Cantidad de outliers unicos:",
##        len(outliers_unicos)
##    )
##
##    print("Valores encontrados:")
##    print(
##        sorted(outliers_unicos[columna].unique())
##    )
##    
#Separar las muestras unicas en entrenamiento y prueba
##train_unico = dataset_unico.sample(
##    frac=train_percentage,
##    random_state=42
##)
##
##test_unico = dataset_unico.drop(train_unico.index)

##train_unico = dataset_unico.groupby(
##    "Grade",
##    group_keys=False
##).sample(
##    frac=train_percentage,
##    random_state=42
##)
##
##test_unico = dataset_unico.drop(train_unico.index)

#Comprobar la distribucion de las muestras unicas
##print("\nDistribucion de las MUESTRAS UNICAS de entrenamiento:")
##print(train_unico["Grade"].value_counts())
##
##print("\nDistribucion de las MUESTRAS UNICAS de prueba:")
##print(test_unico["Grade"].value_counts())

#Tomar solamente las entradas de las muestras unicas
##train_keys = train_unico[columnas_entrada]
##test_keys = test_unico[columnas_entrada]
##
###Recuperar del dataset original todas las repeticiones
###que pertenecen a cada grupo
##trainset = dataset.merge(
##    train_keys,
##    on=columnas_entrada,
##    how="inner"
##)
##
##testset = dataset.merge(
##    test_keys,
##    on=columnas_entrada,
##    how="inner"
##)

#Comprobar como quedo la nueva division
##print("\n------------------------------")
##print("NUEVA DIVISION DEL DATASET")
##print("------------------------------")
##
##print("Filas de entrenamiento:", len(trainset))
##print("Filas de prueba:", len(testset))
##
##print("Muestras unicas de entrenamiento:", len(train_unico))
##print("Muestras unicas de prueba:", len(test_unico))
##
##print("\nDistribucion de clases en entrenamiento:")
##print(trainset["Grade"].value_counts())
##
##print("\nDistribucion de clases en prueba:")
##print(testset["Grade"].value_counts())
##
###Separar entradas y salidas
##train_x = trainset.drop(columns="Grade")
##test_x = testset.drop(columns="Grade")
##
##train_y = trainset["Grade"]
##test_y = testset["Grade"]
##
###Calcular los valores de normalizacion
###utilizando solamente entrenamiento
##max_val = train_x.max(axis=0)
##min_val = train_x.min(axis=0)
##difference = max_val - min_val
##
###Normalizar entrenamiento y prueba
###utilizando los mismos valores de entrenamiento
##train_x = (train_x - min_val) / difference
##test_x = (test_x - min_val) / difference
##
##train_x = train_x.astype(float)
##test_x = test_x.astype(float)
##
###Convertir las salidas a formato one-hot
##train_y = to_categorical(
##    train_y,
##    num_classes=3
##)
##
##test_y = to_categorical(
##    test_y,
##    num_classes=3
##)

#----------------------------------------
# DIVISION TRAIN / VALIDATION / TEST
#----------------------------------------

# Separar primero aproximadamente 20% para TEST
testset = dataset_unico.groupby(
    "Grade",
    group_keys=False
).sample(
    frac=test_percentage,
    random_state=42
)

# El resto queda disponible para Train + Validation
datos_restantes = dataset_unico.drop(
    testset.index
)

# Del 80% restante, tomar 25% para Validation
# 25% de 80% = 20% del total original
validationset = datos_restantes.groupby(
    "Grade",
    group_keys=False
).sample(
    frac=0.25,
    random_state=42
)

# Lo que sobra será Train
trainset = datos_restantes.drop(
    validationset.index
)


#----------------------------------------
# COMPROBACION DE LA DIVISION
#----------------------------------------

print("\n------------------------------")
print("DIVISION TRAIN / VALIDATION / TEST")
print("------------------------------")

print("\nCantidad de muestras:")
print("Train:", len(trainset))
print("Validation:", len(validationset))
print("Test:", len(testset))

print("\nDistribucion TRAIN:")
print(trainset["Grade"].value_counts())

print("\nDistribucion VALIDATION:")
print(validationset["Grade"].value_counts())

print("\nDistribucion TEST:")
print(testset["Grade"].value_counts())


#----------------------------------------
# SEPARAR ENTRADAS Y SALIDAS
#----------------------------------------

train_x = trainset.drop(columns="Grade")
train_y = trainset["Grade"]

validation_x = validationset.drop(columns="Grade")
validation_y = validationset["Grade"]

test_x = testset.drop(columns="Grade")
test_y = testset["Grade"]


#----------------------------------------
# NORMALIZACION
#----------------------------------------

# Los valores de normalizacion se calculan
# solamente con el conjunto de entrenamiento
max_val = train_x.max(axis=0)
min_val = train_x.min(axis=0)

difference = max_val - min_val

# Normalizar los tres conjuntos usando
# los parametros obtenidos de Train
train_x = (train_x - min_val) / difference
validation_x = (validation_x - min_val) / difference
test_x = (test_x - min_val) / difference

train_x = train_x.astype(float)
validation_x = validation_x.astype(float)
test_x = test_x.astype(float)


#----------------------------------------
# CONVERTIR SALIDAS A ONE-HOT
#----------------------------------------

train_y = to_categorical(
    train_y,
    num_classes=3
)

validation_y = to_categorical(
    validation_y,
    num_classes=3
)

test_y = to_categorical(
    test_y,
    num_classes=3
)

#Contrucción de la red neuronal
#si quiero que se actualice automaticamente: network.add(input(shape=(trainset.drop(columns="Grade").shape[1],)))
network = models.Sequential()

#Como son 7 entradas
network.add(Input(shape=(7,)))

#capas ocultas
for i in range(n):
    network.add(layers.Dense(
        units=units,
        activation=activation))

#para las salidas que son 3
network.add(layers.Dense(
    units=3,
    activation="softmax"))

#seleccion del optimizador
if optimizer == "Adam":
    optimizer_model = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

elif optimizer == "SGD":
    optimizer_model = tf.keras.optimizers.SGD(
        learning_rate=learning_rate
    )

elif optimizer == "RMSprop":
    optimizer_model = tf.keras.optimizers.RMSprop(
        learning_rate=learning_rate
    )

elif optimizer == "Adagrad":
    optimizer_model = tf.keras.optimizers.Adagrad(
        learning_rate=learning_rate
    )
    
#compilando la red
network.compile(
    optimizer=optimizer_model,
    loss=loss
)

#entrenamiento
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
##losses = network.fit(
##    x=train_x,
##    y=train_y,
##    validation_data=(
##        test_x,
##        test_y
##    ),
##    batch_size=batch_size,
##    epochs=epochs,
##    #para que no muestre el progreso en pantalla
##    verbose=0
##)

losses = network.fit(
    x=train_x,
    y=train_y,
    validation_data=(validation_x, validation_y),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[early_stopping],
    verbose=0
)

print("\nEpocas realmente ejecutadas:", len(losses.history["loss"]))

#Analisis del Early Stopping
mejor_epoca = losses.history["val_loss"].index(
    min(losses.history["val_loss"])
)

print("\nMejor epoca:", mejor_epoca + 1)
print("Mejor validation loss:", min(losses.history["val_loss"]))
print(
    "Loss en la mejor epoca:",
    losses.history["loss"][mejor_epoca]
)

loss_modelo_restaurado = network.evaluate(
    validation_x,
    validation_y,
    verbose=0
)

print(
    "Validation loss del modelo restaurado:",
    loss_modelo_restaurado
)


#obtener loss final y validation loss final
loss_final = losses.history["loss"][-1]
val_loss_final = losses.history["val_loss"][-1]

#guardar el historial de error
loss_df = pandas.DataFrame(losses.history)

#grafica del error de entrenamiento y validacion
loss_df.loc[:, ["loss", "val_loss"]].plot()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Error de entrenamiento y validacion")
plt.show()

###ejemplo de prediccion
##dato = dataset.sample(n=1)
##
###guardamos el grade real
##grade_real = dato["Grade"].iloc[0]
##
###quitar grade y normalizar los datos
##new_dato = (dato.drop(columns=["Grade"])- min_val)/ difference
##new_dato = new_dato.astype(float)
##
##datoPrueba = new_dato
##
###se hace la prediccion
##prediction = network.predict(datoPrueba)
##
##
###convertir la posicion de la mayor probabilidad en una categoria
##clases = ["low", "medium", "high"]
##
##indice = prediction.argmax(axis=1)[0]
##grade_predicho = clases[indice]
##
###Convertir el grade real a texto
##grade_nombrereal = clases[grade_real]
##
##print("\nDato ingresado:")
##print(dato)
##print("\nProbabilidades:")
##print(prediction)
##print("\nGrade real:")
##print(grade_nombrereal)
##print("\nGrade predicho:")
##print(grade_predicho)
##
###comparacion de valores
##if grade_nombrereal == grade_predicho:
##    print("\nResultado correcto")
##else:
##    print("\nResultado incorrecto")

#Evaluacion del loss con el conjunto de prueba
test_loss = network.evaluate(
    test_x,
    test_y,
    verbose=0
)

#Evaluacion del modelo con todos los datos de prueba
#haciendo predicciones para todo el conjunto de datos de prueba
predicciones = network.predict(test_x, verbose=0)#verbose para no mostrar

#se obtiene la clase con mayor probabilidad
predicciones_clase = predicciones.argmax(axis=1)

#se convierten las salidas reales de one-hot a numero
reales_clase = test_y.argmax(axis=1)

#se cuentan las predicciones correctas
correctas = (predicciones_clase == reales_clase).sum()

#cantidad total de datos de prueba
total = len(reales_clase)

#calculo de exactitud
accuracy = correctas / total

##print("\nEvaluacion general del modelo")
##print("Predicciones correctas:", correctas)
##print("Total de muestras de prueba:", total)
##print("Exactitud:", accuracy)
##print("Exactitud en porcentaje:", accuracy * 100, "%")

#matriz de confusion
matriz = pandas.crosstab(
    reales_clase,
    predicciones_clase,
    rownames=["Grade real"],

    colnames=["Grade predicho"]
)
##
##print("\nMatriz de confusion:")
##print(matriz)

#resultados del experimento
print("\n------------------------------")
print("RESULTADOS DEL EXPERIMENTO")
print("------------------------------")

print("Capas ocultas:", n)
print("Neuronas por capa:", units)
print("Activacion:", activation)
print("Learning rate:", learning_rate)
print("Optimizador:", optimizer)
print("Batch size:", batch_size)
print("Epochs maximas:", epochs)

print("\nTest loss:", test_loss)
##print("\nLoss final:", loss_final)
##print("Validation loss final:", val_loss_final)

print("\nPredicciones correctas:", correctas)
print("Total de muestras de prueba:", total)
print("Exactitud en porcentaje:", accuracy * 100, "%")

print("\nMatriz de confusion:")
print(matriz)





#----------------------------------------------------------
# MODELO FINAL ENTRENADO CON LAS 83 MUESTRAS UNICAS
#----------------------------------------------------------

print("\n======================================")
print("ENTRENAMIENTO DEL MODELO FINAL")
print("======================================")

# Separar entradas y salidas usando todas las muestras unicas
final_x = dataset_unico.drop(columns="Grade")
final_y = dataset_unico["Grade"]

# Calcular normalizacion usando ahora las 83 muestras unicas
max_val_final = final_x.max(axis=0)
min_val_final = final_x.min(axis=0)
difference_final = max_val_final - min_val_final

final_x = (
    final_x - min_val_final
) / difference_final

final_x = final_x.astype(float)

# Convertir Grade a one-hot
final_y = to_categorical(
    final_y,
    num_classes=3
)

# Crear una NUEVA red para el modelo final
final_network = models.Sequential()

final_network.add(
    Input(shape=(7,))
)

# Capas ocultas seleccionadas
for i in range(n):
    final_network.add(
        layers.Dense(
            units=units,
            activation=activation
        )
    )

# Capa de salida
final_network.add(
    layers.Dense(
        units=3,
        activation="softmax"
    )
)

# Optimizador final
optimizer_final = tf.keras.optimizers.Adam(
    learning_rate=learning_rate
)

# Compilar
final_network.compile(
    optimizer=optimizer_final,
    loss=loss
)

# Entrenamiento final
# Usamos 74 epocas porque fue la mejor epoca encontrada
final_network.fit(
    x=final_x,
    y=final_y,
    batch_size=batch_size,
    epochs=74,
    verbose=0
)

print("Modelo final entrenado con:", len(final_x), "muestras unicas")
print("Epocas utilizadas: 74")

#----------------------------------------------------------
# MUESTRAS SOLICITADAS EN LA PRACTICA
#----------------------------------------------------------

muestras_finales = pandas.DataFrame(
    [
        [7.9, 68, 1, 0, 1, 1, 238],
        [5.7, 22, 1, 1, 1, 1, 208],
        [6.8, 81, 0, 0, 1, 1, 257]
    ],
    columns=[
        "pH",
        "Temprature",
        "Taste",
        "Odor",
        "Fat ",
        "Turbidity",
        "Colour"
    ]
)

print("\nMuestras originales:")
print(muestras_finales)

muestras_normalizadas = (
    muestras_finales - min_val_final
) / difference_final

muestras_normalizadas = muestras_normalizadas.astype(float)

predicciones_finales = final_network.predict(
    muestras_normalizadas,
    verbose=0
)

clases = ["low", "medium", "high"]

print("\n======================================")
print("PREDICCIONES DE LAS MUESTRAS FINALES")
print("======================================")

for i in range(len(muestras_finales)):

    indice = predicciones_finales[i].argmax()

    print("\nMuestra", i + 1)

    print(
        "Probabilidades:",
        predicciones_finales[i]
    )

    print(
        "Grade predicho:",
        clases[indice]
    )
