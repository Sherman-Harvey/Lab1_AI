#==========================================================
# PRACTICA 1 - CLASIFICACION DE CALIDAD DE LECHE
# CODIGO FINAL
#==========================================================


#----------------------------------------------------------
# LIBRERIAS
#----------------------------------------------------------

import tensorflow as tf
import pandas
import matplotlib.pyplot as plt

from keras import layers, models, Input
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping


#----------------------------------------------------------
# SEMILLA PARA OBTENER RESULTADOS REPRODUCIBLES
#----------------------------------------------------------

tf.keras.utils.set_random_seed(42)


#----------------------------------------------------------
# HIPERPARAMETROS SELECCIONADOS
#----------------------------------------------------------

train_percentage = 0.60
validation_percentage = 0.20
test_percentage = 0.20

n = 2
units = 10
activation = "tanh"
learning_rate = 0.01
optimizer = "Adam"
loss = "categorical_crossentropy"
batch_size = 32

# Cantidad maxima de epocas.
# EarlyStopping detendra el entrenamiento antes si es necesario.
epochs = 200


#==========================================================
# LECTURA Y PREPARACION DE LOS DATOS
#==========================================================

dataset = pandas.read_csv("milknew.csv")

print("\n======================================")
print("INFORMACION DEL DATASET")
print("======================================")

print("Cantidad total de filas:", len(dataset))

print("\nValores faltantes:")
print(dataset.isnull().sum())

print("\nFilas duplicadas:")
print(dataset.duplicated().sum())


#----------------------------------------------------------
# CONVERSION DE GRADE A NUMEROS
#----------------------------------------------------------

dataset["Grade"] = dataset["Grade"].replace({
    "low": 0,
    "medium": 1,
    "high": 2
})


#----------------------------------------------------------
# COLUMNAS DE ENTRADA
#----------------------------------------------------------

columnas_entrada = [
    "pH",
    "Temprature",
    "Taste",
    "Odor",
    "Fat ",
    "Turbidity",
    "Colour"
]


#----------------------------------------------------------
# OBTENER SOLAMENTE LAS COMBINACIONES UNICAS
#----------------------------------------------------------

dataset_unico = dataset.drop_duplicates(
    subset=columnas_entrada
).copy()

print("\nCantidad de muestras unicas:")
print(len(dataset_unico))

print("\nDistribucion de Grade en las muestras unicas:")
print(dataset_unico["Grade"].value_counts().sort_index())

#==========================================================
# ANALISIS DE VALORES ATIPICOS MEDIANTE IQR
#==========================================================

columnas_continuas = [
    "pH",
    "Temprature",
    "Colour"
]

Q1 = dataset_unico[columnas_continuas].quantile(0.25)
Q3 = dataset_unico[columnas_continuas].quantile(0.75)

IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

print("\n======================================")
print("ANALISIS DE VALORES ATIPICOS")
print("======================================")

for columna in columnas_continuas:

    outliers = dataset_unico[
        (dataset_unico[columna] < limite_inferior[columna]) |
        (dataset_unico[columna] > limite_superior[columna])
    ]

    print("\nVariable:", columna)
    print("Q1:", Q1[columna])
    print("Q3:", Q3[columna])
    print("IQR:", IQR[columna])
    print("Limite inferior:", limite_inferior[columna])
    print("Limite superior:", limite_superior[columna])
    print("Cantidad de valores atipicos:", len(outliers))
    
#==========================================================
# DIVISION TRAIN / VALIDATION / TEST
#==========================================================

# Primero se separa aproximadamente 20 % para Test,
# manteniendo representadas las tres clases.

testset = dataset_unico.groupby(
    "Grade",
    group_keys=False
).sample(
    frac=test_percentage,
    random_state=42
)

datos_restantes = dataset_unico.drop(testset.index)


# Del 80 % restante se toma el 25 %.
# 25 % de 80 % = 20 % del total original.

validation_fraction = (
    validation_percentage /
    (train_percentage + validation_percentage)
)

validationset = datos_restantes.groupby(
    "Grade",
    group_keys=False
).sample(
    frac=validation_fraction,
    random_state=42
)

trainset = datos_restantes.drop(validationset.index)


print("\n======================================")
print("DIVISION DE DATOS")
print("======================================")

print("Train:", len(trainset))
print("Validation:", len(validationset))
print("Test:", len(testset))

print("\nDistribucion Train:")
print(trainset["Grade"].value_counts().sort_index())

print("\nDistribucion Validation:")
print(validationset["Grade"].value_counts().sort_index())

print("\nDistribucion Test:")
print(testset["Grade"].value_counts().sort_index())


#==========================================================
# SEPARAR ENTRADAS Y SALIDAS
#==========================================================

train_x = trainset[columnas_entrada].copy()
train_y = trainset["Grade"].copy()

validation_x = validationset[columnas_entrada].copy()
validation_y = validationset["Grade"].copy()

test_x = testset[columnas_entrada].copy()
test_y = testset["Grade"].copy()


#==========================================================
# NORMALIZACION
#==========================================================

# Los valores de normalizacion se calculan SOLAMENTE
# utilizando Train para evitar fuga de informacion.

min_val = train_x.min(axis=0)
max_val = train_x.max(axis=0)

difference = max_val - min_val

if (difference == 0).any():
    raise ValueError(
        "Existe una columna cuyo maximo y minimo son iguales."
    )


train_x = (train_x - min_val) / difference

validation_x = (
    validation_x - min_val
) / difference

test_x = (
    test_x - min_val
) / difference


train_x = train_x.astype(float)
validation_x = validation_x.astype(float)
test_x = test_x.astype(float)


#----------------------------------------------------------
# ONE-HOT ENCODING DE LAS SALIDAS
#----------------------------------------------------------

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


#==========================================================
# CONSTRUCCION DE RED
#==========================================================

network = models.Sequential()

# Siete entradas
network.add(
    Input(shape=(7,))
)

# Capas ocultas
for i in range(n):

    network.add(
        layers.Dense(
            units=units,
            activation=activation
        )
    )


# Capa de salida:
# low, medium y high

network.add(
    layers.Dense(
        units=3,
        activation="softmax"
    )
)


#----------------------------------------------------------
# OPTIMIZADOR
#----------------------------------------------------------

optimizer_model = tf.keras.optimizers.Adam(
    learning_rate=learning_rate
)


#----------------------------------------------------------
# COMPILACION
#----------------------------------------------------------

network.compile(
    optimizer=optimizer_model,
    loss=loss
)


#==========================================================
# ENTRENAMIENTO CON EARLY STOPPING
#==========================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


history = network.fit(
    x=train_x,
    y=train_y,
    validation_data=(
        validation_x,
        validation_y
    ),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[early_stopping],
    verbose=0
)


#==========================================================
# RESULTADOS DE VALIDACION
#==========================================================

history_df = pandas.DataFrame(
    history.history
)

mejor_epoca = (
    history_df["val_loss"].idxmin() + 1
)

mejor_val_loss = (
    history_df["val_loss"].min()
)

loss_mejor_epoca = history_df.loc[
    mejor_epoca - 1,
    "loss"
]


print("\n======================================")
print("RESULTADOS DEL ENTRENAMIENTO")
print("======================================")

print(
    "Epocas realmente ejecutadas:",
    len(history_df)
)

print(
    "Mejor epoca:",
    mejor_epoca
)

print(
    "Loss en la mejor epoca:",
    loss_mejor_epoca
)

print(
    "Mejor validation loss:",
    mejor_val_loss
)


#==========================================================
# GRAFICA LOSS VS VAL_LOSS
#==========================================================

plt.figure()

plt.plot(
    history_df.index + 1,
    history_df["loss"],
    label="loss"
)

plt.plot(
    history_df.index + 1,
    history_df["val_loss"],
    label="val_loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss de entrenamiento vs validacion")
plt.legend()
plt.grid()

plt.show()


#==========================================================
# EVALUACION FINAL CON TEST
#==========================================================

# Test se utiliza solamente despues de finalizar
# la seleccion del modelo.

test_loss = network.evaluate(
    test_x,
    test_y,
    verbose=0
)


predicciones_test = network.predict(
    test_x,
    verbose=0
)

predichos_test = predicciones_test.argmax(axis=1)

reales_test = test_y.argmax(axis=1)


correctas = (
    predichos_test == reales_test
).sum()

accuracy_test = (
    correctas /
    len(reales_test)
) * 100


print("\n======================================")
print("RESULTADOS FINALES EN TEST")
print("======================================")

print("Test loss:", test_loss)

print(
    "Predicciones correctas:",
    correctas
)

print(
    "Total de muestras de prueba:",
    len(reales_test)
)

print(
    "Exactitud:",
    accuracy_test,
    "%"
)


#----------------------------------------------------------
# MATRIZ DE CONFUSION
#----------------------------------------------------------

matriz = pandas.crosstab(
    pandas.Series(
        reales_test,
        name="Grade real"
    ),
    pandas.Series(
        predichos_test,
        name="Grade predicho"
    )
)

# Se asegura que aparezcan las tres clases
matriz = matriz.reindex(
    index=[0, 1, 2],
    columns=[0, 1, 2],
    fill_value=0
)

print("\nMatriz de confusion:")
print(matriz)


#==========================================================
# MODELO DEFINITIVO PARA PREDICCION
#==========================================================

# Una vez evaluado el modelo, se crea una NUEVA red.
#
# Esta nueva red utiliza todas las 83 muestras unicas.
# Los hiperparametros ya fueron seleccionados anteriormente.
#
# Se utiliza como cantidad de epocas la mejor epoca
# encontrada mediante Validation.


#----------------------------------------------------------
# PREPARAR LAS 83 MUESTRAS
#----------------------------------------------------------

final_x = dataset_unico[
    columnas_entrada
].copy()

final_y = dataset_unico[
    "Grade"
].copy()


#----------------------------------------------------------
# NORMALIZACION DEL MODELO DEFINITIVO
#----------------------------------------------------------

min_val_final = final_x.min(axis=0)

max_val_final = final_x.max(axis=0)

difference_final = (max_val_final - min_val_final)


if (difference_final == 0).any():

    raise ValueError("Existe una columna cuyo maximo y minimo son iguales.")


final_x = (final_x - min_val_final) / difference_final

final_x = final_x.astype(float)


final_y = to_categorical(final_y,num_classes=3)


#----------------------------------------------------------
# CREAR NUEVA RED DESDE CERO
#----------------------------------------------------------

tf.keras.utils.set_random_seed(42)

final_network = models.Sequential()

final_network.add(
    Input(shape=(7,))
)


for i in range(n):

    final_network.add(
        layers.Dense(
            units=units,
            activation=activation
        )
    )


final_network.add(
    layers.Dense(
        units=3,
        activation="softmax"
    )
)


#----------------------------------------------------------
# COMPILACION DEL MODELO DEFINITIVO
#----------------------------------------------------------

final_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


final_network.compile(optimizer=final_optimizer,
    loss=loss
)


#----------------------------------------------------------
# ENTRENAMIENTO CON LAS 83 MUESTRAS UNICAS
#----------------------------------------------------------

final_network.fit(
    x=final_x,
    y=final_y,
    batch_size=batch_size,
    epochs=mejor_epoca,
    verbose=0
)


print("\n======================================")
print("MODELO DEFINITIVO")
print("======================================")

print(
    "Muestras utilizadas:",
    len(final_x)
)

print(
    "Epocas utilizadas:",
    mejor_epoca
)

print("Capas ocultas:", n)
print("Neuronas por capa:", units)
print("Activacion:", activation)
print("Learning rate:", learning_rate)
print("Optimizador:", optimizer)
print("Batch size:", batch_size)


#==========================================================
# MUESTRAS SOLICITADAS EN LA PRACTICA
#==========================================================

muestras = pandas.DataFrame(
    [
        [7.9, 68, 1, 0, 1, 1, 238],
        [5.7, 22, 1, 1, 1, 1, 208],
        [6.8, 81, 0, 0, 1, 1, 257]
    ],
    columns=columnas_entrada
)


print("\n======================================")
print("MUESTRAS A CLASIFICAR")
print("======================================")

print(muestras)


#==========================================================
# NORMALIZAR LAS TRES MUESTRAS
#==========================================================

# Se utilizan exactamente los mismos valores de
# normalizacion del modelo definitivo.

muestras_normalizadas = (
    muestras - min_val_final
) / difference_final

muestras_normalizadas = (
    muestras_normalizadas.astype(float)
)


#==========================================================
# REALIZAR LAS PREDICCIONES
#==========================================================

predicciones = final_network.predict(
    muestras_normalizadas,
    verbose=0
)


clases = [
    "low",
    "medium",
    "high"
]


#==========================================================
# RESULTADOS
#==========================================================

print("\n======================================")
print("PREDICCIONES DE LAS MUESTRAS FINALES")
print("======================================")


for i in range(len(muestras)):

    indice_predicho = (
        predicciones[i].argmax()
    )

    grade_predicho = (
        clases[indice_predicho]
    )

    print("\n------------------------------")

    print(
        "Muestra",
        i + 1
    )

    print("------------------------------")

    print(
        "Probabilidad low:",
        predicciones[i][0]
    )

    print(
        "Probabilidad medium:",
        predicciones[i][1]
    )

    print(
        "Probabilidad high:",
        predicciones[i][2]
    )

    print(
        "Grade predicho:",
        grade_predicho
    )
