import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Dados de treinamento
acertos_texto = np.array([
    [35],
    [40],
    [34],
    [33],
    [36],
    [42],
    [36],
    [36],
    [36],
    [35],
    [35],
    [33],
    [39],
    [33],
    [37]
])

notas_texto = np.array([
    [866.2],
    [842.1],
    [830],
    [807],
    [860.5],
    [930.7],
    [812.9],
    [832.1],
    [812.3],
    [805.2],
    [807.6],
    [785.4],
    [823.9],
    [789.5],
    [820.8]
])

# Normalização dos dados
acertos_max = np.max(acertos_texto)
notas_max = np.max(notas_texto)

acertos_normalizados = acertos_texto / acertos_max
notas_normalizadas = notas_texto / notas_max

# Criação do modelo
model = Sequential()
model.add(Dense(16, input_shape=(1,), activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

# Compilação do modelo
model.compile(loss='mse', optimizer='adam')

# Treinamento do modelo
model.fit(acertos_normalizados, notas_normalizadas, epochs=1000, verbose=0)

# Entrada do usuário
acertos_usuario = np.array([[45]])
acertos_usuario_normalizados = acertos_usuario / acertos_max

# Predição da nota aproximada
nota_aproximada_normalizada = model.predict(acertos_usuario_normalizados)
nota_aproximada = nota_aproximada_normalizada * notas_max

print("A sua nota aproximada é:", nota_aproximada[0])
