import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Gerar dados de exemplo
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Construir o modelo de regressão linear
model = Sequential()
model.add(Dense(1, input_dim=1, kernel_initializer='normal', activation='linear'))

# 3. Compilar o modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# 4. Treinar o modelo
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)

# 5. Avaliar o modelo
loss = model.evaluate(X_test, y_test)
print(f'Loss (Erro Quadrático Médio): {loss}')

# 6. Fazer previsões
y_pred = model.predict(X_test)

# Opcional: Visualizar os resultados
import matplotlib.pyplot as plt

plt.scatter(X_test, y_test, color='red', label='Dados Reais')
plt.scatter(X_test, y_pred, color='blue', label='Previsões')
plt.legend()
plt.show()
