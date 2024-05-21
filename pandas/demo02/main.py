import pandas as pd

# Criando uma Série a partir de uma lista
data = [10, 20, 30, 40, 50]
serie = pd.Series(data)
print(serie)

# Criando uma Série com índices personalizados
data = [10, 20, 30, 40, 50]
indices = ['a', 'b', 'c', 'd', 'e']
serie = pd.Series(data, index=indices)
print(serie)

# Criando uma Série a partir de um dicionário
data_dict = {'a': 10, 'b': 20, 'c': 30, 'd': 40, 'e': 50}
serie = pd.Series(data_dict)
print(serie)

# Acessando elementos
print(serie['c'])  # Saída: 30
print(serie[2])    # Saída: 30

# Operações aritméticas
serie2 = serie * 2
print(serie2)

# Calculando a média
print(serie.mean())  # Saída: 30.0

# Soma dos elementos
print(serie.sum())  # Saída: 150

# Estatísticas descritivas
print(serie.describe())

