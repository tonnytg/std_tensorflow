import pandas as pd

data = ['Antonio', 'Marilia', 'Marcia', 'Maurilio']

df = pd.DataFrame(data)  # Cria um Data Frame

print("df1:", df)

df = pd.read_csv('arquivo.csv', delimiter=';', quotechar="'")
print("df2:", df)

print("def describe:", df.describe())  # Descrevendo dados estatisticos

df_filtred = df[df['age'] > 35]
print("df filtred:", df_filtred)

df_grouped = df.groupby('age').head()
print("df grouped:", df_grouped)

df.head()