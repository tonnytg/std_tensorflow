import pandas as pd

data = {
    'Nome': ['Alice', 'Bob', 'Cecilia'],
    'Idade': [24, 27, 22],
    'Cidade': ['São Paulo', 'Rio de Janeiro', 'Curitiba']
}
df = pd.DataFrame(data)
print(df)

#
df = pd.read_csv('file_name.csv',
			usecols=[1, 2, 3])
print(df.head(10))

df = pd.DataFrame({'Yes': [50, 21], 'No': [131, 2]})
print(df)