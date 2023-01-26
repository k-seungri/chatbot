import pandas as pd

DATA_IN_PATH = './data_in/'

data = pd.read_csv(DATA_IN_PATH+'ChatBotData.csv', encoding='utf-8')

print(data)