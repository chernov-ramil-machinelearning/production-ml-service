import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'E:\python_projects\sorted_roadmap\05_production_projects\production_ml_service\data\bank_churn.csv')

# EDA
print(f'{df.head()}\n')
print(f'Есть ли пустоты в датасете: \n{df.isnull().sum()}\n')
print(f'Длина\Ширина: \n{df.shape}\n')
print(f'Основные данные по таблице: \n{df.describe()}')

# plt.figure() - Закомичены из-за отсутсвия нужды после этапа EDA
# sns.heatmap(df.corr(numeric_only=True), annot=True)
# plt.show()

# Данные чистые, в дополнительной верстке на начальном этапе вне обучения не нуждаюся, из замеченных теоретических проблем, почти нету коррелирующих признаков, все признаки очень слабо коррелируют