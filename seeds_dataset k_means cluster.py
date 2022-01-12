import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
plt.style.use('ggplot')

df = pd.read_csv('c:\\Users\\Abdulrahman\\Desktop\\Machine Learning\\seeds_dataset.csv')
df.replace('?', -99999, inplace=True)
df.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'label']

labels_true = np.array(df.label)
data = df.iloc[:, :7]

model = KMeans(n_clusters=2)
model_selection(model, data, labels_true)

def model_selection(model, data, labels_ture):
    labels_pred = model.fit_predict(data)
    nmi = normalized_mutual_info_score(labels_ture, labels_pred)
    print(nmi)
    
