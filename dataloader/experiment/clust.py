import pandas as pd

csv_file = 'a.csv'
# data = pd.read_csv(csv_file, header=0).T
# data = data.drop(columns=0)
# data.to_csv("a.csv")
data = pd.read_csv(csv_file)
data["color_mode"] = data["color_mode"].apply(lambda x: 0 if x == 'L' else 1)
# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本2个特征，共4个簇，簇中心在[-1,-1], [0,0],[1,1], [2,2]， 簇方差分别为[0.4, 0.2, 0.2]

from sklearn.cluster import KMeans

data = data.loc[:, ['color_mode', 'image_count']]
print(data.head())

y_pred = KMeans(n_clusters=2,random_state=10).fit_predict(data)
# plt.scatter(data[:, 0], data[:, 1], c=y_pred)
# plt.show()
print(y_pred)
