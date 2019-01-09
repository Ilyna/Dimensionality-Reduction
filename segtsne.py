import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pylab

# Importing the dataset
dataset = pd.read_csv('segmentation.data.txt', sep=",", header=2)
X = dataset.iloc[:, 1:20].values
y = dataset.iloc[:, 0]
labels, numbers = pd.factorize(y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 19)
Xpca = pca.fit_transform(X)

# Applying tSNE
from sklearn.manifold import TSNE
tsne = TSNE(n_components = 2)
Xnew = tsne.fit_transform(Xpca)

# Scatter Plot
cdict = {0: 'yellow', 1: 'red', 2: 'blue', 3: 'green', 4: 'black', 5: 'orange', 6: 'pink'}
fig, ax = plt.subplots()
for g in np.unique(labels):
    ix = np.where(labels == g)
    ax.scatter(Xnew[:,0][ix], Xnew[:,1][ix], c = cdict[g], label = g)
ax.legend()
plt.show()