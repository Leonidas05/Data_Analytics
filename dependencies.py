import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

# df = pd.read_excel("StockPrice.xlsx")
# datos = df.loc[:,df.columns != "Date"].to_numpy()
# colnames = df.columns.values
# # print(colnames)

# #matriz cov
# mCov = np.cov(datos,rowvar=False).round(decimals=1)
# mCov.shape

# plt.style.use('_mpl-gallery')
# fig, ax=plt.subplots(figsize = (7,7))
# ax.xaxis.set(ticks=range(0,9),ticklabels=colnames[1:])
# ax.yaxis.set(ticks=range(0,9),ticklabels=colnames[1:])
# ax.grid(False)

# im= ax.imshow(mCov)

# for i in range (0, len(colnames)-1):
#     for j in range (0, len(colnames)-1):
#         ax.text(j,i,mCov[i,j],ha='center',va='center',color="g")
    
# cbar = ax.figure.colorbar(im, ax= ax, format='%0.2f')

# plt.show()


df = pd.read_excel("StockPrice.xlsx")
datos = df.loc[:,df.columns != "Date"].to_numpy()
colnames = df.columns.values
# print(colnames)

#matriz cov
mCorr = np.corrcoef(datos,rowvar=False).round(decimals=1)
mCorr.shape

plt.style.use('_mpl-gallery')
fig, ax=plt.subplots(figsize = (7,7))
ax.xaxis.set(ticks=range(0,9),ticklabels=colnames[1:])
ax.yaxis.set(ticks=range(0,9),ticklabels=colnames[1:])
ax.grid(False)

im= ax.imshow(mCorr)

for i in range (0, len(colnames)-1):
    for j in range (0, len(colnames)-1):
        ax.text(j,i,mCorr[i,j],ha='center',va='center',color="r")
    
cbar = ax.figure.colorbar(im, ax= ax, format='%0.2f')

plt.show()


#grafo de dependencias : dependencias más significativas
#9 casos(variables) con un 10% de error, el valor es 0.6

vsig = 0.7
g = nx.Graph()
for i in range (0, len(colnames)-1):
    for j in range (0, len(colnames)-1):
        if(i != j and (mCorr[i,j] >= vsig or mCorr[i,j] <= -vsig)):
            g.add_edge(i,j)

nx.draw_circular(g, with_labels=True)
plt.show()