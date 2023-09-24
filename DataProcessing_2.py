import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

df=pd.read_csv('./data/Wednesday-workingHours.pcap_ISCX.csv')
df['Label2']=np.where(df[' Label']=='BENIGN', 0, 1)
#print(df['Label2'].head(5))
x=df.corr()['Label2']
x=pd.DataFrame(x)
x=x[(x['Label2']>0.5) | (x['Label2'] <-0.5)]
xt=x.transpose()
print(xt)
#sns.heatmap(xt)
#plt.show()
df2=df[df.columns.intersection(xt.columns)]
train, test=train_test_split(df2, test_size=0.1)
train.to_csv('./data/CorrelationDataTrain.csv', index=False)
test.to_csv('./data/CorrelationDataTest.csv', index=False)