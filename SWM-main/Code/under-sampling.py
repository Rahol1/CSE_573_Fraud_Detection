# -- coding: utf-8 --
import pandas as pd

data = pd.read_csv("Data/bs140513_032310.csv")

fraud1 = data[data['fraud']==1]

fraud0 = data[data['fraud']==0]

connections= []
customers = list(set(fraud1['customer']))
i=0
for cust in customers  :
    print(i)
    i=i+1
    connections.append(fraud0[fraud0['customer']==cust])
    
customer_related = pd.concat(connections)


    


connections= []
merchants = list(set(fraud1['merchant']))
i=0
for mer in merchants  :
    print(i)
    i=i+1
    connections.append(fraud0[fraud0['merchant']==mer])


merchant_related = pd.concat(connections)
