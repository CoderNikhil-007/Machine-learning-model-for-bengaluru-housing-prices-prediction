#!/usr/bin/env python
# coding: utf-8

# In[78]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)


# In[79]:


Data = "C:/Users/Nikhil/Downloads/Bengaluru_House_Data.csv"
DF1 = pd.read_csv(Data)
DF1


# In[80]:


DF1.shape


# In[81]:


DF1.groupby('area_type')['area_type'].agg('count')


# In[82]:


DF1.drop(['area_type','availability','society','balcony'],axis='columns',inplace = True)


# In[83]:


DF1.head()


# In[84]:


DF1.isnull().sum()


# In[85]:


DF1.dropna(inplace=True)


# In[86]:


DF1.isnull().sum()


# In[87]:


DF1.shape


# In[88]:


DF1['BHK'] = DF1['size'].apply(lambda x: int(x.split(' ')[0]))


# In[89]:


DF1


# In[90]:


DF1.total_sqft.unique()


# In[91]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[92]:


DF1[~DF1['total_sqft'].apply(is_float)].head(10)


# In[93]:


def convert(x):
    tokens = x.split('-')
    if len(tokens)==2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


# In[94]:


DF2 = DF1.copy()
DF2['total_sqft']=DF2['total_sqft'].apply(convert)
DF2.loc[30]


# In[95]:


DF2['Price_per_sqft'] = DF2['price']*100000/DF2['total_sqft']
DF2


# In[96]:



DF2.location.unique()


# In[97]:


DF2.location=DF2.location.apply(lambda x: x.strip())


# In[98]:


location_stats = DF2.groupby('location')['location'].agg('count').sort_values(ascending=False)


# In[99]:


location_stats


# In[100]:


len(location_stats[location_stats<=10])


# In[101]:


location_less_than_10 = location_stats[location_stats<=10]
location_less_than_10


# In[102]:


DF2.location = DF2.location.apply(lambda x: 'other' if x in location_less_than_10 else x)


# In[103]:


len(DF2.location.unique())


# In[104]:


DF2


# In[105]:


DF3 = DF2[~(DF2.total_sqft/DF2.BHK<300)]
DF3.shape


# In[106]:


DF3.Price_per_sqft.describe()


# In[107]:


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.Price_per_sqft)
        st = np.std(subdf.Price_per_sqft)
        reduced_df = subdf[(subdf.Price_per_sqft>(m-st)) & (subdf.Price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index= True)
    return df_out


# In[108]:


DF4 = remove_pps_outliers(DF3)
DF4.shape


# In[109]:


DF4


# In[110]:


def plot_scatter_chart(df,location):
    BHK2 = df[(df.location==location)&(df.BHK==2)]
    BHK3 = df[(df.location==location)&(df.BHK==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(BHK2.total_sqft,BHK2.price,color = 'blue',label = '2 BHK', s = 30)
    plt.scatter(BHK3.total_sqft,BHK3.price,color = 'black',label = '3 BHK', s = 30)
    plt.xlabel("Total sqft area")
    plt.ylabel("Price")
    plt.title(location)
    plt.legend()


# In[111]:


plot_scatter_chart(DF4,'Rajaji Nagar')


# In[112]:


def remove_bhk_outliers(df):
    exclusive_id = np.array([])
    for location,location_df in df.groupby('location'):
        BHK_stats = {}
        for bhk,bhk_df in location_df.groupby('BHK'):
            BHK_stats[bhk] = {
                'mean' : np.mean(bhk_df.Price_per_sqft),
                'std' : np.std(bhk_df.Price_per_sqft),
                'count' : bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('BHK'):
            stats = BHK_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclusive_id = np.append(exclusive_id, bhk_df[bhk_df.Price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclusive_id,axis = 'index')


# In[113]:


DF5 = remove_bhk_outliers(DF4)
DF5.shape


# In[114]:


plot_scatter_chart(DF5,'Rajaji Nagar')


# In[115]:


matplotlib.rcParams['figure.figsize'] = (20,10)
plt.hist(DF5.Price_per_sqft,rwidth = 0.8)
plt.xlabel("Price per sqft")
plt.ylabel("count")


# In[116]:


DF5.bath.unique()


# In[117]:


DF5[DF5.bath>10]


# In[118]:


plt.hist(DF5.bath,rwidth = 0.8)
plt.xlabel("Bathrooms")
plt.ylabel("count")


# In[119]:


DF6= DF5[DF5.bath<DF5.BHK+2]
DF6.shape


# In[120]:


DF6.drop(["size","Price_per_sqft"],axis="columns",inplace = True)


# In[121]:


DF6


# In[122]:


Dummies=pd.get_dummies(DF6.location)


# In[123]:


DF7 = pd.concat([DF6,Dummies.drop('other',axis="columns")],axis="columns")


# In[124]:


DF7


# In[125]:


DF7.drop('location',axis="columns",inplace = True)


# In[126]:


DF7.shape


# In[127]:


x = DF7.drop('price',axis="columns")
y = DF7.price


# In[128]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=10)


# In[129]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
model.score(x_test,y_test)


# In[130]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cross_val_score(LinearRegression(),x,y,cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=0))


# In[131]:


np.mean(cross_val_score(LinearRegression(),x,y,cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)))


# In[132]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def finding_best_model(x,y):
    algos = {
        'linear Regression' : {
            'model' : LinearRegression(),
            'params': {
            } 
        },
        'lasso': {
            'model' : Lasso(),
            'params' : {
                'alpha' : [1,2],
                'selection' : ['random','cyclic']
            }
        },
        'Decision Tree Regressor' : {
            'model' : DecisionTreeRegressor(),
            'params' : {
                'criterion' : ['mse','friedman_mse'],
                'splitter' : ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
    for algo_name, config in algos.items():
        clf =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        clf.fit(x,y)
        scores.append({
            'model': algo_name,
            'best_score': clf.best_score_,
            'best_params': clf.best_params_
        })
    
    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

finding_best_model(x,y)


# In[149]:


def price_predict(location,area,bath,bhk):
    X = np.zeros(len(x.columns))
    X[0] = area
    X[1] = bath
    X[2] = bhk
    if location !='other':
        X[np.where(x.columns == location)[0][0]] = 1
    return model.predict([X])[0]


# In[150]:


price_predict('1st Block Jayanagar',2850,4,4)


# In[151]:


x2 = np.zeros(len(x.columns))
x2[0] = 1709
x2[1] = 3
x2[2] = 3
model.predict([x2])[0]


# In[152]:


price_predict('Indira Nagar',1000,3,3)


# In[153]:


import pickle
with open("Banglore_home_prices_data.pickle","wb") as f:
    pickle.dump(model,f)


# In[154]:


with open("Banglore_home_prices_data.pickle","rb") as f:
    mp = pickle.load(f)


# In[155]:


x3 = np.zeros(len(x.columns))
x3[0] = 1000
x3[1] = 3
x3[2] = 3
mp.predict([x3])


# In[171]:


import json


# In[174]:


columns = {
    'data_columns' : [col.lower() for col in x.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))


# In[ ]:





# In[ ]:




