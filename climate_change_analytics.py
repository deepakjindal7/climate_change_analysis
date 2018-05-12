
# coding: utf-8

# In[2]:


import pandas as pd
import seaborn as sns
import plotly as p
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[3]:



data=pd.read_csv("GlobalLandTemperaturesByMajorCity.csv")


# In[4]:


data.head()



# In[5]:


data['AverageTemperature'].fillna(data['AverageTemperature'].mean())
data['AverageTemperatureUncertainty'].fillna(data['AverageTemperatureUncertainty'].mean())


# In[6]:


data.columns


# In[7]:


data.head()


# In[9]:


data['dt']=pd.to_datetime(data['dt'])




# In[12]:


data['year']=data['dt'].dt.year


# In[13]:




data.head()


# In[ ]:


sns.boxplot(x='year', y='AverageTemperature', data=data,saturation=0.6, width=2, dodge=True, fliersize=15, linewidth=10)


# In[ ]:


data.max()


# In[ ]:


from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor


# In[10]:


#X=data.drop(labels=['dt','AverageTemperature','City','Country'],axis=1 )
#y=data['AverageTemperature']
#X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.15, random_state=30)
#r=RandomForestRegressor(n_estimators=30, criterion='mse')
#r.fit(X_train,y_train)
data.info()


# In[ ]:


import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go

app = dash.Dash()

df = data


app.layout = html.Div([
    dcc.Graph(
        id='average-temperature-vs-year',
        figure={
            'data': [
                go.Scatter(
                    x=df[df['City'] == i]['year'],
                    y=df[df['City'] == i]['AverageTemperature'],
                    text=df[df['City'] == i]['Country'],
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=i
                ) for i in df.City.unique()
            ],
            'layout': go.Layout(
                xaxis={'type': 'log', 'title': 'year'},
                yaxis={'title': 'average-temperature'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
        }
    )
    
])

if __name__ == '__main__':
    app.run_server()

