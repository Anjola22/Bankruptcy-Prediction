#!/usr/bin/env python
# coding: utf-8

# ## BANKRUPTCY PROJECT

# ### GROUP 3
# KEYEDE ANJOLA---017
# ADENIYI ADEBIMPE---002
# 

# # BUSINESS UNDERSTANDING
# 
# ### This is based on noting factors that could lead to the bankruptcy of the company. We will be checking out these factors and noticing if they can actually lead to bankruptcy of the company.

# # DATA UNDERSTANDING
# 
# 
# ### Dataset contains series of factors that predicts a company's growth, to figure out if the company is growing positively and negatively.  

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


company_data = pd.read_csv('bankruptcy.csv')
company_data.head(8)


# --Cash flow is the amount of money 
# --Growth Rate is the final value - initial value/ initial value
# --Debt Percent is total Depth of the company divided by 100
# --borrowing depedency is the ratio of your debt dependency
# --Capital is the total amount used to start a business or a company
# --Total Assets is the sum of all assets owned by the company
# --Liabilities refers to debt and expenses
# --Liability assets 
# --Net Income is the total amount of income
# --Net profit is the Total expenses subtracted from The Total income
# --Equity to Liability is the total debt divided by the total equity

# In[3]:


company_data.describe()


# In[4]:


company_data.info()


# In[5]:


company_data.corr()


# In[6]:


type(company_data)


# In[7]:


company_data.dtypes


# In[8]:


company_data.shape 


# The above shows that there are 6144 rows and 12 columns in the dataset

# In[9]:


company_data["Debt Percent"].describe()


# ## Data Quality

# In[10]:


print('null values for company_data ?',company_data.isnull().values.any())


# The above cell shows that there is no null value in the company_data

# In[11]:


company_data.columns[company_data.isna().any()].tolist()


# # Analysis of the distribution of variables using graphs

# In[12]:


import matplotlib.pyplot as plt


# In[13]:


Profit_by_company = company_data.groupby('Bankrupt?').agg({'Net Profit':'sum'}).sort_values('Net Profit',ascending=False)


# In[14]:


Profit_by_company


# In[15]:


Profit_by_company['Net Profit'].plot(kind='pie',autopct='%1.1f%%',legend = True)
plt.show()


# The above illustration shows that 97% out of 100% did not go bankrupt while the remaining percentage went bankrupt because they have a very little amount of net profit compared to the successful companies

# In[16]:


table=pd.crosstab(company_data['Debt Percent'],company_data['Total Assets'])
table.plot(kind='bar', stacked=True,figsize=(10,10))
plt.show()


# # DATE PREPARATION
# 
# ## DATA CLEANING
# 
# ### The features which are in the wrong format; in duplicates; empty cells will be corrected and adjusted. Also, the empty cells will be dropped.

# ### Since there are no empty cells, then there won't be a need of dropping or removing cells
# 

# In[17]:


company_data.dtypes


# ### We will be using the 'Debt Percent' column to Predict bankruptcy, and the above cell shows that its data type is object, so we will be cleaning it by removing the percentage sign

# In[18]:


company_data["Debt Percent"] = company_data["Debt Percent"].map(lambda x : int(x.replace('%','')))


# In[19]:


company_data.head(4)


# In[20]:


company_data["Debt Percent"].describe()


# In[21]:


company_data.dtypes


# In[23]:


import matplotlib.pyplot as plt


# Below is a random distribution curve with standard deviation of 5 and mean of 1.0

# In[24]:


import numpy as np
import matplotlib.pyplot as plt
x=np.random.normal(5.0,1.0,10000)
plt.hist(x,100)
plt.show()


# In[25]:


from sklearn import linear_model


# Using classified supervised learning, the table below shows that the companies that went bankrupt were as a result of their Net Profit, Growth Rate, Debt Percent

# ## Data Prediction

# In[26]:


company_data.groupby('Bankrupt?').mean().T


# The below cell is the simplification of the above 

# In[27]:


#Bankrupt?	0	1
#Cash Flow	0.3239	0.315
#Growth Rate	226738.327	49105260.0
#Debt Percent	11.0792	18.4421
#Borrowing dependency	0.373673	0.3882105
#Total Assets	0.029320	0.051736
#Liabilities	0.462427	0.454684
#Liability Assets	0.000336	0.0263157
#Net Profit	0.608810	0.59862
#Net Income	0.840938	0.827962
#Equity to Liability	0.048900	0.0261052


# In[28]:


import seaborn as sns


# ### We will be considering these factors to predict bankruptcy state of a company

# In[29]:


cp = company_data[['Net Profit', 'Growth Rate', 'Debt Percent']]
cp.head()


# In[30]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split


# ## Machine Learning
# 

# ### This has to do with building models that teaches the machine how and what to predict
# 

# ### Start by first training and testing the model to be accurate

# In[31]:


x = cp
y = company_data['Bankrupt?']
from sklearn.model_selection import train_test_split


# In[32]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state= 40)


# In[33]:


x_train


# In[34]:


y_test


# In[35]:


from sklearn.linear_model import LogisticRegression


# In[36]:


model = LogisticRegression()
model.fit(x_train, y_train)


# In[37]:


predicted = model.predict(x_test)
model.predict_proba(x_test)


# In[38]:


from sklearn import metrics


# In[39]:


metrics.accuracy_score(y_test,predicted)


# In[53]:


print('This bankruptcy prediction is', metrics.accuracy_score(predicted,y_test)*100,'% accurate')


# In[40]:


model.predict([[0.6, 0.0004,16]])


# In[41]:


model.predict([[0.7,0.0009,30]])


# #### array[1] shows that a company that has that Net Profit, Growth rate and Debt Percent will go bankrupt and array[0] means will be successful and not go bankrupt

# In[ ]:




