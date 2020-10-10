
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import pycaret



data = pd.read_csv("https://project-employees.s3.amazonaws.com/employeeinfo.csv")


# In[2]:

data = data.drop(['EmployeeCount','EmployeeNumber','Over18','StandardHours'],axis=1)



# In[3]:

# In[4]:



# In[5]:




# In[7]:





# In[9]:



# In[10]:


# In[11]:

from pycaret.classification import *
sd1=setup(data,target='Attrition')


# In[12]:



# In[13]:

sd1 = create_model('ridge',verbose=False)



# In[14]:

sd1=tune_model(sd1,verbose=False)


# In[15]:

save_model(sd1, model_name = r"C:\Users\GagaN\Desktop\MLproject1\newfile")


# In[ ]:
