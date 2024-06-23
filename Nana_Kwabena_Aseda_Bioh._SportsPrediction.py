#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd


# In[5]:


#Loading datasets into the program
male_players_df = pd.read_csv("C:\\Users\\User\\OneDrive - Ashesi University\\Intro to AI\\male_players.csv", low_memory = False)
players_22_df = pd.read_csv("C:\\Users\\User\\OneDrive - Ashesi University\\Intro to AI\\players_22.csv")


# In[6]:


#getting an initial view of the dataset by displaying the first five rows
male_players_df.head()


# In[22]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
male_players_df.hist(bins=50, figsize=(20,15))
plt.show()


# In[7]:


#dropping irrelevant attributes
male_players_df.drop(['player_id','player_url','player_face_url','fifa_version','fifa_update','fifa_update_date','player_face_url','real_face','dob','club_jersey_number','club_loaned_from','club_joined_date','club_contract_valid_until_year','nationality_id','nationality_name','nation_team_id','nation_position','nation_jersey_number','preferred_foot','weak_foot','release_clause_eur','league_id','league_name','league_level','club_team_id','club_name','club_position','age'],axis=1,inplace=True)
male_players_df.drop(male_players_df.loc[:,'ls':],axis=1,inplace=True)


# In[8]:


potential_drops = []
maintaining_columns = []

for col in male_players_df.columns:
    #storing the columns whose missing values are less than 30% of the dataset shape
    if((male_players_df[col].isnull().sum() < (0.3 * (male_players_df.shape[0])))):
        maintaining_columns.append(col)
        
    #storing the columns whose missing values are more than 30% of the dataset shape    
    else:
        potential_drops.append(col)

print(potential_drops)
print(maintaining_columns)


# In[9]:


#replaces the dataset with columns having missing values less than 30% of the dataset's shape
male_players_df = male_players_df[maintaining_columns]
male_players_df


# In[10]:


import numpy as np
#separates the dataset into non_numeric and numeric data
numeric_data = male_players_df.select_dtypes(include=np.number)
non_numeric= male_players_df.select_dtypes(include=['object'])

non_numeric


# In[11]:


#uses the iterative imputer class to predict the values of the missing data in the numerical columns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imputer = IterativeImputer(max_iter=10, random_state=0)
numeric_data = pd.DataFrame(np.round(imputer.fit_transform(numeric_data)), columns=numeric_data.columns)


# In[12]:


#store the relevant non numeric columns
work_rate = non_numeric['work_rate']
body_type = non_numeric['body_type']

#encode the relevant non numeric columns using pandas' get dummies method
work_rate = pd.get_dummies(work_rate, prefix = 'work_rate_').astype(int)
body_type = pd.get_dummies(body_type, prefix = 'body_type_').astype(int)


#concatenate the numerical columns with the encoded columns
male_players_df = pd.concat([numeric_data,work_rate,body_type], axis=1).reset_index(drop=True)


# In[13]:


#Calculate correlation matrix
corr_matrix = male_players_df.corr()['overall']
corr_matrix


# In[14]:


#selecting attributes with highest correlation(threshold = 0.5)
high_corr_vals = corr_matrix[(corr_matrix >= 0.5) | (corr_matrix <= -0.5)]
high_corr_vals


# In[15]:


#Creating feature subset with attributes having a high correlation with the target variable
feature_subsets = male_players_df[['overall','potential','value_eur','wage_eur','passing','movement_reactions','mentality_composure']]
feature_subsets


# In[16]:


y = feature_subsets['overall'] #dependent variable
X = feature_subsets.drop('overall',axis=1) #independent variables


# In[17]:


#Scaling the data
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scaled = scale.fit_transform(X)
scaled


# In[18]:


#Creating a dataframe from the scaled data of independent variables
X = pd.DataFrame(scaled, columns = X.columns) 


# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


#splitting data into test and training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify =y)


# In[18]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

gb = GradientBoostingRegressor(n_estimators=100)
rf = RandomForestRegressor(n_estimators=100)
xgbmodel = xgb.XGBRegressor(n_estimators=100)



# In[19]:


from sklearn.model_selection import cross_val_score
from collections import namedtuple

#Create a named tuple containing the model name and the model itself as key-value pairs
tuple_for_model = namedtuple('tuple_for_model', ['GradientBoostingRegressor','RandomForestRegressor','XGBRegressor'])
my_tuple = tuple_for_model(GradientBoostingRegressor=gb,RandomForestRegressor=rf,XGBRegressor=xgbmodel)

#Convert to dictionary
my_dict = {key: getattr(my_tuple,key) for key in my_tuple._fields}

#Train each model with cross validation score and display their RMSE
for name, model in my_dict.items():
    scores = cross_val_score(model,X_train,y_train,cv=5,scoring='neg_mean_squared_error')
    print(f'{name} CV RMSE: {np.sqrt(-scores).mean()}')


# In[25]:


from sklearn.model_selection import GridSearchCV
#Defining hyperparameter grids for best model
param_grid_rf = {
    'n_estimators': [50,100,150],
    'max_depth': [3,5,7],
    "max_depth": [5, 10, 15]
}



# In[26]:


from sklearn.metrics import mean_squared_error
#Creating and fitting best model with GridSearchCV
rf_model = RandomForestRegressor()
rf_grid_search = GridSearchCV(rf_model,param_grid=param_grid_rf,scoring='neg_mean_squared_error',cv=5)
rf_grid_search.fit(X_train,y_train)


# In[27]:


print('Best Parameters:',rf_grid_search.best_params_) #determine the best parameters for the model
print('Best CV RMSE:', np.sqrt(-rf_grid_search.best_score_)) #display the best CV RMSE


# In[28]:


#Train and test the optimized best model
model_rf = RandomForestRegressor(max_depth = 15, n_estimators=150)
model_rf.fit(X_train,y_train)

y_pred = model_rf.predict(X_test)
print(np.sqrt(mean_squared_error(y_pred,y_test)))


# In[65]:


import pickle as pkl
pkl.dump(model_rf,open('C:\\Users\\User\\OneDrive - Ashesi University\\Intro to AI' + model_rf.__class__.__name__ + '.pkl','wb'))


# In[30]:


players_22_df.head()


# In[33]:


#select attributes that are relevant to the test
players_22_df = players_22_df[['overall','potential','value_eur','wage_eur','passing','movement_reactions','mentality_composure']]
players_22_df


# In[34]:


#checking for null values
players_22_df.isnull().sum()


# In[38]:


#uses the iterative imputer class to predict the values of the missing data in the dataset
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(max_iter=10, random_state=0)
players_22_df = pd.DataFrame(np.round(imp.fit_transform(players_22_df)), columns=players_22_df.columns)


# In[39]:


players_22_df.isnull().sum() #checking if the imputation was successful


# In[40]:


#splitting dataset into dependent and independent data
dependent = players_22_df['overall']
independent = players_22_df.drop('overall',axis=1)


# In[41]:


#Scaling the independent attributes
scaler = StandardScaler()
scaled_data = scaler.fit_transform(independent)


# In[47]:


#Converting scaled data to a dataframe
independent = pd.DataFrame(scaled_data, columns = independent.columns)
independent


# In[51]:


#Evaluate how good the model is by testing it on completely new data
dependent_pred = model_rf.predict(independent)
test_rmse = np.sqrt(mean_squared_error(dependent_pred,dependent))
print('Test RMSE:',test_rmse)

