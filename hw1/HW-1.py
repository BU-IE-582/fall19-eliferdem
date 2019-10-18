#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Imported Packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import  poisson


# In[4]:


#Import csv files to python as a dataframe
bets = pd.read_csv(r"C:\Users\elif.erdem\Desktop\Master\IE-582\HW-1\bets.csv") 
booking = pd.read_csv(r"C:\Users\elif.erdem\Desktop\Master\IE-582\HW-1\booking.csv") 
goals = pd.read_csv(r"C:\Users\elif.erdem\Desktop\Master\IE-582\HW-1\goals.csv") 
matches = pd.read_csv(r"C:\Users\elif.erdem\Desktop\Master\IE-582\HW-1\matches.csv") 
stats = pd.read_csv(r"C:\Users\elif.erdem\Desktop\Master\IE-582\HW-1\stats.csv") 


# * Matches dataframe consists of the info required for Q1. Hence, this df is analyzed with describe and info detailly.

# * Type of columns which are with match_status, match_hometeam_name, match_awayteam_name is not numerical.
# * Match_status, match_hometeam_score, match_awayteam_score,match_hometeam_halftime_score, match_awayteam_halftime_score,match_hometeam_extra_score, match_awayteam_extra_score, match_hometeam_penalty_score, match_awayteam_penalty_score columns have missing values.
# * match_awayteam_penalty_score , match_hometeam_penalty_score  columns can be dropped since they do not include any info.
# 

# In[5]:


matches = matches.drop(columns=['match_hometeam_penalty_score', 'match_awayteam_penalty_score'])


# In[6]:


matches.describe()


# * Columns that are match_hometeam_score, match_awayteam_score, match_hometeam_halftime_score, match_awayteam_halftime_score, match_hometeam_extra_score, match_awayteam_extra_score analyzed min, max and mean values to analyze outliers

# - English premier league is analyzed for the Q1. Hence, the raw data is limited for the LeagueId=148(EnglishPremier)

# In[7]:


matches_premier= matches[matches['league_id']==148]


# In[8]:


matches_premier.head(2)


# In[9]:


matches_premier.info()


# In[10]:


matches_premier = matches_premier[matches_premier['match_status']=='Finished']


# * Match_satus=Finished is selected

# In[11]:


matches_premier = matches_premier.drop_duplicates()
matches_premier['match_id'].duplicated().unique()


# * Duplicate values and NAN values ara dropped.

# In[12]:


matches_premier.shape


# In[13]:


matches_premier = matches_premier[matches_premier['match_hometeam_score'].notnull()]


# In[14]:


matches_premier = matches_premier[matches_premier['match_awayteam_score'].notnull()]


# * Each match must contain scores

# In[15]:


matches_premier.columns


# In[16]:


matches_premier_control = matches_premier.groupby(['match_id']).count()


# In[17]:


matches_premier_control.shape


# * Data is singuler based of match_id.

# In[18]:


matches_premier = matches_premier[['match_id', 'match_hometeam_score', 'match_awayteam_score', 'match_hometeam_extra_score', 'match_awayteam_extra_score']]


# In[19]:


matches_premier['difference'] =  matches_premier['match_hometeam_score'] -  matches_premier['match_awayteam_score']


# * Only neccessary columns are selected

# In[20]:


matches_premier.describe()


# In[21]:


matches_premier['match_hometeam_score'].size


# # Task 1 

# In[22]:


mean =int( matches_premier['match_hometeam_score'].mean())


# In[23]:


size= int(matches_premier['match_hometeam_score'].size)


# In[24]:


bins_home =int( matches_premier['match_hometeam_score'].max())


# In[25]:


range_min_home= int(matches_premier['match_hometeam_score'].min())


# In[28]:


sns.distplot(matches_premier.match_hometeam_score, kde=False, bins = bins_home) ## bins= max(match_hometeam_score)
poissonpmf = np.histogram(np.random.poisson(lam =mean, size = size),
                            bins=bins_home, range=(range_min_home, bins_home))
plt.plot(poissonpmf[0])
plt.xlabel('Home Goals')
plt.ylabel('Number of Games')


# In[29]:


bins_away =int( matches_premier['match_awayteam_score'].max())


# In[30]:


size_away= int(matches_premier['match_awayteam_score'].size)


# In[31]:


mean_away =int( matches_premier['match_awayteam_score'].mean())


# In[32]:


range_min_away= int(matches_premier['match_awayteam_score'].min())


# In[33]:


sns.distplot(matches_premier.match_awayteam_score, kde=False, bins = bins_away)#bins=max(match_awayteam_score)
poissonpmf = np.histogram(np.random.poisson(lam =mean_away, size = size_away),
                            bins=bins_away, range=(range_min_away, bins_away))
plt.plot(poissonpmf[0])
plt.xlabel('Away Goals')
plt.ylabel('Number of Games')


# In[34]:


sns.distplot(matches_premier.difference, kde=False, bins = 8) ##bins =  max(match_hometeam_score) - min(match_awayteam_score)
plt.xlabel('Home-Away')
plt.ylabel('Number of Games')


# # Task - 2

# In[35]:


bets = bets[bets["odd_bookmakers"].isin(['10Bet', 'bwin', 'Unibet', 'Marathonbet'])]


# In[36]:


bets = bets[bets['variable'].isin(['odd_1', 'odd_x', 'odd_2'])]


# In[37]:


bets.info()


# * There is no NaN value

# In[38]:


bets.describe()


# * especially values columns are analyzed in spite of min, max or mean

# In[39]:


bets_maxepoch = bets.groupby(['match_id', 'odd_bookmakers', 'variable'])['odd_epoch'].max().reset_index()


# In[40]:


bets_maxepoch.head(2)


# In[41]:


bets.head(2)


# In[42]:


bets = pd.merge (bets,bets_maxepoch, how='right', on=bets_maxepoch.columns.tolist())


# In[43]:


bets_control = bets.groupby(['match_id', 'odd_bookmakers', 'odd_epoch', 'variable']).count()


#  * Data must be singular based on matchid, odd_bookmakers,odd_epoch, variable

# * Based on match_id, odd_bookmakers, odd_epoch must be 3 type of variable and 3 row count

# In[44]:


bets_pivot = pd.pivot_table(bets,index=['match_id','odd_bookmakers','odd_epoch'] ,columns='variable',values='value').reset_index()


# In[45]:


bets_pivot['prob_odd_1'] = 1/bets_pivot['odd_1']
bets_pivot['prob_odd_x'] = 1/bets_pivot['odd_x']
bets_pivot['prob_odd_2'] = 1/bets_pivot['odd_2']


# In[46]:


bets_pivot['normalization'] = bets_pivot['prob_odd_1'] + bets_pivot['prob_odd_x'] + bets_pivot['prob_odd_2']


# In[47]:


bets_pivot['norm_odd_1'] = bets_pivot['prob_odd_1'] / bets_pivot['normalization']
bets_pivot['norm_odd_2'] = bets_pivot['prob_odd_2'] / bets_pivot['normalization']
bets_pivot['norm_odd_x'] = bets_pivot['prob_odd_x'] / bets_pivot['normalization']


# In[48]:


bets_pivot = pd.merge(matches_premier, bets_pivot, how='inner', on='match_id')[bets_pivot.columns.tolist() + ['match_hometeam_score', 'match_awayteam_score', 'match_hometeam_extra_score', 'match_awayteam_extra_score']]


# In[49]:


bets_pivot['home -win'] = bets_pivot['prob_odd_1'] -  bets_pivot['prob_odd_2']


# In[50]:


bets_pivot['draw_flag'] = np.where(bets_pivot.match_hometeam_score - bets_pivot.match_awayteam_score==0, 1, 0)


# In[51]:


bin_data = bets_pivot[bets_pivot['odd_bookmakers']=='Marathonbet'][['home -win', 'draw_flag']]
bins = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
bin_data['range'] = pd.cut(bin_data['home -win'], bins=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], right=True, labels=False)+1


# In[52]:


actual = bin_data.groupby(['range']).sum()/bin_data.groupby(['range']).count()


# In[53]:


bin_ranges = []
for i in range(len(bins)-1):
    bin_ranges.append((bins[i]+ bins[i+1])/2)


# In[54]:


ax = sns.scatterplot(x="home -win", 
                       y="norm_odd_x",
                       data=bets_pivot[bets_pivot['odd_bookmakers']=='Marathonbet']).set_title('MarathonBet', fontsize = 15)

plt.plot(bin_ranges,actual['draw_flag'],'co-',
             alpha=1)


# In[55]:


ax = sns.scatterplot(x="home -win", 
                       y="norm_odd_x",
                       data=bets_pivot[bets_pivot['odd_bookmakers']=='10Bet']).set_title('10Bet', fontsize = 15)


# In[56]:


ax = sns.scatterplot(x="home -win", 
                       y="norm_odd_x",
                       data=bets_pivot[bets_pivot['odd_bookmakers']=='bwin']).set_title('bwin', fontsize = 15)


# In[57]:


ax = sns.scatterplot(x="home -win", 
                       y="norm_odd_x",
                       data=bets_pivot[bets_pivot['odd_bookmakers']=='Unibet']).set_title('Unibet', fontsize = 15)


# ## Task - 3

# In[386]:


booking.info()


# In[383]:


booking.shape


# In[384]:


booking = booking[~(booking['time'].str.contains('\+'))]


# In[388]:


booking_for10min = booking[(booking['time'].astype(int)<10) & (booking['card']=='red card')] 


# In[429]:


marathon_bet_pivot['red_card_flag'] = np.where(marathon_bet_pivot['match_id'].isin(list), 
                             1, 0)


# In[433]:


marathon_bet_pivot['after_90_flag'] = np.where((marathon_bet_pivot['match_hometeam_score'] -marathon_bet_pivot['match_hometeam_extra_score'])  -(marathon_bet_pivot['match_awayteam_score'] -marathon_bet_pivot['match_awayteam_extra_score'])!=0, 1, 0)


# In[397]:


list = booking_for10min['match_id'].tolist()

