#!/usr/bin/env python
# coding: utf-8

# In[36]:


# Imported Packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[37]:


#Import csv files to python as a dataframe
bets = pd.read_csv(r"C:\Users\elif.erdem\Desktop\Master\IE-582\HW-1\bets.csv") 
booking = pd.read_csv(r"C:\Users\elif.erdem\Desktop\Master\IE-582\HW-1\booking.csv") 
goals = pd.read_csv(r"C:\Users\elif.erdem\Desktop\Master\IE-582\HW-1\goals.csv") 
matches = pd.read_csv(r"C:\Users\elif.erdem\Desktop\Master\IE-582\HW-1\matches.csv") 
stats = pd.read_csv(r"C:\Users\elif.erdem\Desktop\Master\IE-582\HW-1\stats.csv") 


# * Matches dataframe consists of the info required for Q1. Hence, this df is analyzed with describe and info detailly.

# In[38]:


matches.shape
matches.info()


# * Type of columns which are with match_status, match_hometeam_name, match_awayteam_name is not numerical.
# * Match_status, match_hometeam_score, match_awayteam_score,match_hometeam_halftime_score, match_awayteam_halftime_score,match_hometeam_extra_score, match_awayteam_extra_score, match_hometeam_penalty_score, match_awayteam_penalty_score columns have missing values.
# * match_awayteam_penalty_score , match_hometeam_penalty_score  columns can be dropped since they do not include any info.
# 

# In[39]:


matches = matches.drop(columns=['match_hometeam_penalty_score', 'match_awayteam_penalty_score'])


# In[40]:


matches.describe()


# * Columns that are match_hometeam_score, match_awayteam_score, match_hometeam_halftime_score, match_awayteam_halftime_score, match_hometeam_extra_score, match_awayteam_extra_score analyzed min, max and mean values to analyze outliers

# - English premier league is analyzed for the Q1. Hence, the raw data is limited for the LeagueId=148(EnglishPremier)

# In[41]:


matches_premier= matches[matches['league_id']==148]


# In[42]:


matches_premier.head(2)


# In[43]:


matches_premier.info()


# In[44]:


matches_premier = matches_premier[matches_premier['match_status']=='Finished']


# * Match_satus=Finished is selected

# In[45]:


matches_premier = matches_premier.drop_duplicates()
matches_premier['match_id'].duplicated().unique()


# * Duplicate values and NAN values ara dropped.

# In[46]:


matches_premier.shape


# In[47]:


matches_premier = matches_premier[matches_premier['match_hometeam_score'].notnull()]


# In[48]:


matches_premier = matches_premier[matches_premier['match_awayteam_score'].notnull()]


# * Each match must contain scores

# In[50]:


matches_premier_control = matches_premier.groupby(['match_id']).count()


# In[51]:


matches_premier_control.shape


# * Data is singuler based of match_id.

# In[52]:


matches_premier = matches_premier[['match_id', 'match_hometeam_score', 'match_awayteam_score', 'match_hometeam_extra_score', 'match_awayteam_extra_score']]


# In[54]:


matches_premier['difference'] =  matches_premier['match_hometeam_score'] -  matches_premier['match_awayteam_score']


# * Only neccessary columns are selected

# In[20]:


matches_premier.describe()


# # Task 1 

# In[68]:


mean = matches_premier['match_hometeam_score'].mean()


# In[73]:


size= int(matches_premier['match_hometeam_score'].size)


# In[74]:


bins_home =int( matches_premier['match_hometeam_score'].max())


# In[76]:


range_min_home= int(matches_premier['match_hometeam_score'].min())


# In[77]:


sns.distplot(matches_premier.match_hometeam_score, kde=False, bins = bins_home) ## bins= max(match_hometeam_score)
poissonpmf = np.histogram(np.random.poisson(lam =mean, size = size),
                            bins=bins_home, range=(range_min_home, bins_home))
plt.plot(poissonpmf[0])
plt.xlabel('Home Goals')
plt.ylabel('Number of Games')


# In[78]:


bins_away =int( matches_premier['match_awayteam_score'].max())


# In[79]:


size_away= int(matches_premier['match_awayteam_score'].size)


# In[80]:


mean_away = matches_premier['match_awayteam_score'].mean()


# In[81]:


range_min_away= int(matches_premier['match_awayteam_score'].min())


# In[82]:


sns.distplot(matches_premier.match_awayteam_score, kde=False, bins = bins_away)#bins=max(match_awayteam_score)
poissonpmf = np.histogram(np.random.poisson(lam =mean_away, size = size_away),
                            bins=bins_away, range=(range_min_away, bins_away))
plt.plot(poissonpmf[0])
plt.xlabel('Away Goals')
plt.ylabel('Number of Games')


# * The first two plots are consistent with Poisson distribution

# In[237]:


sns.distplot(matches_premier.difference,  bins = 14) ##bins= max(difference) -min(difference)
plt.xlabel('Home-Away')
plt.ylabel('Number of Games')


# In[84]:


matches_premier.describe()


# # Task - 2

# In[88]:


bets.head(5)


# In[89]:


bets = bets[bets["odd_bookmakers"].isin(['10Bet', 'bwin', 'Unibet', 'Marathonbet'])]


# In[90]:


bets = bets[bets['variable'].isin(['odd_1', 'odd_x', 'odd_2'])]


# In[91]:


bets.info()


# * There is no NaN value

# In[92]:


bets.describe()


# * especially values columns are analyzed in spite of min, max or mean

# In[94]:


bets_maxepoch = bets.groupby(['match_id', 'odd_bookmakers', 'variable'])['odd_epoch'].max().reset_index()


# In[95]:


bets_maxepoch.head(2)


# In[550]:


bets.head(2)


# In[96]:


bets = pd.merge (bets,bets_maxepoch, how='right', on=bets_maxepoch.columns.tolist())


# In[552]:


bets_control = bets.groupby(['match_id', 'odd_bookmakers', 'odd_epoch', 'variable']).count()
bets_control.shape


# In[553]:


bets.shape


#  * Data must be singular based on matchid, odd_bookmakers,odd_epoch, variable

# * Based on match_id, odd_bookmakers, odd_epoch must be 3 type of variable and 3 row count

# In[97]:


bets_pivot = pd.pivot_table(bets,index=['match_id','odd_bookmakers','odd_epoch'] ,columns='variable',values='value').reset_index()


# In[99]:


bets_pivot.head(2)


# In[100]:


bets_pivot['prob_odd_1'] = 1/bets_pivot['odd_1']
bets_pivot['prob_odd_x'] = 1/bets_pivot['odd_x']
bets_pivot['prob_odd_2'] = 1/bets_pivot['odd_2']


# In[101]:


bets_pivot['normalization'] = bets_pivot['prob_odd_1'] + bets_pivot['prob_odd_x'] + bets_pivot['prob_odd_2']


# In[103]:


bets_pivot['norm_odd_1'] = bets_pivot['prob_odd_1'] / bets_pivot['normalization']
bets_pivot['norm_odd_2'] = bets_pivot['prob_odd_2'] / bets_pivot['normalization']
bets_pivot['norm_odd_x'] = bets_pivot['prob_odd_x'] / bets_pivot['normalization']


# In[111]:


bets_pivot.head(2)


# In[110]:


matches_premier.head(2)


# In[109]:


bets_pivot = pd.merge(matches_premier, bets_pivot, how='inner', on='match_id')[bets_pivot.columns.tolist() + ['match_hometeam_score', 'match_awayteam_score', 'match_hometeam_extra_score', 'match_awayteam_extra_score']]


# In[113]:


bets_pivot['home-away'] = bets_pivot['prob_odd_1'] -  bets_pivot['prob_odd_2']


# In[114]:


bets_pivot['draw_flag'] = np.where(bets_pivot.match_hometeam_score - bets_pivot.match_awayteam_score==0, 1, 0)


# In[125]:


bets_pivot.head(2)


# In[117]:


bin_data = bets_pivot[bets_pivot['odd_bookmakers']=='Marathonbet'][['home-away', 'draw_flag']]
bins = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
bin_data['range'] = pd.cut(bin_data['home-away'], bins=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], right=True, labels=False)+1


# In[130]:


actual = bin_data.groupby(['range']).sum()/bin_data.groupby(['range']).count()


# In[131]:


bin_ranges = []
for i in range(len(bins)-1):
    bin_ranges.append((bins[i]+ bins[i+1])/2)


# In[236]:


ax = sns.scatterplot(x="home-away", 
                       y="norm_odd_x",
                       data=bets_pivot[bets_pivot['odd_bookmakers']=='Marathonbet']).set_title('MarathonBet', fontsize = 15)

plt.plot(bin_ranges,actual['draw_flag'])
plt.xlabel('P(Home)-P(Away)')
plt.ylabel('P(Draw)')
plt.legend(loc='best', labels=['actual_outcome', 'bookmakers_probabilties'])


# In[139]:


bin_data = bets_pivot[bets_pivot['odd_bookmakers']=='10Bet'][['home-away', 'draw_flag']]
bins = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
bin_data['range'] = pd.cut(bin_data['home-away'], bins=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], right=True, labels=False)+1


# In[140]:


actual = bin_data.groupby(['range']).sum()/bin_data.groupby(['range']).count()


# In[141]:


bin_ranges = []
for i in range(len(bins)-1):
    bin_ranges.append((bins[i]+ bins[i+1])/2)


# In[227]:


ax = sns.scatterplot(x="home-away", 
                       y="norm_odd_x",
                       data=bets_pivot[bets_pivot['odd_bookmakers']=='10Bet']).set_title('10Bet', fontsize = 15)

plt.plot(bin_ranges,actual['draw_flag'])
plt.xlabel('P(Home)-P(Away)')
plt.ylabel('P(Draw)')
plt.legend(loc='best', labels=['actual_outcome', 'bookmakers_probabilties'])


# In[143]:


bin_data = bets_pivot[bets_pivot['odd_bookmakers']=='bwin'][['home-away', 'draw_flag']]
bins = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
bin_data['range'] = pd.cut(bin_data['home-away'], bins=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], right=True, labels=False)+1
actual = bin_data.groupby(['range']).sum()/bin_data.groupby(['range']).count()
bin_ranges = []
for i in range(len(bins)-1):
    bin_ranges.append((bins[i]+ bins[i+1])/2)


# In[228]:


ax = sns.scatterplot(x="home-away", 
                       y="norm_odd_x",
                       data=bets_pivot[bets_pivot['odd_bookmakers']=='bwin']).set_title('bwin', fontsize = 15)

plt.plot(bin_ranges,actual['draw_flag'])
plt.xlabel('P(Home)-P(Away)')
plt.ylabel('P(Draw)')
plt.legend(loc='best', labels=['actual_outcome', 'bookmakers_probabilties'])


# In[146]:


bin_data = bets_pivot[bets_pivot['odd_bookmakers']=='Unibet'][['home-away', 'draw_flag']]
bins = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
bin_data['range'] = pd.cut(bin_data['home-away'], bins=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], right=True, labels=False)+1
actual = bin_data.groupby(['range']).sum()/bin_data.groupby(['range']).count()
bin_ranges = []
for i in range(len(bins)-1):
    bin_ranges.append((bins[i]+ bins[i+1])/2)


# In[229]:


ax = sns.scatterplot(x="home-away", 
                       y="norm_odd_x",
                       data=bets_pivot[bets_pivot['odd_bookmakers']=='Unibet']).set_title('Unibet', fontsize = 15)


plt.plot(bin_ranges,actual['draw_flag'])
plt.xlabel('P(Home)-P(Away)')
plt.ylabel('P(Draw)')
plt.legend(loc='best', labels=['actual_outcome', 'bookmakers_probabilties'])


# * Odds has not been well-aligned with the actual outcome of the matches in any case
# 

# ## Task - 3

# In[386]:


booking.info()


# In[150]:


booking.shape


# In[152]:


booking = booking[~(booking['time'].str.contains('\+'))]


# In[153]:


booking_for10min = booking[(booking['time'].astype(int)<10) & (booking['card']=='red card')] 


# In[158]:


list_red=booking_for10min['match_id']


# In[159]:


list_red


# In[160]:


bets_pivot['red_card_flag'] = np.where(bets_pivot['match_id'].isin(list_red), 
                             1, 0)


# In[163]:


goals.info()


# In[171]:


goals_after_90 = goals[goals['time']>'90']


# In[175]:


goals_after_90['Home_Score']=[int(i.split(' - ', 1)[0]) for i in goals_after_90['score']]
goals_after_90['Away_Score']=[int(i.split(' - ', 1)[1]) for i in goals_after_90['score']]


# In[180]:


goals_after_90['Is_Draw'] = np.where(goals_after_90['Home_Score'] - goals_after_90['Away_Score']==0,
                             1, 0)


# In[182]:


goals_after_90['Is_Draw_Previous'] =  np.where(abs(goals_after_90['Home_Score'] - goals_after_90['Away_Score'])==1,
                             1, 0)


# In[192]:


goals_after = goals_after_90[(goals_after_90['Is_Draw']==1) | (goals_after_90['Is_Draw_Previous']==1) ]


# In[194]:


bets_pivot.head(2)


# In[195]:


goals_after.head(2)


# In[205]:


bets_final = pd.merge(bets_pivot, goals_after, how='left', on='match_id')[bets_pivot.columns.tolist() + ['Is_Draw', 'Is_Draw_Previous'] ]


# In[206]:


bets_final.head()


# In[207]:


bets_final = bets_final[~((bets_final['red_card_flag']==1) | (bets_final['Is_Draw']==1) 
                        |(bets_final['Is_Draw_Previous']==1)) ]


# In[230]:


bin_data = bets_final[bets_final['odd_bookmakers']=='Unibet'][['home-away', 'draw_flag']]
bins = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
bin_data['range'] = pd.cut(bin_data['home-away'], bins=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], right=True, labels=False)+1
actual = bin_data.groupby(['range']).sum()/bin_data.groupby(['range']).count()
bin_ranges = []
for i in range(len(bins)-1):
    bin_ranges.append((bins[i]+ bins[i+1])/2)
ax = sns.scatterplot(x="home-away", 
                       y="norm_odd_x",
                       data=bets_final[bets_final['odd_bookmakers']=='Unibet']).set_title('Unibet', fontsize = 15)


plt.plot(bin_ranges,actual['draw_flag'])
plt.xlabel('P(Home)-P(Away)')
plt.ylabel('P(Draw)')
plt.legend(loc='best', labels=['actual_outcome', 'bookmakers_probabilties'])


# In[231]:


bin_data = bets_final[bets_final['odd_bookmakers']=='bwin'][['home-away', 'draw_flag']]
bins = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
bin_data['range'] = pd.cut(bin_data['home-away'], bins=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], right=True, labels=False)+1
actual = bin_data.groupby(['range']).sum()/bin_data.groupby(['range']).count()
bin_ranges = []
for i in range(len(bins)-1):
    bin_ranges.append((bins[i]+ bins[i+1])/2)
ax = sns.scatterplot(x="home-away", 
                       y="norm_odd_x",
                       data=bets_final[bets_final['odd_bookmakers']=='bwin']).set_title('bwin', fontsize = 15)


plt.plot(bin_ranges,actual['draw_flag'])
plt.xlabel('P(Home)-P(Away)')
plt.ylabel('P(Draw)')
plt.legend(loc='best', labels=['actual_outcome', 'bookmakers_probabilties'])


# In[233]:


bin_data = bets_final[bets_final['odd_bookmakers']=='Marathonbet'][['home-away', 'draw_flag']]
bins = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
bin_data['range'] = pd.cut(bin_data['home-away'], bins=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], right=True, labels=False)+1
actual = bin_data.groupby(['range']).sum()/bin_data.groupby(['range']).count()
bin_ranges = []
for i in range(len(bins)-1):
    bin_ranges.append((bins[i]+ bins[i+1])/2)
ax = sns.scatterplot(x="home-away", 
                       y="norm_odd_x",
                       data=bets_final[bets_final['odd_bookmakers']=='Marathonbet']).set_title('Marathonbet', fontsize = 15)


plt.plot(bin_ranges,actual['draw_flag'])
plt.xlabel('P(Home)-P(Away)')
plt.ylabel('P(Draw)')
plt.legend(loc='best', labels=['actual_outcome', 'bookmakers_probabilties'])


# In[234]:


bin_data = bets_final[bets_final['odd_bookmakers']=='10Bet'][['home-away', 'draw_flag']]
bins = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
bin_data['range'] = pd.cut(bin_data['home-away'], bins=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], right=True, labels=False)+1
actual = bin_data.groupby(['range']).sum()/bin_data.groupby(['range']).count()
bin_ranges = []
for i in range(len(bins)-1):
    bin_ranges.append((bins[i]+ bins[i+1])/2)
ax = sns.scatterplot(x="home-away", 
                       y="norm_odd_x",
                       data=bets_final[bets_final['odd_bookmakers']=='10Bet']).set_title('10Bet', fontsize = 15)


plt.plot(bin_ranges,actual['draw_flag'])
plt.xlabel('P(Home)-P(Away)')
plt.ylabel('P(Draw)')
plt.legend(loc='best', labels=['actual_outcome', 'bookmakers_probabilties'])


# * for each bookmaker, eventhough we can see a slight improvement compated to Task 2 as we expected by cleaning noise, the odds has not been well-aligned with the actual outcome. Since the number of red cards in the first ten minutes are very low, they were not effective as much as last minute goals.
