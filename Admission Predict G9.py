#!/usr/bin/env python
# coding: utf-8

# In[201]:


import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt


# In[202]:


os.chdir("E:\\FORE\\Term 2\\Python\\Projectdatasets_DEVP\\Group 9")
ap = pd.read_csv("1-Admission_Predict.csv")


# In[203]:


ap.head()


# In[204]:


ap.dtypes


# In[205]:


bins = [0, 25, 50, 80, 95 , 100]
names = ['Very Unlikely', 'Unlikely', 'Average Chance','Good Chance' ,'Extremely Likely']
ap['Chance of Admit'] = pd.cut(ap['Chance of Admit ']*100, bins, labels=names)
ap.head() #custom bins 0-25,26-50,51-80,81-95,96-100


# In[206]:


_=sns.displot(ap["GRE Score"], bins = 15) #distribution of GRE Scores


# In[207]:


sns.jointplot(x=ap["GRE Score"], y= ap["TOEFL Score"], kind = "reg") #Data exploration


# In[8]:


gre = ap[['GRE Score','SOP']].value_counts(normalize=True)*100 #not used


# In[9]:


gre #not used


# In[ ]:





# In[215]:


sns.pairplot(ap)

#investing relationship between variables


# In[208]:


#investigating relation between GRE Score and chance of admit
_=sns.displot(x = ap["GRE Score"],col=  ap["Chance of Admit"], data = ap)


# In[209]:


#plot to see data spread
plt.figure(figsize = (30,9))
_=sns.displot(x = ap["GRE Score"],hue =  ap["Chance of Admit"], kind = "kde", data = ap)


# In[210]:


#investigating research based outcomes on chance of admit
_=sns.displot(x = ap["GRE Score"],col=  ap["Research"],hue = "Chance of Admit", multiple = "stack",data = ap, palette = "husl")


# In[211]:


#relation between chance of admit, university rating, and chance of admit
plt.figure(figsize = (10,10))
_=sns.swarmplot(x="University Rating", y= "Chance of Admit ", data=ap, hue="Research",color ='r')


# In[212]:


#we can see that people involved in research are more likely to get a higher research score
plt.figure(figsize = (10,8))
_=sns.boxplot(x="University Rating", y= "GRE Score",hue="Research", data=ap )


# In[213]:


#relation between LOR and higher GRE Score
plt.figure(figsize = (10,8))
_=sns.boxplot(x="Research", y= "GRE Score",hue="LOR ",data=ap, palette = "cubehelix" )


# In[214]:


plt.figure(figsize = (15,12))

_=sns.scatterplot(x = "CGPA", y= "Chance of Admit ", hue = "GRE Score", style = "University Rating",
                  size = "LOR ",sizes = (15,150) , data= ap,palette = "viridis")
#relationship between chance of admit and CGPA


# In[216]:


plt.figure(figsize = (20,20))

_=sns.scatterplot(x = "CGPA", y= "Chance of Admit ", hue = "TOEFL Score", style = "Research",
                  size = "GRE Score",sizes = (20,300), data= ap,palette = "rocket_r")

#including TOEFL score to find insights into Chance of Admit


# In[16]:


ap.head()


# In[19]:


ap.head()


# In[20]:


ap2 = ap[['University Rating','LOR ','CGPA']]


# In[22]:


ap2 = ap2.pivot_table(index = 'LOR ', columns = 'University Rating', values = 'CGPA')


# In[39]:


ap2


# In[217]:


plt.figure(figsize = (15,7))
_=sns.heatmap(ap2, cmap="Spectral",cbar_kws={"orientation": "horizontal"})

#heatmap to show relation and affect of LOR, as well as average CGPA differentials in University Ratings


# In[230]:


ap['Chance of Admit %'] = ap["Chance of Admit "]*100


# In[231]:


ap.head()


# In[232]:


ap3 = ap[['University Rating','Chance of Admit','CGPA']]


# In[233]:


ap3 = ap3.pivot_table(index = 'University Rating', columns = 'Chance of Admit', values = 'CGPA')


# In[234]:


ap3


# In[235]:


plt.figure(figsize = (15,9))
_= sns.heatmap(ap3, linecolor ="black", linewidth = "0.005",cmap="YlGnBu",cbar_kws={"orientation": "horizontal"})

#heatmap to show relation and affect of University rating, as well as average CGPA to chance of admit


# In[236]:


ap.head()


# In[237]:


ap['University Rating Category'] = ap['University Rating']


# In[238]:


ap['University Rating Category'] = pd.Categorical(ap['University Rating Category'])


# In[239]:


ap.dtypes


# In[240]:



selected_region = alt.selection(type="single", encodings=['x'])

heatmap = alt.Chart(ap).mark_rect().encode(
    alt.X('GRE Score', bin=True),
    alt.Y('TOEFL Score', bin=True),
    alt.Color('CGPA',
        scale=alt.Scale(scheme='Spectral'),
        legend=alt.Legend(title='GRE Score')
    )
).properties(
    width=350
)


# In[241]:



circles = heatmap.mark_point().encode(
    alt.ColorValue('brown'),
    alt.Size('count()',
        legend=alt.Legend(title='Number of Candidates')
    )
).transform_filter(
    selected_region
)


# In[242]:


bars = alt.Chart(ap).mark_bar().encode(
    x='University Rating Category',
    y='count()',
    color=alt.condition(selected_region, alt.ColorValue("turquoise"), alt.ColorValue("grey"))
).properties(
    width=350
).add_selection(selected_region)

heatmap + circles | bars

#can filter number of candidates in each TOEFL+GRE Score bin, 
#and also find distribution of candidates according to University Ratings


# In[243]:


plt.figure(figsize = (12,12))
alt.Chart(ap).mark_circle().encode(
x='CGPA:Q',
y='Chance of Admit :Q',
color='University Rating:N',
tooltip = ['GRE Score', 'TOEFL Score']
).interactive()

#implementing interactive dashboard


# In[244]:


chart = alt.Chart(ap).mark_circle().encode(
y='CGPA',
color='Research',
tooltip = ['GRE Score', 'TOEFL Score']
).interactive()
chart1 = chart.encode(x='Chance of Admit %')
chart2 = chart.encode(x='GRE Score')
alt.vconcat(chart1, chart2)

#trying out implementation of interactivity in 2 charts 


# In[196]:


ap.head()


# In[ ]:





# In[ ]:




