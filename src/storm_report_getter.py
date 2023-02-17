#!/usr/bin/env python
# coding: utf-8

# ## Download Observed Storm Reports 
# 
# This notebook uses the StormReportDownloader from `wofs_ml_severe` to download the 
# Storm Events database based on the specified years. The Storm Events database 
# are observed storm reports that have been verified/vetted. To learn about the Storm Events database, [click here](https://www.ncdc.noaa.gov/stormevents/)
# 

# In[1]:


# Download https://github.com/WarnOnForecast/wofs_ml_severe
# change your system path.
import sys
sys.path.append('/home/monte.flora/python_packages/wofs_ml_severe')
from wofs_ml_severe.data_pipeline.storm_report_downloader import StormReportDownloader

from glob import glob
from os.path import join


# In[2]:


OUTDIR = '/work/mflora/LSRS'
# The WoFS dataset spans from 2017-2021. 
years = ['2017', '2018', '2019', '2020', '2021', '2022']


# In[3]:


# This function will download the StormEvent files per year.
downloader = StormReportDownloader(OUTDIR)
downloader.get_storm_events(years)


# In[ ]:


# The StormEvent database stores the timing of events in local time rather than UTC. 
# The `format_data` correctly formats the timing and combines all the individual StormEvent 
# files into a single file. This function returns a pandas.Dataframe
paths = glob(join(OUTDIR, 'StormEvents_details*'))
df = downloader.format_data(paths)


df.reset_index(drop=True, inplace=True)
df.to_csv(join(OUTDIR, f'StormEvents_{years[0]}-{years[-1]}.csv'))

