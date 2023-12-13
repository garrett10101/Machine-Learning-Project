# SUBMISSION FORMAT FOR THE REPORTS

#  Weather Data to Predict TDS, Water Temparture, and Water Quality
**Garrett DiPalma, Nilu Sah, Isaiah Gage** 

## Project Summary

The Project is to predict given certain time, weather, and lat/lon values (or cluster) what is the predicted TDS and Water Temperature the lake.


## Problem Statement 

The problem I'm solving is if the TDS sensors, when it's turned back on, has issues, or futher down spring lake there are not that many markers however TDS and Water Temperature vary allot downstream. The idea is that if all we have is weather data, given certain location markers on the lake what is the predicted TDS and Water Temperature.


## Dataset 

There are 4 datasets I'm using, two weather datasets (G3425 (4103 * 17) and KHYI (114389 * 43)  Stations), a TDS dataset (36831 * 9) and . There will need to be some extrapolation of the data to match it into one large dataframe. One, any missing data from G3425 needs to be filled in with KHYI data. The timestamps need to be from (06/11/22 - 06/11/23) for all datasets need to be 1 second intervals. Once the data is correctly aligned with the TDS data, they will be combined into one dataframe. Timestamp will be dropped. The Final_Dataset shape with extrapolation is (31536001 * 67).