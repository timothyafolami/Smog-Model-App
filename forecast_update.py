import pandas as pd




forecasted_df = pd.read_csv('data_forecast/forecast_0_day_lag.csv')
# loading district dataset
district = pd.read_csv('Join.csv')
# selecting only location_id and district
district = district[['location_id', 'district']]
index = forecasted_df.index
# converting location_id to string
forecasted_df['location_id'] = forecasted_df['location_id'].astype(float)
district['location_id'] = district['location_id'].astype(float)
# merging forecasted data with district data  based on location_id
forecasted_df = forecasted_df.merge(district, on='location_id', how='left') 
# rewriting the column names in Alphabetical order
forecasted_df = forecasted_df.reindex(sorted(forecasted_df.columns), axis=1)
# making sure that the pollutant names start with capital letters
forecasted_df.columns = forecasted_df.columns.str.capitalize()