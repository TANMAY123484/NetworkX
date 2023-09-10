# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 13:11:20 2023

@author: Admin
"""

import pandas as pd


df=pd.read_csv("OD__pois_allocated_feeder_stops1.csv")

columns = ['origin_id', 'destination_id', 'total_cost']
df= df[columns]


df2=pd.read_csv("OD_matrix_existing_feeder_stops.csv")
df2=df2.iloc[:,[0,1,5]]

df2=df2.fillna(0)
df3=pd.read_csv("intersection_feeder_ward_before1.csv")


nearest_feeders2 = df.loc[df.groupby('origin_id')['total_cost'].idxmin()]

# Merge to combine distances
result = nearest_feeders2.merge(df2, left_on='destination_id', 
                                right_on='origin_id', suffixes=('_poi_feeder', '_feeder_feeder'))
result.to_csv("result.csv")
# Calculate the new distance
result['combined_distance'] = result['total_cost_poi_feeder'] + result['total_cost_feeder_feeder']

# Keeping only necessary columns
final_df = result[['origin_id_poi_feeder', 'destination_id_feeder_feeder', 'combined_distance']]
final_df.columns = ['origin_id', 'destination_id', 'total_cost']


final_df['time'] = final_df['total_cost'].apply(lambda x: x/3600 if x < 500 else x/20000)
final_df_1 = final_df[final_df['time']<1]
final_df_1.to_csv("poi_feeder_final_distance.csv")

final_df_1 = final_df_1.merge(df3[['bus_stop_i', 'id']], 
                              left_on='destination_id', 
                              right_on='bus_stop_i', 
                              how='left').drop('bus_stop_i', axis=1)
final_df_1.rename(columns={'id': 'ward_id'}, inplace=True)

ward_counts = final_df_1['ward_id'].value_counts()
print(ward_counts)

accessible_pois = final_df_1[final_df_1['time'] < 1]
poi_counts_per_feeder = accessible_pois.groupby('destination_id')['origin_id'].nunique()
ward_accessibility = df3.merge(poi_counts_per_feeder, left_on='bus_stop_i', right_index=True, how='left').fillna(0)

ward_accessibility_ranking = ward_accessibility.groupby('id').sum()['origin_id']
