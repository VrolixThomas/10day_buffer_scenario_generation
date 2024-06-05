import pandas as pd
import datetime
import inspect
from pathlib import Path
import sys
import os
import pycatch22

class BufferAttributesCalculator:
    def __init__(self, df):
        self.df = df
        self.column_functions = {}
        
        # Get all the methods of the class
        methods = inspect.getmembers(self, predicate=lambda x: inspect.ismethod(x) and not x.__name__.startswith('__'))

        # Filter and add relevant methods to column_functions
        for name, method in methods:
            if name.startswith("calculate_"):
                self.column_functions[name[len("calculate_"):]] = method

    
    
    def calculate_total_consumption_buffer(self, buffer_data):
        return buffer_data.sum().sum()
            
    def calculate_highest(self, buffer_data):
        return buffer_data.max().max()

    def calculate_average(self, buffer_data):
        return buffer_data.mean().mean()
    
    def calculate_std(self, buffer_data):
        return buffer_data.std().std()
    
    def calculate_skewness(self, buffer_data):
        return buffer_data.skew().skew()
    
    def calculate_kurtosis(self, buffer_data):
        return buffer_data.kurt().kurt()
    
    def calculate_median(self, buffer_data):
        return buffer_data.median().median()
    
    def calculate_min(self, buffer_data):
        return buffer_data.min().min()

    def filter_data_by_date_and_meter(self, meter_id, start_date):
        end_date = start_date + datetime.timedelta(days=10)
        date_mask = (self.df.index.get_level_values(1) >= start_date) & (self.df.index.get_level_values(1) < end_date)
        meter_mask = self.df.index.get_level_values(0) == meter_id
        filtered_data = self.df[date_mask & meter_mask]

        return filtered_data
    
    def weekday_mask(self, buffer_data):
        return pd.to_datetime(buffer_data.index.get_level_values(1)).day_of_week < 5
    
    def weekend_mask(self, buffer_data):
        
        return pd.to_datetime(buffer_data.index.get_level_values(1)).day_of_week >= 5
    
    def calculate_morning_peak(self, buffer_data):
        selected_columns = buffer_data.loc[:, '2010-01-01 06:00:00':'2010-01-01 11:30:00']
        max_values = selected_columns.max(axis=1)
        # Calculate the average of the max values
        return max_values.mean()
    
    def calculate_lunch_peak(self, buffer_data):
        selected_columns = buffer_data.loc[:, '2010-01-01 11:30:00':'2010-01-01 14:30:00']
        max_values = selected_columns.max(axis=1)
        # Calculate the average of the max values
        return max_values.mean()
    
    def calculate_afternoon_peak(self, buffer_data):
        selected_columns = buffer_data.loc[:, '2010-01-01 14:30:00':'2010-01-01 18:00:00']
        max_values = selected_columns.max(axis=1)
        # Calculate the average of the max values
        return max_values.mean()
    
    def calculate_evening_peak(self, buffer_data):
        selected_columns = buffer_data.loc[:, '2010-01-01 18:00:00':'2010-01-01 21:30:00']
        max_values = selected_columns.max(axis=1)
        # Calculate the average of the max values
        return max_values.mean()
    
    def calculate_night_peak(self, buffer_data):
        columns_midnight_to_six = buffer_data.loc[:, '2010-01-01 00:00:00':'2010-01-01 06:00:00']
        columns_night = buffer_data.loc[:, '2010-01-01 21:30:00':'2010-01-01 23:30:00']
        selected_columns = pd.concat([columns_midnight_to_six, columns_night], axis=1)
        max_values = selected_columns.max(axis=1)
        # Calculate the average of the max values
        return max_values.mean()
    
    def calculate_morning_total_consumption(self, buffer_data):
        selected_columns = buffer_data.loc[:, '2010-01-01 06:00:00':'2010-01-01 11:30:00']
        return selected_columns.sum().sum()
    
    def calculate_lunch_total_consumption(self, buffer_data):
        selected_columns = buffer_data.loc[:, '2010-01-01 11:30:00':'2010-01-01 14:30:00']
        return selected_columns.sum().sum()
    
    def calculate_afternoon_total_consumption(self, buffer_data):
        selected_columns = buffer_data.loc[:, '2010-01-01 14:30:00':'2010-01-01 18:00:00']
        return selected_columns.sum().sum()
    
    def calculate_evening_total_consumption(self, buffer_data):
        selected_columns = buffer_data.loc[:, '2010-01-01 18:00:00':'2010-01-01 21:30:00']
        return selected_columns.sum().sum() 
    
    def calculate_night_total_consumption(self, buffer_data):    
        columns_midnight_to_six = buffer_data.loc[:, '2010-01-01 00:00:00':'2010-01-01 06:00:00']
        columns_night = buffer_data.loc[:, '2010-01-01 21:30:00':'2010-01-01 23:30:00']
        selected_columns = pd.concat([columns_midnight_to_six, columns_night], axis=1)
        return selected_columns.sum().sum()

    def make_buffer_attributes(self, result_path, meter_ids=None, start_date=datetime.date(2010, 1, 1)):
        if Path(result_path).exists():
            return
                
        if meter_ids is None:
            meter_ids = set(self.df.index.get_level_values(0))
        count = 0
        total = len(meter_ids)
        columns = list(self.column_functions.keys())
        buffer_attributes_df = pd.DataFrame(columns=columns)
        buffer_attributes_df.index.name = "meterID"

        for meter_id in meter_ids:
            if count % 200 == 0:
                print(f"Running meter {meter_id} ({count}/{total})")
            count+=1
            buffer_data = self.filter_data_by_date_and_meter(meter_id, start_date)
            row_values = [self.column_functions[column](buffer_data) for column in columns]
            buffer_attributes_df.loc[meter_id] = row_values
        buffer_attributes_df.to_pickle(result_path)
        
    def make_buffer_attributes_all(self, result_path, meter_ids=None, start_date=datetime.date(2010, 1, 1)):
        if Path(result_path).exists():
            return
    
        if meter_ids is None:
            meter_ids = set(self.df.index.get_level_values(0))
        
        count = 0
        total = len(meter_ids)
        columns = list(self.column_functions.keys())
        
        # Create column names for weekday and weekend versions
        columns_all = columns
        columns_weekday = [f"{column}_weekday" for column in columns]
        columns_weekend = [f"{column}_weekend" for column in columns]
        columns_catch22 = ['mode_5', 'mode_10', 'acf_timescale', 'acf_first_min', 'ami2', 'trev', 'high_fluctuation', 'stretch_high', 'transition_matrix', 'periodicity', 'embedding_dist', 'ami_timescale', 'whiten_timescale', 'outlier_timing_pos', 'outlier_timing_neg', 'centroid_freq', 'stretch_decreasing', 'entropy_pairs', 'rs_range', 'dfa', 'low_freq_power', 'forecast_error']
        
        # Create an empty DataFrame with the new column names
        buffer_attributes_df = pd.DataFrame(columns=columns_all + columns_weekday + columns_weekend + columns_catch22)
        buffer_attributes_df.index.name = "meterID"

        for meter_id in meter_ids:
            if count % 200 == 0:
                print(f"Running meter {meter_id} ({count}/{total})")
            count += 1
            buffer_data_all = self.filter_data_by_date_and_meter(meter_id, start_date)
            buffer_data_weekday = buffer_data_all[self.weekday_mask(buffer_data_all)]
            buffer_data_weekend = buffer_data_all[self.weekend_mask(buffer_data_all)]
            
            
            # Calculate values for all days
            row_values_all = [self.column_functions[column](buffer_data_all) for column in columns_all]
            
            # Calculate values for weekdays
            row_values_weekday = [self.column_functions[column](buffer_data_weekday) for column in columns_all]
            
            # Calculate values for weekends
            row_values_weekend = [self.column_functions[column](buffer_data_weekend) for column in columns_all]
            
            values_list = buffer_data_all.values.flatten().tolist()
            buffer_data_catch22 = pycatch22.catch22_all(values_list,short_names=True)['values']
            
            # Combine all values into a single row and append to the DataFrame
            row_values_combined = row_values_all + row_values_weekday + row_values_weekend + buffer_data_catch22
            buffer_attributes_df.loc[meter_id] = row_values_combined
        
        buffer_attributes_df.to_pickle(result_path)

if __name__ == "__main__":
    print("starting")
    current_dir = os.getcwd()
    london_data_dir = os.path.join(current_dir, 'Data/London_dataset/preprocessed')
    irish_data_dir = os.path.join(current_dir, 'Data/Irish_dataset/preprocessed')
    
    data = pd.read_pickle(os.path.join(irish_data_dir,'daily_data_df_filtered.pkl'))

    # Example usage:
    for i in range(2,13):
        print(f"Starting month {i}")
        start_date = datetime.date(2010, i, 1)
        calculator = BufferAttributesCalculator(data)
        calculator.make_buffer_attributes_all(os.path.join(irish_data_dir,f'buffer_attributes_month_{i}_all_catch22_df.pkl'),start_date=start_date)
    print("done")


    