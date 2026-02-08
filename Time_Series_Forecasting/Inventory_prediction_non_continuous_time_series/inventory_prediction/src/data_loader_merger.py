import pandas as pd
import os

def load_raw_data(file_path):

    return pd.read_csv(file_path)

#load datsets

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

raw_data_folder_path = os.path.join(base_path,'data','raw','m5-forecasting-accuracy')

calendar_df = load_raw_data(os.path.join(raw_data_folder_path, 'calendar.csv'))
print('calendar dataset loaded')

sell_prices_df = load_raw_data(os.path.join(raw_data_folder_path, 'sell_prices.csv'))
print('sell prices dataset loaded')

sales_train_valid_df = load_raw_data(os.path.join(raw_data_folder_path, 'sales_train_validation.csv'))
print('sales train validation dataset loaded')

items_to_select = ['FOODS_3_586','FOODS_3_090','FOODS_3_555','FOODS_3_252','FOODS_3_587']

#process sales_train_val

# #select top 5 items from store tx_1
sales_train_validation_df_store_subset_tx1 = sales_train_valid_df[sales_train_valid_df['store_id'] == 'TX_1']


sales_train_validation_df_store_subset_tx1_top_5_items = sales_train_validation_df_store_subset_tx1[sales_train_validation_df_store_subset_tx1['item_id'].isin(items_to_select)]

sales_train_validation_df_store_subset_tx1_top_5_items_items_only = sales_train_validation_df_store_subset_tx1_top_5_items.drop(columns=['id','dept_id', 'cat_id', 'store_id', 'state_id'])

#transpose the dataframe to have items as columns
sales_train_validation_df_store_subset_tx1_top_5_items_items_only_transposed = sales_train_validation_df_store_subset_tx1_top_5_items_items_only.T

#convert the index to a column
sales_train_validation_df_store_subset_tx1_top_5_items_items_only_transposed.reset_index(inplace=True)

#set row 0 as the header
sales_train_validation_df_store_subset_tx1_top_5_items_items_only_transposed.columns = sales_train_validation_df_store_subset_tx1_top_5_items_items_only_transposed.iloc[0]

#drop the first row
sales_train_validation_df_store_subset_tx1_top_5_items_items_only_transposed.drop(index=0, inplace=True)

#reorder columns
sales_train_validation_df_store_subset_tx1_top_5_items_items_only_transposed = sales_train_validation_df_store_subset_tx1_top_5_items_items_only_transposed[['item_id']+items_to_select]

tx1_top_5_long = sales_train_validation_df_store_subset_tx1_top_5_items_items_only_transposed.copy()
tx1_top_5_long.rename(columns={'item_id': 'day_id'}, inplace=True)

print('sales train validation dataset processed')


# #merge calendar and items sold

tx1_top_5_long_calendar_info = tx1_top_5_long.merge(calendar_df, left_on='day_id', right_on='d', how='left')
tx1_top_5_long_calendar_info.head(5)
# #convert date column to datetime

# # Set the date as the index
# tx1_top_5_long_calendar_info.set_index('date', inplace=True)
# Drop the 'd' column as it's no longer needed
tx1_top_5_long_calendar_info.drop(columns=['d','day_id'], inplace=True)

print('calendar and items sold merged')

#subset sales price data for items and store of interest
sell_prices_df_store_subset_tx1 = sell_prices_df[sell_prices_df['store_id'] == 'TX_1']
sell_prices_df_store_subset_tx1_top_5_items = sell_prices_df_store_subset_tx1[sell_prices_df_store_subset_tx1['item_id'].isin(items_to_select)]

#convert sell prices to use wm_yr_wk as index, unique items as columns and sell prices as values
# print(sell_prices_df_store_subset_tx1_top_5_items.head(5))
sell_prices_pivot = sell_prices_df_store_subset_tx1_top_5_items.pivot(index='wm_yr_wk', columns='item_id', values='sell_price')

print('sell prices dataset processed')
#merge in sales price data with amount sold and calendar info
tx1_top_5_long_cal_sell_prices = tx1_top_5_long_calendar_info.merge(sell_prices_pivot, left_on='wm_yr_wk', right_on='wm_yr_wk', how='left')

print('sell prices merged in with calendar, amount sold')

#change col names
new_col_names = {}
for item in items_to_select:
    new_col_names[item+'_x'] = item+'_amount_sold'
    new_col_names[item+'_y'] = item+'_sell_price'
tx1_top_5_long_cal_sell_prices.rename(columns=new_col_names, inplace=True)

tx1_top_5_long_cal_sell_prices['date'] = pd.to_datetime(tx1_top_5_long_cal_sell_prices['date'])
tx1_top_5_long_cal_sell_prices.set_index('date', inplace=True)

print(f'final columns: {tx1_top_5_long_cal_sell_prices.columns}')

merged_dataset_location = os.path.join(base_path, 'data', 'merged', 'tx1_top_5_long_cal_info_sell_prices.csv')

tx1_top_5_long_cal_sell_prices.to_csv(merged_dataset_location, index=True)
print('merged dataset created and saved to \n{merged_dataset_location}')

print('Data loading and merging completed successfully')
