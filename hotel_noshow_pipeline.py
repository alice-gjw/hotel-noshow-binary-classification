import sqlite3
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from datetime import datetime
from scipy import stats
import matplotlib.pyplot as plt




# 1) DATA RETRIEVAL
# df = get_noshow_data()
# print(df.head())
def get_noshow_data(db_path='data/noshow.db'):
    try:
        conn = sqlite3.connect('data/noshow.db')
        query = "SELECT * FROM noshow"
        df = pd.read_sql(query, conn)
        return df
    finally:
        conn.close()


def preprocess_data(df):
    df = handle_missing_values(df)
    df = datatype_conversion(df)
    df = feature_engineering(df)
    df = encoding(df)
    df = handle_outliers(df)
    df = scale_features(df)
    df = select_features(df)
    
    return df

# data_preprocessing

# 2) MISSING VALUES 
def handle_missing_values(df):
    missing_values = df.isnull().sum()
    missing_percentage = 100 * df.isnull().sum() / len(df)
    missing_table = pd.concat([missing_values, missing_percentage], axis=1, keys=['Missing Count', 'Missing Percentage'])

    new_df = df.copy()

    # num_adults, num_children. Don't know why I was doing that here.
    num_map = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5}
    new_df['num_adults'] = new_df['num_adults'].replace(num_map)
    new_df['num_adults'] = pd.to_numeric(new_df['num_adults'], errors='coerce')
    new_df['num_adults'] = new_df['num_adults'].astype('Int64')
    new_df['num_children'] = pd.to_numeric(df['num_children'], errors='coerce')
    new_df['num_children'] = np.round(new_df['num_children'])
    new_df['num_children'] = new_df['num_children'].astype('Int64')

    # cleaning price data to use as prediction for room type prediction
    new_df['price'] = new_df['price'].replace('None', np.nan)
    new_df['price'] = new_df['price'].astype(str).str.replace(r'[^\d.]', '', regex=True)
    new_df['price'] = pd.to_numeric(new_df['price'], errors='coerce')
    new_df['price_sgd'] = new_df.apply(
        lambda row: row['price'] * 1.35 if 'USD' in str(row['price']) else row['price'], 
        axis=1
    )

    # room imputation
    room_features = ['price_sgd', 'num_adults', 'num_children', 'branch']
    known_rooms = new_df.dropna(subset=['room']).copy()
    missing_rooms = new_df[new_df['room'].isnull()].copy()
    X = known_rooms[room_features].copy()
    y = known_rooms['room'].copy()
    X = X.dropna()
    y = y[X.index]

    numeric_features = ['price_sgd', 'num_adults', 'num_children']
    categorical_features = ['branch']
    missing_rooms['branch'] = missing_rooms['branch'].replace({None: np.nan})

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ])

    X = preprocessor.fit_transform(X)
    feature_names = (
        numeric_features + 
        preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(['branch']).tolist()
    )

    X = pd.DataFrame(X, columns=feature_names)

    le_room = LabelEncoder()
    y = le_room.fit_transform(y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    X_predict = missing_rooms[room_features]
    X_predict = preprocessor.transform(X_predict)
    X_predict = pd.DataFrame(X_predict, columns=feature_names)

    predicted_rooms = le_room.inverse_transform(model.predict(X_predict))
    new_df.loc[new_df['room'].isnull(), 'room'] = predicted_rooms
    null_count = new_df['room'].isnull().sum()

    # using room to predict price for imputation
    price_features = ['room', 'num_adults', 'num_children', 'branch']

    known_prices = new_df.dropna(subset=['price_sgd']).copy()
    missing_prices = new_df[new_df['price_sgd'].isnull()].copy()

    X_price = known_prices[price_features]
    y_price = known_prices['price_sgd']

    price_preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), ['num_adults', 'num_children']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['room', 'branch'])
    ])

    price_model = Pipeline([
        ('preprocessor', price_preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    price_model.fit(X_price, y_price)

    X_price_predict = missing_prices[price_features]
    predicted_prices = price_model.predict(X_price_predict)

    new_df.loc[new_df['price_sgd'].isnull(), 'price_sgd'] = predicted_prices

    # imputing for the rest of the columns
    new_df = new_df.drop('price', axis=1)

    numeric_columns = ['no_show', 'arrival_day', 'checkout_day', 'price_sgd', 'num_adults', 'num_children']
    for col in numeric_columns:
        new_df[col] = new_df[col].fillna(new_df[col].median())

    categorical_columns = ['branch', 'country', 'room', 'platform', 'first_time', 'booking_month', 'arrival_month', 'checkout_month']
    for col in categorical_columns:
        new_df[col] = new_df[col].fillna(new_df[col].mode()[0])

    new_nan_count = new_df.isna().sum()

    pass



# 3) DATA TYPE CONVERSION

def datatype_conversion(df):
    # converting floats to int
    new_df['no_show'] = new_df['no_show'].astype('Int64')
    new_df['arrival_day'] = new_df['arrival_day'].astype('Int64')
    new_df['checkout_day'] = new_df['checkout_day'].astype('Int64')

    month_columns = ['booking_month', 'arrival_month', 'checkout_month']
    month_order =['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

    for col in month_columns:
        new_df[col] = pd.Categorical(new_df[col], categories=month_order, ordered=True)

    new_df['first_time'] = new_df['first_time'].map({'Yes': 1, 'No': 0}).astype(int)

    for col in ['branch', 'country', 'room', 'platform']:
        new_df[col] = new_df[col].astype('category')
    
    pass




# 4) FEATURE ENGINEERING
# month_difference variable creation between booking_month and arrival_month

def feature_engineering(df):
        
    new_df = new_df.dropna()

    month_to_num = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
    }

    new_df['booking_month_num'] = new_df['booking_month'].map(month_to_num).astype(int)
    new_df['arrival_month_num'] = new_df['arrival_month'].map(month_to_num).astype(int)

    new_df['month_difference'] = (new_df['arrival_month_num'] - new_df['booking_month_num']) % 12

    # calculating stay_length from arrival_day to checkout_day
    from datetime import datetime

    def create_date(month, day, year=2023):
        month_num = month_to_num[month]
        try:
            return datetime(year, month_num, day)
        except ValueError:
            return None

    new_df['arrival_date'] = new_df.apply(lambda row: create_date(row['arrival_month'], row['arrival_day']), axis=1)
    new_df['checkout_date'] = new_df.apply(lambda row: create_date(row['checkout_month'], row['checkout_day']), axis=1)

    invalid_arrival = new_df[new_df['arrival_date'].isnull()]
    invalid_checkout = new_df[new_df['checkout_date'].isnull()]
    new_df = new_df.dropna(subset=['arrival_date', 'checkout_date'])

    new_df['stay_length'] = (new_df['checkout_date'] - new_df['arrival_date']).dt.days
    new_df.loc[new_df['stay_length'] < 0, 'stay_length'] += 365
    
    pass




# 5) ENCODING
def encoding(df):
    
    new_df = new_df.drop(['booking_month', 'arrival_month', 'checkout_month'], axis=1)
    columns_to_encode = ['branch', 'country', 'room', 'platform']
    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_columns = onehot_encoder.fit_transform(df[columns_to_encode])

    new_column_names = onehot_encoder.get_feature_names_out(columns_to_encode)
    encoded_df = pd.DataFrame(encoded_columns, columns=new_column_names, index=df.index)
    new_df_encoded = pd.concat([new_df.drop(columns=columns_to_encode), encoded_df], axis=1)



# 6) OUTLIERS
def handle_outliers(df):

    def detect_outliers_zscore(data, threshold=3):
        z_scores = np.abs(stats.zscore(data))
        return z_scores > threshold

    outliers_price = detect_outliers_zscore(new_df_encoded['price_sgd'])
    outliers_month_diff = detect_outliers_zscore(new_df_encoded['month_difference'])
    outliers_stay = detect_outliers_zscore(new_df_encoded['stay_length'])

    def detect_outliers_iqr(data):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (data < lower_bound) | (data > upper_bound)

    outliers_price_iqr = detect_outliers_iqr(new_df_encoded['price_sgd'])
    outliers_month_diff_iqr = detect_outliers_iqr(new_df_encoded['month_difference'])
    outliers_stay_iqr = detect_outliers_iqr(new_df_encoded['stay_length'])


    # I just kind of gave up here, come back and read after. Just removed all the NaN values
    total_rows = len(new_df_encoded)
    nan_percentage = (7321 / total_rows) * 100

    def remove_outliers_zscore(data, threshold=3):
        data_clean = data.dropna()
        z_scores = np.abs(stats.zscore(data_clean))
        mask = z_scores < threshold
        return data_clean[mask]

    # what even is a boolean mask of values to keep
    price_no_outliers = remove_outliers_zscore(new_df_encoded['price_sgd'])
    stay_length_no_outliers = remove_outliers_zscore(new_df_encoded['stay_length'])
        
    price_removed = 1 - len(price_no_outliers) / new_df_encoded['price_sgd'].notna().sum()
    stay_removed = 1 - len(stay_length_no_outliers) / new_df_encoded['stay_length'].notna().sum()


    # creates new columns for price_sgd_clean and stay_length_clean (outliers removed)
    new_df_encoded['price_sgd_clean'] = new_df_encoded['price_sgd']
    new_df_encoded.loc[new_df_encoded['price_sgd'].notna(), 'price_sgd_clean'] = price_no_outliers

    new_df_encoded['stay_length_clean'] = new_df_encoded['stay_length']
    new_df_encoded.loc[new_df_encoded['stay_length'].notna(), 'stay_length_clean'] = stay_length_no_outliers

    new_df_encoded = new_df_encoded.drop(['price_sgd', 'stay_length'], axis=1)


    # dealing with more NaN errors
    columns_with_nan = new_df_encoded.columns[new_df_encoded.isnull().any()].tolist()
    columns_with_nan = [col for col in columns_with_nan if col != 'booking_id']

    for column in columns_with_nan:
        if new_df_encoded[column].dtype in ['int64', 'float64', 'Int64']:  # For numeric columns
            median_value = new_df_encoded[column].median()
            new_df_encoded[column] = new_df_encoded[column].fillna(median_value)
        elif pd.api.types.is_object_dtype(new_df_encoded[column]):  # For categorical columns
            mode_value = new_df_encoded[column].mode()[0]
            new_df_encoded[column] = new_df_encoded[column].fillna(mode_value)

    new_df_encoded = new_df_encoded.drop(['booking_id', 'arrival_day', 'checkout_day'], axis=1)

    median_arrival = new_df_encoded['arrival_date'].median()
    median_checkout = new_df_encoded['checkout_date'].median()

    new_df_encoded['arrival_date'] = new_df_encoded['arrival_date'].fillna(median_arrival)
    new_df_encoded['checkout_date'] = new_df_encoded['checkout_date'].fillna(median_checkout)

    pass



# 7) SCALING
def scale_features(df):
    scaler = StandardScaler()
    columns_to_scale = ['num_adults', 'num_children', 'booking_month_num', 'arrival_month_num', 'month_difference', 'price_sgd_clean', 'stay_length_clean']
    scaled_columns = [f"{col}_scaled" for col in columns_to_scale]
    new_df_encoded[scaled_columns] = scaler.fit_transform(new_df_encoded[columns_to_scale])

    pass



# 8) CORRELATION ANALYSIS

def select_features(df):
    feature_columns = [col for col in new_df_encoded.columns if col != 'no_show']
    correlations_with_no_show = new_df_encoded[feature_columns].corrwith(new_df_encoded['no_show'])
    correlations_sorted = correlations_with_no_show.abs().sort_values(ascending=False)

    high_corr_features = correlations_sorted[abs(correlations_sorted) >= 0.05]
    columns_to_keep = high_corr_features.index.tolist()

    if 'no_show' not in columns_to_keep:
        columns_to_keep.append('no_show')

    final_df = new_df_encoded[columns_to_keep]
    
    

def main():
    
    df = get_noshow_data()
    final_df = preprocess_data(df)
    return final_df











































































