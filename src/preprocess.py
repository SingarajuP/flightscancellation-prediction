import pandas as pd
import pickle

def convert_time(df):
    
    df.fl_date = pd.to_datetime(df.fl_date, format='%Y-%m-%d')
    df['month'] = pd.DatetimeIndex(df.fl_date).month
    df['day_of_year'] = df.fl_date.dt.dayofyear
    df['day_of_week'] = df.fl_date.dt.dayofweek
    df['crs_dep_time'] = df['crs_dep_time'].apply(lambda x: '{0:0>4}'.format(x))
    df['dep_hour'] = df['crs_dep_time'].str[:2]

    return df

def bins(df):
    hours = {'00':0,'01':0,'02':0,'03':0,'04':0,'05':0,'06':1,'07':1,'08':1,'09':1,'10':1,'11':2,'12':2,'13':2,'14':2,'15':2,'16':2,'17':3,'18':3,'19':3,'20':4,'21':4,'22':4,'23':4,'24': 4}
    df['dep_part_day'] = df['dep_hour'].map(hours)
    season = {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4, 12: 1}
    df['season'] = df['month'].map(season)
    
    return df

def carrier_map(df):
    dic={"Endeavor Air Inc":"9E","Cape Air":"9K" ,"American Airlines Inc":"AA","Alaska Airlines Inc":"AS","Trans States Airlines":"AX","JetBlue Airways":"B6","Commutair Aka Champlain Enterprises":"C5","Compass Airlines":"CP", "Delta Air Lines Inc":"DL","Empire Airlines Inc":"EM","ExpressJet Airlines Inc":"EV","Frontier Airlines Inc":"F9","Allegiant Air":"G4", "GoJet Airlines LLC d/b/a United Express":"G7","Hawaiian Airlines Inc":"HA","Peninsula Airways Inc":"KS", "Envoy Air":"MQ","Spirit Air Lines":"NK","PSA Airlines Inc":"OH","SkyWest Airlines Inc":"OO","Piedmont Airlines":"PT", "Horizon Air":"QX","United Air Lines Inc":"UA","Virgin America":"VX","Southwest Airlines Co":"WN","Mesa Airlines Inc":"YV","Republic Airways":"YX", "Air Wisconsin Airlines Corp":"ZW"}
    df=df.replace({"op_unique_carrier": dic})
    print("Carrier mapping:\n",df)
    return df
def get_dummies(df):
    one_hot_encoder = pickle.load(open("./models/one_hot_encoder_allcancel.pkl", "rb"))
    categorical_vars = ["op_unique_carrier", "origin", "dest"]
    df_vars_array = one_hot_encoder.transform(df[categorical_vars])
    df_vars = pd.DataFrame(df_vars_array)
    df = pd.concat([df.reset_index(drop=True), df_vars.reset_index(drop=True)], axis = 1)
    df.drop(categorical_vars, axis = 1, inplace = True)
    print("Dummies:",df)
    return df
def dtypes(df):
    df.columns=df.columns.astype(str)
    df=df.astype('int')
    return df