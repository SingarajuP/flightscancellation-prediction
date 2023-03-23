import streamlit as st
import pickle
import pandas as pd
import sys
sys.path.append("src/")
from src.preprocess import bins,carrier_map,convert_time,dtypes,get_dummies

from src.predict import classify
model = pickle.load(open("./models/xgb_app_clf_sampling_allcancel.pkl", "rb"))

def inputs():

    st.title("Flight cancellation prediction")

    date=st.date_input("Enter the date of travel")
    time=st.number_input('Enter the departure time: Ex: 1234 for 12:34,730 for 7:30,1445 for 2.45pm ',value=0000)
    carrier=st.selectbox('Carrier',["Endeavor Air Inc","Cape Air","American Airlines Inc","Alaska Airlines Inc","Trans States Airlines","JetBlue Airways","Commutair Aka Champlain Enterprises","Compass Airlines",
    "Delta Air Lines Inc","Empire Airlines Inc","ExpressJet Airlines Inc","Frontier Airlines Inc","Allegiant Air",
    "GoJet Airlines LLC d/b/a United Express","Hawaiian Airlines Inc","Peninsula Airways Inc",
    "Envoy Air","Spirit Air Lines","PSA Airlines Inc","SkyWest Airlines Inc","Piedmont Airlines",
    "Horizon Air","United Air Lines Inc","Virgin America","Southwest Airlines Co","Mesa Airlines Inc","Republic Airways",
    "Air Wisconsin Airlines Corp"])
    origin=st.selectbox('Origin airport',['ABE', 'ABI', 'ABQ', 'ABR', 'ABY', 'ACK', 'ACT', 'ACV', 'ACY',
       'ADK', 'ADQ', 'AEX', 'AGS', 'AKN', 'ALB', 'ALO', 'ALW', 'AMA',
       'ANC', 'APN', 'ART', 'ASE', 'ATL', 'ATW', 'ATY', 'AUS', 'AVL',
       'AVP', 'AZA', 'AZO', 'BDL', 'BET', 'BFF', 'BFL', 'BFM', 'BGM',
       'BGR', 'BHM', 'BIL', 'BIS', 'BJI', 'BKG', 'BLI', 'BLV', 'BMI',
       'BNA', 'BOI', 'BOS', 'BPT', 'BQK', 'BQN', 'BRD', 'BRO', 'BRW',
       'BTM', 'BTR', 'BTV', 'BUF', 'BUR', 'BWI', 'BZN', 'CAE', 'CAK',
       'CDC', 'CDV', 'CGI', 'CHA', 'CHO', 'CHS', 'CID', 'CIU', 'CKB',
       'CLE', 'CLL', 'CLT', 'CMH', 'CMI', 'CMX', 'CNY', 'COD', 'COS',
       'COU', 'CPR', 'CRP', 'CRW', 'CSG', 'CVG', 'CWA', 'CYS', 'DAB',
       'DAL', 'DAY', 'DBQ', 'DCA', 'DEN', 'DFW', 'DHN', 'DIK', 'DLG',
       'DLH', 'DRO', 'DRT', 'DSM', 'DTW', 'DUT', 'DVL', 'EAR', 'EAT',
       'EAU', 'ECP', 'EGE', 'EKO', 'ELM', 'ELP', 'ERI', 'ESC', 'EUG',
       'EVV', 'EWN', 'EWR', 'EYW', 'FAI', 'FAR', 'FAT', 'FAY', 'FCA',
       'FLG', 'FLL', 'FLO', 'FNT', 'FSD', 'FSM', 'FWA', 'GCC', 'GCK',
       'GEG', 'GFK', 'GGG', 'GJT', 'GNV', 'GPT', 'GRB', 'GRI', 'GRK',
       'GRR', 'GSO', 'GSP', 'GST', 'GTF', 'GTR', 'GUC', 'GUM', 'HDN',
       'HGR', 'HHH', 'HIB', 'HLN', 'HNL', 'HOB', 'HOU', 'HPN', 'HRL',
       'HSV', 'HTS', 'HVN', 'HYA', 'HYS', 'IAD', 'IAG', 'IAH', 'ICT',
       'IDA', 'ILM', 'IMT', 'IND', 'INL', 'IPT', 'ISN', 'ISP', 'ITH',
       'ITO', 'JAC', 'JAN', 'JAX', 'JFK', 'JHM', 'JLN', 'JMS', 'JNU',
       'KOA', 'KTN', 'LAN', 'LAR', 'LAS', 'LAW', 'LAX', 'LBB', 'LBE',
       'LBF', 'LBL', 'LCH', 'LCK', 'LEX', 'LFT', 'LGA', 'LGB', 'LIH',
       'LIT', 'LNK', 'LNY', 'LRD', 'LSE', 'LWB', 'LWS', 'LYH', 'MAF',
       'MBS', 'MCI', 'MCO', 'MDT', 'MDW', 'MEI', 'MEM', 'MFE', 'MFR',
       'MGM', 'MHK', 'MHT', 'MIA', 'MKE', 'MKG', 'MKK', 'MLB', 'MLI',
       'MLU', 'MMH', 'MOB', 'MOT', 'MQT', 'MRY', 'MSN', 'MSO', 'MSP',
       'MSY', 'MTJ', 'MVY', 'MYR', 'OAJ', 'OAK', 'OGD', 'OGG', 'OGS',
       'OKC', 'OMA', 'OME', 'ONT', 'ORD', 'ORF', 'ORH', 'OTH', 'OTZ',
       'OWB', 'PAE', 'PAH', 'PBG', 'PBI', 'PDX', 'PGD', 'PGV', 'PHF',
       'PHL', 'PHX', 'PIA', 'PIB', 'PIE', 'PIH', 'PIR', 'PIT', 'PLN',
       'PNS', 'PPG', 'PQI', 'PRC', 'PSC', 'PSE', 'PSG', 'PSM', 'PSP',
       'PUB', 'PUW', 'PVD', 'PVU', 'PWM', 'RAP', 'RDD', 'RDM', 'RDU',
       'RFD', 'RHI', 'RIC', 'RKS', 'RNO', 'ROA', 'ROC', 'ROP', 'ROW',
       'RST', 'RSW', 'SAF', 'SAN', 'SAT', 'SAV', 'SBA', 'SBN', 'SBP',
       'SBY', 'SCC', 'SCE', 'SCK', 'SDF', 'SEA', 'SFB', 'SFO', 'SGF',
       'SGU', 'SHD', 'SHV', 'SIT', 'SJC', 'SJT', 'SJU', 'SLC', 'SLN',
       'SMF', 'SMX', 'SNA', 'SPI', 'SPN', 'SPS', 'SRQ', 'STC', 'STL',
       'STS', 'STT', 'STX', 'SUN', 'SUX', 'SWF', 'SWO', 'SYR', 'TLH',
       'TOL', 'TPA', 'TRI', 'TTN', 'TUL', 'TUS', 'TVC', 'TWF', 'TXK',
       'TYR', 'TYS', 'UIN', 'USA', 'VEL', 'VLD', 'VPS', 'WRG', 'WYS',
       'XNA', 'XWA', 'YAK', 'YKM', 'YUM'])
    dest=st.selectbox('Destination airport',['ABE', 'ABI', 'ABQ', 'ABR', 'ABY', 'ACK', 'ACT', 'ACV', 'ACY',
       'ADK', 'ADQ', 'AEX', 'AGS', 'AKN', 'ALB', 'ALO', 'ALW', 'AMA',
       'ANC', 'APN', 'ART', 'ASE', 'ATL', 'ATW', 'ATY', 'AUS', 'AVL',
       'AVP', 'AZA', 'AZO', 'BDL', 'BET', 'BFF', 'BFL', 'BFM', 'BGM',
       'BGR', 'BHM', 'BIL', 'BIS', 'BJI', 'BKG', 'BLI', 'BLV', 'BMI',
       'BNA', 'BOI', 'BOS', 'BPT', 'BQK', 'BQN', 'BRD', 'BRO', 'BRW',
       'BTM', 'BTR', 'BTV', 'BUF', 'BUR', 'BWI', 'BZN', 'CAE', 'CAK',
       'CDC', 'CDV', 'CGI', 'CHA', 'CHO', 'CHS', 'CID', 'CIU', 'CKB',
       'CLE', 'CLL', 'CLT', 'CMH', 'CMI', 'CMX', 'CNY', 'COD', 'COS',
       'COU', 'CPR', 'CRP', 'CRW', 'CSG', 'CVG', 'CWA', 'CYS', 'DAB',
       'DAL', 'DAY', 'DBQ', 'DCA', 'DEN', 'DFW', 'DHN', 'DIK', 'DLG',
       'DLH', 'DRO', 'DRT', 'DSM', 'DTW', 'DUT', 'DVL', 'EAR', 'EAT',
       'EAU', 'ECP', 'EGE', 'EKO', 'ELM', 'ELP', 'ERI', 'ESC', 'EUG',
       'EVV', 'EWN', 'EWR', 'EYW', 'FAI', 'FAR', 'FAT', 'FAY', 'FCA',
       'FLG', 'FLL', 'FLO', 'FNT', 'FSD', 'FSM', 'FWA', 'GCC', 'GCK',
       'GEG', 'GFK', 'GGG', 'GJT', 'GNV', 'GPT', 'GRB', 'GRI', 'GRK',
       'GRR', 'GSO', 'GSP', 'GST', 'GTF', 'GTR', 'GUC', 'GUM', 'HDN',
       'HGR', 'HHH', 'HIB', 'HLN', 'HNL', 'HOB', 'HOU', 'HPN', 'HRL',
       'HSV', 'HTS', 'HVN', 'HYA', 'HYS', 'IAD', 'IAG', 'IAH', 'ICT',
       'IDA', 'ILM', 'IMT', 'IND', 'INL', 'IPT', 'ISN', 'ISP', 'ITH',
       'ITO', 'JAC', 'JAN', 'JAX', 'JFK', 'JHM', 'JLN', 'JMS', 'JNU',
       'KOA', 'KTN', 'LAN', 'LAR', 'LAS', 'LAW', 'LAX', 'LBB', 'LBE',
       'LBF', 'LBL', 'LCH', 'LCK', 'LEX', 'LFT', 'LGA', 'LGB', 'LIH',
       'LIT', 'LNK', 'LNY', 'LRD', 'LSE', 'LWB', 'LWS', 'LYH', 'MAF',
       'MBS', 'MCI', 'MCO', 'MDT', 'MDW', 'MEI', 'MEM', 'MFE', 'MFR',
       'MGM', 'MHK', 'MHT', 'MIA', 'MKE', 'MKG', 'MKK', 'MLB', 'MLI',
       'MLU', 'MMH', 'MOB', 'MOT', 'MQT', 'MRY', 'MSN', 'MSO', 'MSP',
       'MSY', 'MTJ', 'MVY', 'MYR', 'OAJ', 'OAK', 'OGD', 'OGG', 'OGS',
       'OKC', 'OMA', 'OME', 'ONT', 'ORD', 'ORF', 'ORH', 'OTH', 'OTZ',
       'OWB', 'PAE', 'PAH', 'PBG', 'PBI', 'PDX', 'PGD', 'PGV', 'PHF',
       'PHL', 'PHX', 'PIA', 'PIB', 'PIE', 'PIH', 'PIR', 'PIT', 'PLN',
       'PNS', 'PPG', 'PQI', 'PRC', 'PSC', 'PSE', 'PSG', 'PSM', 'PSP',
       'PUB', 'PUW', 'PVD', 'PVU', 'PWM', 'RAP', 'RDD', 'RDM', 'RDU',
       'RFD', 'RHI', 'RIC', 'RKS', 'RNO', 'ROA', 'ROC', 'ROP', 'ROW',
       'RST', 'RSW', 'SAF', 'SAN', 'SAT', 'SAV', 'SBA', 'SBN', 'SBP',
       'SBY', 'SCC', 'SCE', 'SCK', 'SDF', 'SEA', 'SFB', 'SFO', 'SGF',
       'SGU', 'SHD', 'SHV', 'SIT', 'SJC', 'SJT', 'SJU', 'SLC', 'SLN',
       'SMF', 'SMX', 'SNA', 'SPI', 'SPN', 'SPS', 'SRQ', 'STC', 'STL',
       'STS', 'STT', 'STX', 'SUN', 'SUX', 'SWF', 'SWO', 'SYR', 'TLH',
       'TOL', 'TPA', 'TRI', 'TTN', 'TUL', 'TUS', 'TVC', 'TWF', 'TXK',
       'TYR', 'TYS', 'UIN', 'USA', 'VEL', 'VLD', 'VPS', 'WRG', 'WYS',
       'XNA', 'XWA', 'YAK', 'YKM', 'YUM'])
    
    if st.button("Submit"):
        df=pd.DataFrame({'fl_date':[date],'crs_dep_time':[time],'op_unique_carrier':[carrier],'origin':[origin],'dest':[dest]})
        print("First dataframe:",df)
        df=convert_time(df)
        df=bins(df)
        print("After time and bins :",df)
        df=carrier_map(df)
        df=get_dummies(df)
        df=dtypes(df)
        df=df.drop(['fl_date','crs_dep_time'],axis=1)
        print("Before goinng to the model:",df)
        pred,pred_prob=classify(model,df)
        st.write("Prediction for your flight is : {} with a probability of {}".format(pred,pred_prob))

inputs()