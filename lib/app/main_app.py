import streamlit as st
import pandas as pd
from deta import Deta

#streamlit run c:/Users/pierr/OneDrive/Documents/CollatManagement/main.py
def app():
    #Key to read/write in Deta base
    key="a0ujshlg_M37v7zCfNRvrPj9RiaD7ykTESJFdEbm4"
    #Establishing connection
    deta = Deta(key)

    #db = deta.Base("example-db")
    #st.write(db.fetch().items)
    #db.put({"name": "Pierre", "age": "24"})
    #st.write(db.fetch().items)
    date = st.date_input("Date of portfolio to upload")
    filename_input_portfolio = st.file_uploader("Upload Portfolio CSV or XLSX file", type=([".csv",".xlsx"]))
    if filename_input_portfolio:
        if "csv" in filename_input_portfolio.name:
            data_input_portfolio = pd.read_csv(filename_input_portfolio,sep=";").fillna("None")
        elif "xlsx" in filename_input_portfolio.name:
            data_input_portfolio = pd.read_excel(filename_input_portfolio,engine="openpyxl").fillna("None")
        st.write("Top rows of uploaded data")
        st.write(data_input_portfolio.head())

        deta_db_input_portfolio = deta.Base("portfolio_"+str(date))
        df_input_portfolio = data_input_portfolio.to_dict('index')
        for k in df_input_portfolio.keys():
            if "Code ISIN" not in df_input_portfolio[k].keys():
                st.write("Make sure there's a column Code ISIN in your file as this column will be used as key")
            else:
                deta_db_input_portfolio.put(df_input_portfolio[k],df_input_portfolio[k]["Code ISIN"])
        #deta_db_input_portfolio.put_many(list(df_input_portfolio.values()))
        #data2 = pd.read_csv("lib/data/blbl.csv")
        #st.write(data2.head())
        #nb = st.text_input("enternb")
        #d = {'col1': [nb, nb], 'col2': [3, 4]}
        #df = pd.DataFrame(data=d)
        #df.to_csv("lib/data/blbl.csv")
        
        #st.download_button('Download CSV', "blblblbl")