## Importing libraries
import streamlit as st
import pandas as pd
from deta import Deta
import numpy as np
from math import floor, ceil
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import os

def app():
        """
        Main function that is called when running the app
        """
        st.title("Portfolio & Allocations Visualization")
        #Key to read/write in Deta base
        key="a0ujshlg_M37v7zCfNRvrPj9RiaD7ykTESJFdEbm4"
        #Establishing connection with Deta
        deta = Deta(key)
        date_selected = st.date_input("Date for which to retrieve portfolio")
        st.subheader("--Fetch portfolio--")
        deta_db_portfolio= deta.Base("portfolio_"+str(date_selected))
        deta_db_portfolio_list = deta_db_portfolio.fetch().items
        if len(deta_db_portfolio_list)!=0:
                deta_db_portfolio_df = pd.DataFrame(deta_db_portfolio_list)
                st.write("Found agents already defined in database for the selected date")
                st.write(deta_db_portfolio_df.set_index("Code ISIN"))

        st.subheader("--Fetch constraints--")
        deta_db_constraints = deta.Base("constraints_"+str(date_selected))
        deta_db_constraints_list = deta_db_constraints.fetch().items
        if len(deta_db_constraints_list)!=0:
                deta_db_constraints_df = pd.DataFrame(deta_db_constraints_list)
                st.write("Found agents already defined in database for the selected date")
                st.write(deta_db_constraints_df.set_index("counterparty"))
        counterparty_1_allocation_df = deta_db_portfolio_df[deta_db_portfolio_df["allocation qtt1"]!=0][["Code ISIN","allocation qtt1"]]
        counterparty_1_allocation_df["Counterparty"]="Counterparty 1"
        counterparty_1_allocation_df.rename(columns={"allocation qtt1":"Quantity Allocated"},inplace=True)
        counterparty_2_allocation_df = deta_db_portfolio_df[deta_db_portfolio_df["allocation qtt2"]!=0][["Code ISIN","allocation qtt2"]]
        counterparty_2_allocation_df["Counterparty"]="Counterparty 2"
        counterparty_2_allocation_df.rename(columns={"allocation qtt2":"Quantity Allocated"},inplace=True)
        counterparty_3_allocation_df = deta_db_portfolio_df[deta_db_portfolio_df["allocation qtt3"]!=0][["Code ISIN","allocation qtt3"]]
        counterparty_3_allocation_df["Counterparty"]="Counterparty 3"
        counterparty_3_allocation_df.rename(columns={"allocation qtt3":"Quantity Allocated"},inplace=True)
        counterparties_allocation_df = pd.concat([counterparty_1_allocation_df,counterparty_2_allocation_df,counterparty_3_allocation_df])
        st.download_button('Download Portfolio (CSV)',convert_df(deta_db_portfolio_df), "portfolio_"+str(date_selected)+".csv",'text/csv')
        st.download_button('Download Constraints (CSV)',convert_df(deta_db_constraints_df), "constraints_"+str(date_selected)+".csv",'text/csv')
        st.download_button('Download Counterpaties Allocations (CSV)',convert_df(counterparties_allocation_df), "counterparties_allocation_"+str(date_selected)+".csv",'text/csv')


@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')