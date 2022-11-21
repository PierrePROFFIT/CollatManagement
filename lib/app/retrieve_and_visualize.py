## Importing libraries
import streamlit as st
import pandas as pd
from deta import Deta
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        #Fetch portfolio from deta database
        deta_db_portfolio= deta.Base("portfolio_"+str(date_selected))
        deta_db_portfolio_list = deta_db_portfolio.fetch().items
        if len(deta_db_portfolio_list)!=0:
                deta_db_portfolio_df = pd.DataFrame(deta_db_portfolio_list)
                st.write("Found agents already defined in database for the selected date")
                st.write(deta_db_portfolio_df.set_index("Code ISIN"))

        st.subheader("--Fetch constraints--")
        #Fetch constraints from deta database
        deta_db_constraints = deta.Base("constraints_"+str(date_selected))
        deta_db_constraints_list = deta_db_constraints.fetch().items
        if len(deta_db_constraints_list)!=0:
                deta_db_constraints_df = pd.DataFrame(deta_db_constraints_list)
                st.write("Found agents already defined in database for the selected date")
                st.write(deta_db_constraints_df.set_index("counterparty"))
        
        #Construct counterparties_allocation_df
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

        #Display download button
        st.download_button('Download Portfolio (CSV)',convert_df(deta_db_portfolio_df), "portfolio_"+str(date_selected)+".csv",'text/csv')
        st.download_button('Download Constraints (CSV)',convert_df(deta_db_constraints_df), "constraints_"+str(date_selected)+".csv",'text/csv')
        st.download_button('Download Counterpaties Allocations (CSV)',convert_df(counterparties_allocation_df), "counterparties_allocation_"+str(date_selected)+".csv",'text/csv')

        st.subheader("--Output Visualization--")
        #Display charts
        visualize(deta_db_portfolio_df,"Type","pie")
        visualize(deta_db_portfolio_df,"Currency","pie")
        visualize(deta_db_portfolio_df,"Ratings Agency Worst of 2","bar")
        visualize(deta_db_portfolio_df,"Issuer Sector","bar")
        visualize(deta_db_portfolio_df,"Issuer Country","bar")
        visualize(deta_db_portfolio_df,"Code ISIN","bar")
        

def visualize(data,field,plot_type="pie"):
    """
    Function that enables to display pie or bar charts
    """
    if plot_type=="pie":
        fig = make_subplots(rows=1, cols=3,specs=[[{"type": "pie"}, {"type": "pie"},{"type": "pie"}]],subplot_titles=("Counterparty 1", "Counterparty 2","Counterparty 3"))
        for i in [1,2,3]:
                data2 = data.groupby(field)[["allocation "+str(i)]].sum()
                fig.add_trace(go.Pie(labels=data2.index, values = data2["allocation "+str(i)]),row=1,col=i)
        fig.update_layout(title="Allocation by "+field)
        st.plotly_chart(fig)
    elif plot_type=="bar":
        fig = make_subplots(rows=1, cols=3,specs=[[{"type": "bar"}, {"type": "bar"},{"type": "bar"}]],subplot_titles=("Counterparty 1", "Counterparty 2","Counterparty 3"))
        for i in [1,2,3]:
                data2 = data.groupby(field)[["allocation "+str(i)]].sum()
                fig.add_trace(go.Bar(x=data2.index, y = data2["allocation "+str(i)]),row=1,col=i)
        fig.update_layout(title="Allocation by "+field)
        st.plotly_chart(fig)

@st.cache
def convert_df(df):
    """
    Function that converts df to csv
    """
    return df.to_csv().encode('utf-8')