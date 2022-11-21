# Importing libraries
import streamlit as st

#Importing package modules
from lib.app import collateral_allocater
from lib.app import retrieve_and_visualize

#TO RUN THE CODE : 
#Step 1: Open cmd prompt
#Step 2: Copy path to CollatManagement/main.py
#Step 3: Run the following command in cmd prompt : streamlit run #path_to_main#/main.py

st.set_page_config(layout="wide")
pages = {
        "Collateral Allocation":collateral_allocater,
        "Retrieve, Visualize & Download":retrieve_and_visualize
        }
st.sidebar.title("Collateral Manager")
select_page = st.sidebar.radio("GO TO : ", list(pages.keys()))
page = pages[select_page]
page.app()