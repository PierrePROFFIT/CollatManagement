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
from lib.app import collateral_allocater
from lib.app import retrieve_and_visualize

#TO RUN THE CODE : 
#Step 1: Open cmd prompt
#Step 2: Copy path to CollatManagement/main.py
#Step 3: Run the following command in cmd prompt : streamlit run #path_to_main#/main.py
#add archive viz
#adtv ok
#Check column name
#exposure close 300 298
#download isin
#eur mkt cap ok

st.set_page_config(layout="wide")
pages = {
        "Collateral Allocation":collateral_allocater,
        "Retrieve, Visualize & Download":retrieve_and_visualize
        }
st.sidebar.title("Collateral Manager")
select_page = st.sidebar.radio("GO TO : ", list(pages.keys()))
page = pages[select_page]
page.app()