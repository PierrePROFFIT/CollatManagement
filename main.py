## Importing app (ie pages)
from lib.app import main_app

## Importing libraries
import streamlit as st

#st.set_page_config(layout="wide")
pages = {
        "Import Portfolio in Cloud Database":main_app
        }
st.sidebar.title("Collateral Manager")
select_page = st.sidebar.radio("GO TO : ", list(pages.keys()))
page = pages[select_page]
page.app()