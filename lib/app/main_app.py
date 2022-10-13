import streamlit as st
import pandas as pd

def app():
    st.title("Hello")
    file_png = st.file_uploader("Upload a PNG image", type=([".csv"]))

    if file_png:
        #file_png_bytes = st.file_reader(file_png)
        #st.image(file_png_bytes)
        st.write(pd.read_csv(file_png).head())
        data2 = pd.read_csv("lib/data/blbl.csv")
        st.write(data2.head())
        nb = st.text_input("enternb")
        d = {'col1': [nb, nb], 'col2': [3, 4]}
        df = pd.DataFrame(data=d)
        df.to_csv("lib/data/blbl.csv")
        st.download_button('Download CSV', "blblblbl")