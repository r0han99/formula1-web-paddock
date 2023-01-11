import streamlit as st 
import pandas as pd
import ast 
from itertools import cycle



df = pd.read_csv('FINAL-CAR-IMAGES_2012-2022.csv')
st.set_page_config(layout='wide')
for year in range(2012, 2023):
    st.title(year)
    cols = st.columns(3)
    idx = cycle([0,1,2])
    car, link = df[df['Unnamed: 0']==year]['Car'], df[df['Unnamed: 0']==year]['Image-Link']
    for c, l in zip(car, link):
        cols[next(idx)].image(l, caption=c)
        
        

