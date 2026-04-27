import streamlit as st
import pickle
import pandas as pd
import numpy as np


infile=open("D:/Machine_Learning/Cases/Wisconsin/bestModel_breastCancer.pkl",'rb')

objload=pickle.load(infile)
infile.close()

st.title("BreastCancer Prediction")

col1,col2=st.columns([0.3,0.3])

with col1:
    Code = st.number_input(label="Code :",step=1)
    Clump=st.number_input(label="Clump thickness :",min_value=1,max_value=10,step=1)
    UniCell_Size = st.number_input(label="Uniformity of Cell Size :",min_value=1,max_value=10,step=1)
    Uni_CellShape = st.number_input(label="Uniformity of Cell Shape :",min_value=1,max_value=10,step=1)
    MargAdh = st.number_input(label="Marginal Adhesion :",min_value=1,max_value=10,step=1)
with col2:
    SEpith = st.number_input(label="Single Epithelial Cell Size :",min_value=1,max_value=10,step=1)
    BareN = st.number_input(label="Bare Nuclei:",min_value=1,max_value=10,step=1)
    BChromatin = st.number_input(label="Bland Chromatin :",min_value=1,max_value=10,step=1)
    NoemN = st.number_input(label="Normal Nucleoli:",min_value=1,max_value=10,step=1)
    Mitoses = st.number_input(label="Mitoses:",min_value=1,max_value=10,step=1)

df=pd.DataFrame({'Code':[Code],'Clump':[Clump],'UniCell_Size':[UniCell_Size],'Uni_CellShape':[Uni_CellShape],'MargAdh':[MargAdh],
                 'SEpith':[SEpith],'BareN':[BareN],'BChromatin':[BChromatin],'NoemN':[NoemN],'Mitoses':[Mitoses]
                 })

pred=objload.predict(df)[0]
st.write("Predicted Class:")
st.title("{}".format(pred))