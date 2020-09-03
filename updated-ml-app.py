import pandas as pd
import numpy as np
import streamlit as st
import pickle
import xgboost


st.title('House Price Prediction ')

st.write('This app predicts the price of house based on different parameters.')

from PIL import Image
image = Image.open('home.png')
st.image(image)

st.write('''
    shelter is the one of the fundamental need of human being,
    in today's world house plays the role of peramnent shelter.
    so humans are needed to give very careful attention when purchasing the house.
    to solve this problem, this app helps you to find the best estimated price of house 
    based on several different requirements.

    following the is description of terms used in this prediction app.
    user need to give short word input in respective field. 

    * OverallQual: Rates the overall material and finish of the house.

    * GrLivArea: Above grade (ground) living area square feet.

    * GarageCars: Size of garage in car capacity.

    * ExterQual: Evaluates the quality of the material on the exterior.
    (Ex = Excellent, Gd = Good,TA = Average/Typical, Fa = Fair,Po = Poor) 

    * KitchenQual: Kitchen quality.
    (Ex = Excellent,Gd = Good,TA = Typical/Average,Fa= Fair,Po = Poor)

    * GarageArea: Size of garage in square feet.

    * BsmtQual: Evaluates the height of the basement.
    (  Ex =   Excellent (100+ inches), 
       Gd =  Good (90-99 inches)
       TA =  Typical (80-89 inches)
       Fa =  Fair (70-79 inches)
       Po =  Poor (<70 inches
       NA =  No Basement)

    * TotalBsmtSF: Total square feet of basement area.

    * GarageFinish: Interior finish of the garage.
    (Fin=Finished,RFn=Rough Finished, Unf = Unfinished ,NA=No Garage)

     ''')


st.header('User Input Parameters')
def user_input_features():
    OverallQual=st.number_input('OverallQual'),
    GrLivArea=st.number_input('GrLivArea'),
    GarageCars=st.number_input('GarageCars'),
    ExterQual=st.text_input('ExterQual'),
    KitchenQual=st.text_input('KitchenQual'),
    GarageArea=st.number_input('GarageArea'),
    BsmtQual=st.text_input('BsmtQual'),
    TotalBsmtSF=st.number_input('TotalBsmtSF'),
    GarageFinish=st.text_input('GarageFinish')
        
    
    data={'OverallQual': OverallQual,
         'GrLivArea':GrLivArea,
         'GarageCars':GarageCars,
         'ExterQual':ExterQual,
         'KitchenQual':KitchenQual,
         'GarageArea':GarageArea,
         'BsmtQual':BsmtQual,
         'TotalBsmtSF':TotalBsmtSF,
         'GarageFinish':GarageFinish }
    features=pd.DataFrame(data,index=[0])
    return features
    
df= user_input_features()

st.subheader('User Input Parameters')

st.write(df)

transform={
    'ExterQual':{'Ex':1,'Fa':2,'Gd':3,'TA':4},
    'KitchenQual':{'Ex':1,'Fa':2,'Gd':3,'TA':4},
    'BsmtQual':{'Ex':1,'Fa':2,'Gd':3,'Na':4,'TA':5},
    'GarageFinish':{'Fin':1,'Na':2,'RFn':3,'Unf':4}
}

for col in transform:
    df.loc[:, col] = df.loc[:, col].map(transform[col], na_action='ignore')


model=pickle.load(open('house-final.pkl','rb'))       

predictions=model.predict(df)
kl=''

if st.button('Predict'):
    kl=f'Price of the house is{predictions}'
st.success(kl)

if st.button("About"):
    st.text('Developed by Amey Girdhari')
    st.text('Built with streamlit')