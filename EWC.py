import numpy as np
import pickle
import pandas as pd
import streamlit as st
#collect the model and min-max object file
filename ='EWC_1.pkl' 
scaler= 'minmax_scaler.pkl'
target= 'minmax_target.pkl'

def load_model():
    return pickle.load(open(filename,'rb'))


def load_scaler():
    return pickle.load(open(scaler,'rb'))

def target_scaler():
    return pickle.load(open(target,'rb'))

model = load_model()
scaler = load_scaler()
target= target_scaler()

def welcome():
    return "Welcome All"

def inverse_transform(y_pred):
  pred=target.inverse_transform([y_pred])
  return pred

def prediction_LOGEC3(X, scaler, model):
  #Scale the input
  scaled_input = scaler.transform(X)
  prediction = model.predict(scaled_input)
  return prediction


def convert_to_log10(input):
  log=np.log10(input)
  return log


def main():
    st.title(" EDELWEISS CONNECT ITS SKIN SENSITIZATION SOLUTION ")
    st.markdown('An Artificial Neural Network Regression model Utilizing invitro and inchemo(h-CLAT,DPRA) assay Descriptors for predicting skin Sensitization EC3 values')
    html_temp = """
    EWC-1 SKIN SENSITIZATION PREDICTION APP
    """
    if st.button('INFORMATION ABOUT THIS WEB APP BEFORE USE'):
          st.write("The Edelweiss ITS skin sensitization model predicts the Murine local lymph node assay (LLNA) EC3 value of a substance. The model uses Adverse Outcome Pathway data obtained from in-chemo and invitro assays to reflect the underlying immune response that leads to skin sensitization.")
          st.header('TRAIN DATA')
          st.write('The EWC-1 model is trained using the direct peptide reactivity assay (DPRA) and  Human cell line activation assay data(h-CLAT) data. The DPRA assay is an in-chemo assay that measures a substances ability to form a hapten-protein complex while the human Cell line activation assay is an in-vitro test that accesses the ability of a substance to induce or mobilize dendric cells in the skin. Results from DPRA and h-CLAT reflect the molecular initiating event in AOP for skin sensitization the first key event in Skin Sensitization Adverse outcome pathway.Both tests are part of the integrated testing strategies for skin sensitization as stipulated by OECD.')
       
          st.header('WEB APP INPUT')
          st.write('The Web app utilizes input data for DPRA and h-CLAT assay to aid robust and précised prediction capabilities. the model accepts the following input parameter. The model accepts this input,Transforms them and automatically select the right input used for making prediction.')
          st.write('DPRA – Average of DPRA LysD AND DPRA CysD')
          st.write('h-CLAT - Minimum of CD86-EC150, CD54-EC200, and CV75')
          
          st.header('LINK TO REFERENCE PAPERS')
          st.write('Original model by Hirota et al.,2015 [Orignal model](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/jat.3105)')
          st.write('Reference model by Kleinstreuer et al.,2018 [Reference Model](https://www.tandfonline.com/loi/itxc20)')
          st.write('OECD guildline for LLNA assay [Guildline for LLNA assay](https://www.oecd.org/env/test-no-429-skin-sensitisation-9789264071100-en.htm)')
          st.write('OECD guildline for in-vitro assay [Guildline for in-invitro assay](https://www.oecd.org/env/guidance-document-on-good-in-vitro-method-practices-givimp-9789264304796-en.htm)')
          st.header('EXTRA INFORMATION')
          st.write('in certain condition where CD86_EC150, CD54_EC200 any any other input isnt avaliable CV75 should be used while the other unavaliable  inputs are set at 0.1') 
    st.markdown(html_temp, unsafe_allow_html=True)
    DPRA_LysD = st.number_input("DPRA LysD", min_value=0.1, max_value=100.0, value=1.0, step=1.0,)
    DPRA_CysD = st.number_input("DPRA CysD", min_value=0.1, max_value=100.0, value=1.0, step=1.0,)
    log_DPRA_mean = (DPRA_LysD + DPRA_CysD) / 2
    log_DPRA_mean=convert_to_log10(log_DPRA_mean)

    CD86_EC150= st.number_input("CD86 EC150 ", min_value=0.1, max_value=5000.0, value=1.0, step=1.0)
    CD54_EC200 = st.number_input("CD54 EC200 ", min_value=0.1, max_value=5000.0, value=1.0, step=1.0)
    CV75 = st.number_input("CV75", min_value=1.0, max_value=5000.0, value=1.0, step=1.0,)
    log_hCLAT_MIT = min(CD86_EC150, CD54_EC200, CV75)
    log_hCLAT_MIT=convert_to_log10(log_hCLAT_MIT )
    X= np.array([[log_DPRA_mean,log_hCLAT_MIT]])
    if st.button("Predict"):
     
      
    # Call the prediction function
      result=prediction_LOGEC3(X, scaler, model)
     
    # Convert the prediction back to the original scale
    
      
      result = inverse_transform(result)
      EC3_Value=result.item()
      EC3_Value=round(EC3_Value,2)
      if result is not None:
        if float(result) < (-1):
            result = 'Strong'
        elif float(result) >= (-1) and float(result) < 0:
            result = 'Strong'
        elif float(result) >= 0 and float(result) < 1:
            result = 'weak'
        elif float(result) >1:
            result = 'Weak'
        else:
            result = 'Non'
        
        st.success(f'The EC3 value is {EC3_Value} and the chemical potency is {result}')

if __name__=='__main__':
    main()
