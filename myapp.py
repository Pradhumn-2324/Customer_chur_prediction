import streamlit as st
import pandas as pd
# import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import joblib
import warnings
warnings.filterwarnings("ignore")
from keras.models import load_model
import plotly.graph_objects as go
from plotly.subplots import make_subplots


model2 = load_model("my_model.h5")




with open("Churn_pred.joblib","rb") as file:
        pipeline = joblib.load(file)


def  Transformation(X):
    X['Gender'] = X['Gender'].map({"Male":0,"Female":1})
    X['Location'] = X['Location'].map({'Los Angeles': 0, 'New York': 1, 'Miami': 2, 'Chicago': 3, 'Houston': 4})
    X = X.astype('float32')
    
    scaler = pipeline.named_steps['scaler']
    pca = pipeline.named_steps['pca']
    
    X_ans_scale = scaler.transform(X)
    X_pca = pca.transform(X_ans_scale)
    
    return X_pca


def NaiveByes(df,flag=False):
    m_nb = pipeline.named_steps['GNaiveB']
    if flag:
        return m_nb.predict(df)[0]
    return m_nb.predict_proba(df)[0]


def Kneighbour(df,flag=False):
    m_knn = pipeline.named_steps['KNeighbour']
    if flag:
        return m_knn.predict(df)[0]
    return m_knn.predict_proba(df)[0]

def Randomforest(df,flag=False):
    m_for = pipeline.named_steps['RandForest']
    if flag:
        return m_for.predict(df)[0]
    return m_for.predict_proba(df)[0]

def Ensemblemod(df,flag=False): 
    m_ensemble = pipeline.named_steps['Ensemble']
    if flag:
        return m_ensemble.predict(df)[0]
    return m_ensemble.predict_proba(df)[0]


def Logistic(df):
    m_log = pipeline.named_steps['Logistic']
    
    return m_log.predict_proba(df)[0]

def Ann_model(df):
    prediction_probabilities = model2.predict(df)

    probability_of_positive_class = prediction_probabilities[0][0]
    return probability_of_positive_class*100    


def main():
    st.title("Churn Prediction Application")

    
    with st.form("User Information"):
        name = st.text_input("Name")
        if name: 
            st.header(f"Welcome {name}")
            st.subheader("Please fill the following details of your customer")
        
        age = st.number_input("Age", min_value=18, max_value=100)

        gender = st.selectbox("Gender",("Male",'Female'))

        loc = st.selectbox("Location",['Houston', 'Los Angeles', 'Miami', 'Chicago', 'New York'])
        
        subscription = st.number_input("Subscription Length By Month",min_value=1,max_value=24)
        
        Bill = st.number_input("Total Bill",min_value=30)
        
        TotalGB = st.number_input("Total Data",min_value=10,max_value=500)
        
        
        
        submitted = st.form_submit_button("Submit")

        
        if submitted:
            st.header("This machine models are predicting good ")
            data = {'Age': [age],
                        'Gender': [gender],
                        'Location': loc,
                        'Subscription_Length_Months': [subscription],
                        'Monthly_Bill': [Bill],
                        'Total_Usage_GB': [TotalGB]
                        }
            df = pd.DataFrame(data)
            
            pca = Transformation(df)
            
            predictions = {
            "Gaussian Naive Byes": NaiveByes(pca),
            "K-Neighbour": Kneighbour(pca),
            "Random Forest": Randomforest(pca) ,
            "Ensemble": Ensemblemod(pca)
        }
            labels=["Not leave","Leave"]
            fig, ax = plt.subplots(2, 2, figsize=(12, 6))
            ax[0][0].pie(predictions['Gaussian Naive Byes'], labels=labels, autopct='%1.1f%%',wedgeprops={'edgecolor': 'black'}, startangle=90)
            ax[0][0].set_title('Naive byes')
            ax[0][1].pie(predictions['K-Neighbour'], labels=labels, autopct='%1.1f%%',wedgeprops={'edgecolor': 'black'}, startangle=90)
            ax[0][1].set_title('K-Neighbour')
            ax[1][0].pie(predictions['Random Forest'], labels=labels, autopct='%1.1f%%', wedgeprops={'edgecolor': 'black'},startangle=90)
            ax[1][0].set_title('Random Forest')
            ax[1][1].pie(predictions['Ensemble'], labels=labels, autopct='%1.1f%%', wedgeprops={'edgecolor': 'black'},startangle=90)
            ax[1][1].set_title('Ensemble model')
            st.pyplot(fig)
            st.write('''As we can see that with majority employee will not leave''')
            
            ans_dic= {'GNB':[NaiveByes(pca,True)], 
                      'RandFor':[Randomforest(pca,True)],
                      'knn' : [Kneighbour(pca,True)],
                      'ensemble' : [Ensemblemod(pca,True)] }
            ans_df =pd.DataFrame(ans_dic)
            log_pred = Logistic(ans_df)
            ann_pred = [Ann_model(ans_df)*100,100-Ann_model(ans_df)*100]
            print(ann_pred)
            fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{'type': 'domain'}, {'type': 'domain'}]],
            subplot_titles=("Logistic", "Neural Model"),
            # Add annotations in the center of the donut pies
            
            )

            # Create Pie chart traces
            trace1 = go.Pie(labels=labels, values=log_pred, name='Logistic')
            trace2 = go.Pie(labels=labels, values=ann_pred, name='Neural Model')

            # Add traces to the subplots
            fig.add_trace(trace1, row=1, col=1)
            fig.add_trace(trace2, row=1, col=2)

            # Update subplot titles
            fig.update_layout(title_text="Pie Chart Subplots")

            # Show the plot using Streamlit
            st.plotly_chart(fig)
if __name__ == "__main__":
    main()
