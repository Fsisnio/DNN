
import streamlit as st
import torch
import numpy as np
import altair as alt
from model import Classifier  # Ensure this matches your model's class

st.set_page_config(
    page_title="Bank Deposit DNN Prediction Model",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state='auto')

alt.themes.enable("dark")



# CSS to inject contained in a multiline string
custom_css = """
    <style>
        .title {
            color: #FF0000;
            font-size: 24px;
            text-align: center;
        }
        .overview {
            font-size: 18px;
            margin-top: 20px;
        }
        .data-table {
            margin-top: 20px;
        }
        /* Additional custom CSS can go here */
    </style>
"""



# Inject custom CSS with markdown
st.markdown(custom_css, unsafe_allow_html=True)

# Use the custom CSS classes in markdown
st.markdown('<div class="title">This is a DNN Prediction Model for a Portuguese Bank Marketing</div>', unsafe_allow_html=True)
st.markdown("""
    <div class="overview">
    Implementation of a DNN prediction model to determine whether clients of a banking institution will subscribe to a bank term deposit or not. The data is related to direct marketing campaigns of a Portuguese banking institution. 
    These marketing campaigns were based on phone calls. Often, several contacts with the same client were necessary to determine if the product (bank term deposit) would be subscribed to ("yes") or not ("no"). The current model takes into account 20 features from each client. 
    The model was developed with a base of 45,307 different clients. </div>
""", unsafe_allow_html=True)



import streamlit as st

with st.sidebar:
    st.title('🧮Bank Deposit DNN Prediction Mode')
    
    # Adding a comment section
    st.subheader('Leave a Comment')
    user_comment = st.text_area("Share your thoughts:", help='Please, Write your comment here')
    
    # Optional: Add a button to submit the comment
    if st.button('Submit Comment'):
        st.write('Comment submitted:', user_comment)
    
    # Adding a comment section
    st.subheader('Technologies and Librairies')
    st.write('Streamlit,Pytorch,Pandas, NumPy,Scikit-learn, Matplotlib, HTML, CSS.')





# Load the model
def load_model():
    model = Classifier(input_size=20)  # Adjust the input_size accordingly
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Streamlit UI
st.title('Bank Deposit DNN Prediction Mode')
st.write('Enter the input values for prediction:')

# Define a list of feature names and subcategories for features that need them
feature_names = ['Age', 'Occupation', 'Marital', 'Education', 'Default', 'Housing', 'Loan', 'Contact', 'Month', 'Day_of_week', 'Duration', 'Campaign', 'Pdays', 'Previous', 'Poutcome', 'Emp.var.rate', 'Cons.price.idx', 'Cons.conf.idx', 'Euribor3m', 'Nr.employed']
subcategories = {
    'Occupation': ['administrator', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'],
    'Education': ["unknown", "secondary", "primary", "tertiary"],
    'Marital': ["married", "divorced", "single"],
    'Default': ["Yes", "No"],
    'Loan': ["Yes", "No"],
    'Housing' :  ["Yes", "No"],
    'Contact': ["unknown", "telephone", "cellular"],
    'Day_of_week'   :  ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
    'Month'   :  ["January", "February", "March", "April", "May", "June","July", "August", "September", "October", "November", "December"],
    'Poutcome' : ["Unknown","failure","success"]
    

    
}


# Adding a section for the legend to describe the variables
with st.expander("Legend for Variables", expanded=False):
    st.markdown("""
    **Age:** Client's age in years.

    **Occupation:** Client's job type. Options include administrator, blue-collar, entrepreneur, and more.

    **Marital:** Client's marital status. Options include married, divorced, single.

    **Education:** Level of education. Options include unknown, secondary, primary, tertiary.

    **Default:** Indicates if the client has credit in default. Yes or No.

    **Housing:** Indicates if the client has a housing loan. Yes or No.

    **Loan:** Indicates if the client has a personal loan. Yes or No.

    **Contact:** Contact communication type. Options include unknown, telephone, cellular.

    **Month:** Last contact month of the year.

    **Day_of_week:** Last contact day of the week.

    **Duration:** Last contact duration, in seconds (numeric).
    
    **campaign:** number of contacts performed during this campaign and for this client (numeric, includes last contact).
                
    **pdays:** number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted).
                
    **previous:** number of contacts performed before this campaign and for this client (numeric).
                
    **poutcome:** outcome of the previous marketing campaign (categorical: "unknown","other","failure","success").
                
    **emp.var.rate:** employment variation rate. It offers insights into the employment trends during the period covered by the data .
                
    **cons.price.idx:** the consumer price index.This numeric value represents the level of prices associated with consumer goods and services purchased by households, providing insight into the inflation rate affecting the cost of living during the period covered by the data.
                
    **cons.conf.idx:**  consumer confidence index.It measures the degree of confidence that households have regarding the performance of the economy.
                
    **euribor3m:** Euro Interbank Offered Rate. It  refers to the Euro Interbank Offered Rate for loans with a three-month maturity. It's a benchmark rate that reflects the average interest rate at which major European banks are prepared to lend to one another in the euro interbank market for three-month loans.
                
    **nr.employed:** refers to the number of employees indicator, commonly used as a quarterly metric in various datasets, particularly those related to economics and finance.             .

    *Please note that the options listed under each category provide a comprehensive overview of the types of data expected.
    """)

# Initialize an empty dictionary to store input features
input_features = {}

# Create input fields for each feature, including subcategories
# Create input fields for each feature, including subcategories
for feature in feature_names:
    if feature == 'Age':
        # Setting minimum age limit to 18
        input_features[feature] = st.number_input(feature, min_value=18, value=18)
    elif feature in subcategories:
        option = st.selectbox(feature, options=subcategories[feature])
        input_features[feature] = option
    else:
        input_features[feature] = st.number_input(feature, value=0.0)



# Button to make prediction
if st.button('Predict'):
    input_values = [float(value) if not isinstance(value, str) else float(subcategories[feature].index(value)) for feature, value in input_features.items()]
    input_array = np.array([input_values], dtype=np.float32)
    input_tensor = torch.from_numpy(input_array)
    
    with torch.no_grad():
        prediction = model(input_tensor)
        predicted_probs = torch.softmax(prediction, dim=1)
        predicted_class = predicted_probs.argmax(1).item()
    
    # Define a dictionary to map predicted classes to sentences
    class_to_sentence = {
        0: "The client is predicted not to subscribe to the bank term deposit.",
        1: "The client is predicted to subscribe to the bank term deposit."
    }
    
    # Use the predicted class to get the corresponding sentence
    prediction_sentence = class_to_sentence[predicted_class]
    
    st.write(prediction_sentence)


with st.expander('About', expanded=True):
    st.write('''
        - Data Source: https://archive.ics.uci.edu/dataset/222/bank+marketing
        - Purpose: Building and Deployment of a DNN Model 
        - Author: Spéro FALADE
        - Contact: faladespero1@gmail.com
        - LinkedIn:https://www.linkedin.com/in/sp%C3%A9ro-falade-977180103/
        - Copyright:SFApril@2024       
    ''')