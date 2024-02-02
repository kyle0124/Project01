import streamlit as st
import json
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
from sklearn import metrics, preprocessing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


import warnings
warnings.filterwarnings('ignore')


####################################################################################################

if 'name' not in st.session_state :
    st.session_state['name'] = ''

if 'current_age' not in st.session_state :
    st.session_state['current_age'] = 21

if 'start_age' not in st.session_state :
    st.session_state['start_age'] = 20

if 'smokedose' not in st.session_state :
    st.session_state['smokedose'] = 1

if 'smokedur' not in st.session_state :
    st.session_state['smokedur'] = 1

if 'gender' not in st.session_state :
    st.session_state['gender'] = 1

####################################################################################################








####################################################################################################

def LoadJson(path) :
    f = open(path, 'r')
    result = json.load(f)
    return result



def cancerAge(smokedose, smokedur, start_age, gender) :
    df = pd.read_csv('./test02.csv')
    y = df['AGE']
    X = df.drop(columns=['AGE', 'SEX', 'smoke', 'smokequityr', 'total_smoke', 
                           'height', 'weight', 'quit_smokedur'])
     
    depth_grid = np.arange(1, 20)
    max_leaf_nodes_grid = np.arange(2, 100, 2)
    min_sample_leaf_grid = np.arange(10, 100, 2)

    parameters = {'max_depth':depth_grid, 
                'min_samples_leaf':min_sample_leaf_grid,
                'max_leaf_nodes':max_leaf_nodes_grid}
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    randCV = RandomizedSearchCV(GradientBoostingRegressor(),
                                param_distributions=parameters,
                                n_jobs=-1, cv=10)
    randCV.fit(X_train, y_train)
    best_estimator_rand = randCV.best_estimator_
    pred_rand = best_estimator_rand.predict(X_test)

    percentage = np.round(metrics.r2_score(y_test, pred_rand), 3) * 100

    tmp_list = [[smokedose, smokedur, start_age, gender, 1]]
    tmp = pd.DataFrame(tmp_list, columns=X.columns)
    print(tmp)
    y_pred = best_estimator_rand.predict(tmp)

    result = np.round(y_pred[0], 3)
    return_list = [result, gender, percentage]
    return return_list

#@st.cache_data
def deathAge(cancer_age) : 
    df = pd.read_csv('./test/test02_jounal-24-2.csv')
    age = cancer_age[0]
    gender = cancer_age[1]
    difference_age = np.round(df[(df['Age'] == np.round(age)) & (df['gender'] == gender)]['Difference_pred_abs'].values[0], 3)
    # print(age, gender)
    # print(difference_age)
    death_age = age + difference_age

    return death_age



#@st.cache_data
def deathDate(death_age, current_age) :
    today = datetime.today().date()
    remain_days = (death_age - current_age) * 365
    # print(remain_days)
    # print(today + timedelta(days=remain_days))
    return today + timedelta(days=remain_days)




####################################################################################################

col1, col2, col3 = st.columns([1, 3, 1])
with col1 :
    # lottie = LoadJson('lottie-stock-candle-loading.json')
    # st_lottie(lottie, loop=True, speed=1, width=200, height=200)
    ''


with col2 :
    ''
    ''
    st.image('./resources/DeathNote_white.png')
    #st.title('Death Note')
    st.markdown("""
        <style>
        button[title="View fullscreen"]{
        visibility: hidden;
        }
        .custom-font {
        font-size:20px !important;
        float: right;
        }
        </style>
        """, unsafe_allow_html=True)
    st.markdown('<p class="custom-font">Predict your date of death</p>', unsafe_allow_html=True)

with col3 :
    ''



####################################################################################################
    

with st.sidebar.form(key='smoke', clear_on_submit=False) :
    st.header('당신의 금연을 응원합니다')

    name = st.text_input('성함')
    current_age = st.number_input('연령', 10, 100)

    gender = st.radio('성별', ['남성', '여성'])
    if gender == '남성' :
        gender = 1
    else :
        gender = 0

    smokedose_list = [0.1, 0.3, 0.5, 1, 1.5, 2]
    smokedose = st.radio('하루 흡연량(갑)', smokedose_list)

    smokedur = st.number_input('흡연 기간(년)', 1, 100)





    if st.form_submit_button('Submit') :
        st.session_state['name'] = name
        st.session_state['current_age'] = current_age
        st.session_state['smokedose'] = smokedose
        st.session_state['smokedur'] = smokedur
        st.session_state['gender'] = gender
        # print('======================')
        # print(name)
        # print(current_age)
        # print(smokedose)
        # print(smokedur)
        # print(gender)
        # print('======================')
        # print(st.session_state['current_age'])
        st.experimental_rerun()

# st.write(gender)
# st.write(type(gender))
# st.write(current_age)
# st.write(type(current_age))

# st.write(type(smokedur))
# st.write(type(smokedose))

# print(st.session_state['current_age'], st.session_state['smokedur'])

start_age = st.session_state['current_age'] - st.session_state['smokedur']

# print(st.session_state['smokedose'], st.session_state['smokedur'],
#                        start_age, st.session_state['gender'])
# print(start_age)

cancer_age = cancerAge(st.session_state['smokedose'], st.session_state['smokedur'],
                       start_age, st.session_state['gender'])


st.subheader(st.session_state['name'] + '님의 금연을 응원합니다')






# st.write(cancer_age)

''
'---'
''



col4, col5, col6 = st.columns([1, 3, 1])
with col4 :
    ''
with col5 :
    death_age = deathAge(cancer_age)
    death_date = deathDate(death_age, st.session_state['current_age'])
    st.write('날짜')
    st.error(death_date)
    st.warning(death_date - datetime.today().date())
    st.write('정확도')
    st.success(cancer_age[2])


with col6 :
    ''


''
'---'
''


























