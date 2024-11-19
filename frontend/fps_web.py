#importar librerias
import streamlit as st
import pickle
import pandas as pd

#Extrar los archivos pickle
with open('rf_model.pkl', 'rb') as rf:
    rf_model = pickle.load(rf)

with open('lgbm_model.pkl', 'rb') as lgbm:
    lgbm_model = pickle.load(lgbm)


#funcion para clasificar las plantas 
def classify(num):
    if num == 0:
        return 'Setosa'
    elif num == 1:
        return 'Versicolor'
    else:
        return 'Virginica'

def main():
    #titulo
    st.title('Modelamiento de Iris by @L33PIF')
    #titulo de sidebar
    st.sidebar.header('User Input Parameters')

    #funcion para poner los parametrso en el sidebar
    def user_input_parameters():
        sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
        sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
        petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
        petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
        data = {'sepal_length': sepal_length,
                'sepal_width': sepal_width,
                'petal_length': petal_length,
                'petal_width': petal_width,
                }
        features = pd.DataFrame(data, index=[0])
        return features

    df = user_input_parameters()

    #escoger el modelo preferido
    option = ['Linear Regression', 'Logistic Regression', 'SVM']
    model = st.sidebar.selectbox('Which model you like to use?', option)

    st.subheader('User Input Parameters')
    st.subheader(model)
    st.write(df)

    if st.button('RUN'):
        if model == 'Random Forest':
            st.success(classify(rf_model.predict(df)))
        elif model == 'LGBM Classifier':
            st.success(classify(lgbm_model.predict(df)))


if __name__ == '__main__':
    main()
    