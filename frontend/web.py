import requests
import pickle
import streamlit as st
import pandas as pd
import json
import joblib

api_key = "8455d4662cfb441db06008d90d513c5d"

#CpuName, GpuName, GameName, GameSetting, Game Resolution

#Extrar los archivos pickle
def loadModels():
    rf_model = joblib.load('../models/regressor/rf_model_regressor.pkl')
    lgbm_model = joblib.load('../models/regressor/lgbm_model_regressor.pkl')
    return rf_model, lgbm_model
    
def get_background_image(game):
    bg_image = requests.get(f"https://api.rawg.io/api/games/{game}?key={api_key}").json()['background_image']
    return bg_image

def loadJSON():
    with open('../data/json/variables_dict.json') as json_file:
        return json.load(json_file)

def loadPipeline():
    return joblib.load('../models/pipeline.pkl')

def classify(num):
    if num == 0:
        return 'Playable'
    elif num == 1:
        return 'Good'
    else:
        return 'Excelent'
    
def user_input_parameters(variables_dict):
    cpu_name = st.sidebar.selectbox('CPU', variables_dict['CpuName'])
    gpu_name = st.sidebar.selectbox('GPU', variables_dict['GpuName'])
    
    game_name = st.sidebar.selectbox('Game Name', variables_dict['GameName'])
    game_setting = st.sidebar.selectbox('Game Setting', variables_dict['GameSetting'])
    game_resolution = st.sidebar.selectbox('Game Resolution', variables_dict['GameResolution'])
    
    cpu = variables_dict['CpuName'][cpu_name]
    gpu = variables_dict['GpuName'][gpu_name]
    
    # Convertir todos los valores a listas
    data = {
        'CpuNumberOfCores': [cpu['CpuNumberOfCores']],
        'CpuNumberOfThreads': [cpu['CpuNumberOfThreads']],
        'CpuBaseClock': [cpu['CpuBaseClock']],
        'CpuCacheL1': [cpu['CpuCacheL1']],
        'CpuCacheL2': [cpu['CpuCacheL2']],
        'CpuCacheL3': [cpu['CpuCacheL3']],
        'CpuFrequency': [cpu['CpuFrequency']],
        'CpuMultiplier': [cpu['CpuMultiplier']],
        'CpuMultiplierUnlocked': [cpu['CpuMultiplierUnlocked']],
        'CpuProcessSize': [cpu['CpuProcessSize']],
        'CpuTDP': [cpu['CpuTDP']],
        'CpuTurboClock': [cpu['CpuTurboClock']],
        
        'GpuArchitecture': [gpu['GpuArchitecture']],
        'GpuBandwidth': [gpu['GpuBandwidth']],
        'GpuBaseClock': [gpu['GpuBaseClock']],
        'GpuBoostClock': [gpu['GpuBoostClock']],
        'GpuBus nterface': [gpu['GpuBus nterface']],
        'GpuDieSize': [gpu['GpuDieSize']],
        'GpuDirectX': [gpu['GpuDirectX']],
        'GpuFP32Performance': [gpu['GpuFP32Performance']],
        'GpuMemoryBus': [gpu['GpuMemoryBus']],
        'GpuMemorySize': [gpu['GpuMemorySize']],
        'GpuMemoryType': [gpu['GpuMemoryType']],
        'GpuOpenCL': [gpu['GpuOpenCL']],
        'GpuPixelRate': [gpu['GpuPixelRate']],
        'GpuProcessSize': [gpu['GpuProcessSize']],
        'GpuNumberOfROPs': [gpu['GpuNumberOfROPs']],
        'GpuNumberOfShadingUnits': [gpu['GpuNumberOfShadingUnits']],
        'GpuNumberOfTMUs': [gpu['GpuNumberOfTMUs']],
        'GpuTextureRate': [gpu['GpuTextureRate']],
        'GpuNumberOfTransistors': [gpu['GpuNumberOfTransistors']],
        
        'GameName': [game_name],
        'GameSetting': [game_setting],
        'GameResolution': [game_resolution],
                
        'GpuCpuRatio': [gpu['GpuBoostClock'] / cpu['CpuTurboClock']],
        'CacheCoreRatio': [(cpu['CpuCacheL1'] + cpu['CpuCacheL2'] + cpu['CpuCacheL3']) / cpu['CpuNumberOfCores']],
        'CpuOverclockPotential': [(cpu['CpuTurboClock'] - cpu['CpuBaseClock']) / cpu['CpuBaseClock']],
        'EffectiveMemory': [gpu['GpuMemoryBus'] * gpu['GpuBandwidth']],
        'PerformanceClockRatio': [cpu['CpuTurboClock'] / gpu['GpuBoostClock']],
        'PerformancePerWatt': [gpu['GpuFP32Performance'] / gpu['GpuNumberOfTransistors']],
        'CpuCoreThreadRatio': [cpu['CpuNumberOfCores'] / cpu['CpuNumberOfThreads']],
        'CpuPowerIndex': [cpu['CpuTurboClock'] * cpu['CpuNumberOfCores'] * cpu['CpuCacheL3']],
        'GpuAverageClock': [(gpu['GpuBaseClock'] + gpu['GpuBoostClock']) / 2],
    }
    
    # Ahora creamos el DataFrame
    features = pd.DataFrame(data)
    return features
    
def main():
    st.title('Modelamiento de FPS by Andre Pilco, John Sovero e Iam Alvarez')
    st.sidebar.header('User Input Parameters')
    
    variables_dict = loadJSON()
    rf_model, lgbm_model = loadModels()
    df = user_input_parameters(variables_dict)
    
    pipeline = loadPipeline()
    encoded = pipeline.transform(df)
    
    data = pd.DataFrame(encoded, columns=['nominal__GpuArchitecture_1', 'nominal__GpuArchitecture_2',
       'nominal__GpuArchitecture_3', 'nominal__GpuArchitecture_4',
       'nominal__GpuArchitecture_5', 'nominal__GpuArchitecture_6',
       'nominal__GpuArchitecture_7', 'nominal__GpuArchitecture_8',
       'nominal__GpuArchitecture_9', 'nominal__GpuArchitecture_10',
       'nominal__GpuArchitecture_11', 'nominal__GpuArchitecture_12',
       'nominal__GpuArchitecture_13', 'nominal__GpuArchitecture_14',
       'nominal__GameName_1', 'nominal__GameName_2', 'nominal__GameName_3',
       'nominal__GameName_4', 'nominal__GameName_5', 'nominal__GameName_6',
       'nominal__GameName_7', 'nominal__GameName_8', 'nominal__GameName_9',
       'nominal__GameName_10', 'nominal__GameName_11', 'nominal__GameName_12',
       'nominal__GameName_13', 'nominal__GameName_14', 'nominal__GameName_15',
       'nominal__GameName_16', 'nominal__GameName_17', 'nominal__GameName_18',
       'nominal__GameName_19', 'nominal__GameName_20', 'nominal__GameName_21',
       'nominal__GameName_22', 'nominal__GameName_23', 'nominal__GameName_24',
       'nominal__GameName_25', 'nominal__GameName_26', 'nominal__GameName_27',
       'nominal__GameName_28', 'nominal__GameName_29', 'nominal__GameName_30',
       'ordinal__GpuBus nterface', 'ordinal__GpuDirectX',
       'ordinal__GpuMemoryType', 'ordinal__GameSetting',
       'remainder__CpuNumberOfCores', 'remainder__CpuNumberOfThreads',
       'remainder__CpuBaseClock', 'remainder__CpuCacheL1',
       'remainder__CpuCacheL2', 'remainder__CpuCacheL3',
       'remainder__CpuFrequency', 'remainder__CpuMultiplier',
       'remainder__CpuMultiplierUnlocked', 'remainder__CpuProcessSize',
       'remainder__CpuTDP', 'remainder__CpuTurboClock',
       'remainder__GpuBandwidth', 'remainder__GpuBaseClock',
       'remainder__GpuBoostClock', 'remainder__GpuDieSize',
       'remainder__GpuFP32Performance', 'remainder__GpuMemoryBus',
       'remainder__GpuMemorySize', 'remainder__GpuOpenCL',
       'remainder__GpuPixelRate', 'remainder__GpuProcessSize',
       'remainder__GpuNumberOfROPs', 'remainder__GpuNumberOfShadingUnits',
       'remainder__GpuNumberOfTMUs', 'remainder__GpuTextureRate',
       'remainder__GpuNumberOfTransistors', 'remainder__GameResolution',
       'remainder__GpuCpuRatio', 'remainder__CacheCoreRatio',
       'remainder__CpuOverclockPotential', 'remainder__EffectiveMemory',
       'remainder__PerformanceClockRatio', 'remainder__PerformancePerWatt',
       'remainder__CpuCoreThreadRatio', 'remainder__CpuPowerIndex',
       'remainder__GpuAverageClock'])
    
    option = ['Random Forest', 'LGBM']
    model = st.sidebar.selectbox('Which model you like to use?', option)
    
    st.subheader('User Input Parameters')
    st.subheader(model)

    if st.button('RUN'):
        if model == 'Random Forest':
            prediction = rf_model.predict(data)
            st.success(prediction)
        elif model == 'LGBM':
            prediction = lgbm_model.predict(data)
            st.success(prediction)

if __name__ == '__main__':
    main()