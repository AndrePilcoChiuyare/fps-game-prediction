import requests
import streamlit as st
import pandas as pd
import json
import joblib

api_key = "8455d4662cfb441db06008d90d513c5d"

#CpuName, GpuName, GameName, GameSetting, Game Resolution

#Extrar los archivos pickle
def loadModels():
    lgbm_model = joblib.load('../models/regressor/lgbm_model_regressor.pkl')
    return lgbm_model
    
def get_game_info(game):
    info = requests.get(f"https://api.rawg.io/api/games/{game}?key={api_key}").json()
    return info

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
    
game_name_mapping = {
    "fortnite": "Fortnite",
    "minecraft": "Minecraft",
    "grandTheftAuto5": "Grand Theft Auto V",
    "apexLegends": "Apex Legends",
    "counterStrikeGlobalOffensive": "Counter-Strike: Global Offensive",
    "playerUnknownsBattlegrounds": "PlayerUnknown's Battlegrounds",
    "overwatch": "Overwatch",
    "battlefield1": "Battlefield 1",
    "dota2": "Dota 2",
    "leagueOfLegends": "League of Legends",
    "rocketLeague": "Rocket League",
    "callOfDutyBlackOps4": "Call of Duty: Black Ops 4",
    "worldOfTanks": "World of Tanks",
    "battlefield4": "Battlefield 4",
    "arkSurvivalEvolved": "ARK: Survival Evolved",
    "rust": "Rust",
    "aWayOut": "A Way Out",
    "airMechStrike": "AirMech Strike",
    "battletech": "BattleTech",
    "callOfDutyWW2": "Call of Duty: WWII",
    "destiny2": "Destiny 2",
    "farCry5": "Far Cry 5",
    "frostpunk": "Frostpunk",
    "pathOfExile": "Path of Exile",
    "radicalHeights": "Radical Heights",
    "rainbowSixSiege": "Rainbow Six Siege",
    "seaOfThieves": "Sea of Thieves",
    "starcraft2": "StarCraft II",
    "totalWar3Kingdoms": "Total War: Three Kingdoms",
    "warframe": "Warframe",
}

game_name_mapping_api = {
    "Fortnite": "fortnite",
    "Minecraft": "minecraft",
    "Grand Theft Auto V": "grand-theft-auto-v",
    "Apex Legends": "apex-legends",
    "Counter-Strike: Global Offensive": "counter-strike-global-offensive",
    "PlayerUnknown's Battlegrounds": "playerunknowns-battlegrounds",
    "Overwatch": "overwatch",
    "Battlefield 1": "battlefield-1",
    "Dota 2": "dota-2",
    "League of Legends": "league-of-legends",
    "Rocket League": "rocket-league",
    "Call of Duty: Black Ops 4": "call-of-duty-black-ops-4",
    "World of Tanks": "world-of-tanks",
    "Battlefield 4": "battlefield-4",
    "ARK: Survival Evolved": "ark-survival-evolved",
    "Rust": "rust",
    "A Way Out": "a-way-out",
    "AirMech Strike": "airmech-strike",
    "BattleTech": "battletech",
    "Call of Duty: WWII": "call-of-duty-wwii",
    "Destiny 2": "destiny-2",
    "Far Cry 5": "far-cry-5",
    "Frostpunk": "frostpunk",
    "Path of Exile": "path-of-exile",
    "Radical Heights": "radical-heights",
    "Rainbow Six Siege": "rainbow-six-siege",
    "Sea of Thieves": "sea-of-thieves",
    "StarCraft II": "starcraft-ii",
    "Total War: Three Kingdoms": "total-war-three-kingdoms",
    "Warframe": "warframe",
}
    
# Update the user_input_parameters function
def user_input_parameters(variables_dict):
    # Sidebar inputs
    cpu_name = st.sidebar.selectbox('CPU', variables_dict['CpuName'])
    gpu_name = st.sidebar.selectbox('GPU', variables_dict['GpuName'])
    
    # Use well-written names for display but store the raw name for processing
    raw_game_name = st.sidebar.selectbox(
        'Game Name', variables_dict['GameName'], format_func=lambda x: game_name_mapping.get(x, x)
    )
    game_setting = st.sidebar.selectbox('Game Setting', variables_dict['GameSetting'])
    game_resolution = st.sidebar.selectbox('Game Resolution', variables_dict['GameResolution'])
    
    cpu = variables_dict['CpuName'][cpu_name]
    gpu = variables_dict['GpuName'][gpu_name]
    
    # Create the input data
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
        
        'GameName': [raw_game_name],  # Keep the raw game name for processing
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
    
    # Return the input data and the well-written game name for display
    features = pd.DataFrame(data)
    return features, game_name_mapping[raw_game_name]
    
def main():
    st.title('FPS calculator')
    st.sidebar.header('Choose your hardware and game settings')
    
    variables_dict = loadJSON()
    lgbm_model = loadModels()
    df, display_game_name = user_input_parameters(variables_dict)
    
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

    if st.button('Calculate FPS'):
        api_game_name = game_name_mapping_api[display_game_name]
        game_info = get_game_info(api_game_name)
        prediction = lgbm_model.predict(data)
        # title
        st.title(f"{display_game_name}")
        st.image(game_info['background_image'], width=800)
        if prediction > 100:
            st.success(f"Predicted FPS: {prediction}")
        elif prediction > 60:
            st.warning(f"Predicted FPS: {prediction}")
        else:
            st.error(f"Predicted FPS: {prediction}")
        st.write(f"Website: {game_info['website']}")
        # insert html code
        st.markdown(game_info['description'], unsafe_allow_html=True)
        if game_info['metacritic'] != None:
            if game_info['metacritic'] > 75:
                st.success(f"Metacritic score: {game_info['metacritic']}")
            elif game_info['metacritic'] > 50:
                st.warning(f"Metacritic score: {game_info['metacritic']}")
            else:
                st.error(f"Metacritic score: {game_info['metacritic']}")
        

if __name__ == '__main__':
    main()