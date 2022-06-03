from datetime import datetime
from email.policy import default
from unicodedata import category
import pandas as pd 
import numpy as np 
import streamlit as st
import requests
import fastf1
import fastf1.plotting
import matplotlib.pyplot as plt
from tables import Cols
from src.about import about_cs
from pathlib import Path
import base64
from dateutil import parser
import plotly.graph_objects as go
import copy
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm


# fastf1.Cache.enable_cache('./cache')  


# default subroutines
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def load_miscellaneous_data():

    rounds = pd.read_csv('./data/year_wise_rounds.csv').set_index('Unnamed: 0')
    
    return rounds



def instantiate_API_keys():


    API_elements = {

            'drivers_wr': 'http://ergast.com/api/f1/{}/{}/drivers.json',
            'drivers_wor': 'http://ergast.com/api/f1/{}/drivers.json'

    }

    return API_elements



def drivers_summary(api_elements, yearwise_rounds):

    # API access links 
    drivers_wr = api_elements['drivers_wr']
    drivers_wor = api_elements['drivers_wor']

    # test
    data = requests.get(drivers_wor.format('2019'))
    
    st.write(data.json())

def date_modifier(date_obj, type=1):

    if date_obj != 'None':
        if type == 1:
            morphed_date = date_obj.strftime('%d %b, %Y')
        elif type == 2:
            morphed_date = date_obj.strftime('%d %b')
        return morphed_date
    else:
        return None
        


@st.cache(persist=True, suppress_st_warning=True, allow_output_mutation=True)
def summarised_session(session,data,mode=None,round_number=None):

    
    
    if session == "Qualifying":
        items_required = ['DriverNumber', 'BroadcastName', 'Abbreviation', 'TeamName', 'TeamColor', 'FullName', 'Position']
        
        if mode == 'Knocked':
            
            round_dict = {'Q1':'Q1', 'Q2':'Q1','Q3':'Q2'}
            items_required.append(round_dict[round_number])
            package = {}
            temp_package = {}

            if data[data[round_number]=='0'].empty:
                return None
        
            else:
                for item in items_required:
                    temp_package[item] = data[data[round_number]=='0'].T.loc[item]
                
                package['driver_data'] = temp_package
                package['round_number'] = round_number
                
                return package
        
        elif mode == 'Final Grid-Positions':

            x = data.copy(deep=True)
            
            package = {}
            temp_package = {}
            
            for item in items_required:
                temp_package[item] = x[item].to_list()
    
            package['driver_data'] = temp_package
        
            order = x['DriverNumber'].to_list()
            x = x.set_index('DriverNumber',drop=True)
            Q3t = x[x['Q3'] !='0']['Q3']
            Q3_idx = Q3t.index
            Q2t = x[x['Q2'] !='0']['Q2'].drop(Q3_idx,axis=0)
            Q2_idx = Q2t.index
            idx_compiled = list(Q3_idx) + list(Q2_idx)

            if x[x['Q1'] == '0'].empty:
                Q1t = x[x['Q1'] != '0']['Q1']
                Q1t = Q1t.drop(idx_compiled, axis=0)
                times= Q3t.append((Q2t,Q1t))
                
            else:
                no_time = x[x['Q1'] == '0']['Q1']
                Q1t = x[x['Q1'] != '0']['Q1']
                Q1t = Q1t.drop(idx_compiled, axis=0)
                times= Q3t.append((Q2t,Q1t,no_time))

            package['Times'] = times

            return package

        elif mode in ["Q1 Grid-Positions", "Q2 Grid-Positions", "Q3 Grid-Positions"]:

            data = data[data[round_number] != '0']
            temp_package = {}
            package = {}

            for item in items_required:
                temp_package[item] = data[item].to_list()
                
            package['driver_data'] = temp_package
            package['Times'] = data[round_number]

            return package
            
    
    elif mode in ["Practice 1", "Practice 2", "Practice 3"]:
        
        pass

@st.experimental_singleton
def load_session_data(year, event, session_select):
    
    session_obj = fastf1.get_session(year, event, session_select)
    with st.spinner(session_obj.load()):
        session_results = session_obj.results.reset_index(drop=True)
        return session_results
    
@st.experimental_singleton
def return_session_object(year,event, session_select):

    session = fastf1.get_session(year, event, session_select)
    with st.spinner(session.load()):
        pass

    return session

@st.experimental_singleton(show_spinner=True)
def fetch_event_schedule(year):

    event_schedule = fastf1.get_event_schedule(year)
    event_names = event_schedule['EventName'].to_list()
    event_names.insert(0, 'List of GPs')

    return event_names, event_schedule



def display_schedule(year, circuit_cdf, circuits_rdf):

    event_schedule = fastf1.get_event_schedule(year)
    event_names = event_schedule['EventName'].to_list()

    # st.code(event_schedule['OfficialEventName'])

    
    circuit_flag = 0
    for event_name in event_names:

        # slicing data 
        event_data = event_schedule[event_schedule['EventName'] == event_name].T
        event_data = event_data.fillna('None')

        # circuit = circuits_df.loc[event_name, 'Circuits']
        # locality = circuits_df.loc[event_name, 'Localities']

        country = event_data.loc['Country'].values[0]

        # circuit name
        try:
            circuit = circuits_rdf.loc[event_name, 'Circuits']
            locality = circuits_rdf.loc[event_name, 'Localities']
        except:
            circuit = circuits_cdf.loc[country, 'Circuits']
            locality = circuits_cdf.loc[country, 'Localities']


        # packaging event summarised-information
        package = {}
        items = event_schedule.columns[:-1]
        for item in items:
            package[item] = event_data.loc[item].values[0]

        sessions_list = [package[x] for x in ['Session'+str(i) for i in range(1,6)]]

        # checkered flag
        current_date = datetime.now()
        if current_date > package['EventDate']:
            
            st.markdown(f'''<p style="font-size:30px; font-weight:bold; font-family:formula1, syne;"> <img src="https://countryflagsapi.com/png/{package["Country"]}" width="50">  <u>{package["EventName"]}</u>  |  <span style="font-size:23px;">{date_modifier(package["EventDate"])} <br>{circuit}, {locality} <img src='data:image/png;base64,{img_to_bytes('./assets/checkered-flag.png')}' class='img-fluid' width=50 ></span> </p>''',unsafe_allow_html=True)
        
        else:
            st.markdown(f'''<p style="font-size:30px; font-weight:bold; font-family:formula1, syne;"> <img src="https://countryflagsapi.com/png/{package["Country"]}" width="50">  <u>{package["EventName"]}</u>  |  <span style="font-size:23px;">{date_modifier(package["EventDate"])} <br>{circuit}, {locality} </span></p>''',unsafe_allow_html=True)


        cols = st.columns(len(sessions_list))
        for i, session_name in enumerate(sessions_list, start=0) :
            cols[i].markdown('> <p style="font-size:17px; font-weight:bold; font-family:formula1, syne;"><u>{}</u><p>'.format(session_name),unsafe_allow_html=True)
            cols[i].markdown('> <p style="font-size:13px; font-weight:bold; font-family:formula1, syne;">{}<p>'.format(date_modifier(package['Session'+str(i+1)+"Date"])),unsafe_allow_html=True)
        
        st.markdown('***')

def timedelta_conversion(timevar):
    

    if timevar.seconds == 0 and timevar.microseconds == 0:
        return 'No Time'
        
    else:
        return parser.parse(str(timevar).split(' ')[-1:][0]).strftime('%-M:%S:%f')

def delta_variation(driver_time, fastest_time):

    
    if driver_time > fastest_time:
        positive_delta = driver_time - fastest_time
        return '+{}'.format(str(positive_delta).split('.')[-1:][0][:4])
    elif driver_time < fastest_time:
        negative_delta = driver_time - fastest_time
        return '-{}'.format(str(positive_delta).split('.')[-1:][0][:4])
    else:
        return '000'

@st.experimental_singleton(show_spinner=True)
def fetch_circuits_data(year):

    url = f'http://ergast.com/api/f1/{year}.json'
    result= requests.get(url)
    races = result.json()['MRData']['RaceTable']['Races']
    localities = []
    countries = []
    circuits = []
    racenames = []
    for race in races:
        circuits.append(race['Circuit']['circuitName'])
        racenames.append(race['raceName'])
        countries.append(race['Circuit']['Location']['country'])
        localities.append(race['Circuit']['Location']['locality'])
        
    circuits_df = {'Countries': countries, 'Localities': localities, 'Circuits': circuits, 'Race Name':racenames, }
    circuits_rdf = pd.DataFrame(circuits_df).set_index('Race Name',drop=True)
    circuits_cdf = pd.DataFrame(circuits_df).set_index('Countries',drop=True)
    
    return circuits_cdf, circuits_rdf

def fetch_circuit_name():

    pass


def speed_visualisation(package):

    if len(package) == 2:
        iterations = 2

    else:
        iterations = 1

    fig = go.Figure()
    for i in range(iterations):

        x, y, color, AB = package[i]

        fig.add_trace(go.Scatter(x=x, y=y,
                                line = dict(color=color),
                            mode='lines',
                            name=f'{AB} Speed'))


    fig.update_layout(paper_bgcolor="white", template='gridon', showlegend=True)


    return fig

def gear_heatmap(x,y,tel,driver,event,year):
    '''

        This code snippet is lifted from the official fastf1 documentation 
    
    
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    gear = tel['nGear'].to_numpy().astype(float)

    cmap = cm.get_cmap('Paired')
    lc_comp = LineCollection(segments, norm=plt.Normalize(1, cmap.N+1), cmap=cmap)
    lc_comp.set_array(gear)
    lc_comp.set_linewidth(4)

    plt.gca().add_collection(lc_comp)
    plt.axis('equal')
    plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

    # title = plt.suptitle(
    #     f"Lap Gear Shift Visualization\n"
    #     f"{driver} - {event} {year}"
    # )

    cbar = plt.colorbar(mappable=lc_comp, label="Gear", boundaries=np.arange(1, 10))
    cbar.set_ticks(np.arange(1.5, 9.5))
    cbar.set_ticklabels(np.arange(1, 9))


    st.pyplot(plt.show())


def qualifying(summarised_results, year):

     # Driver dict
    drivers = summarised_results['FullName'].to_list()
    driver_dict = {}
    for driver in drivers:
        temp_list = temp_list = [summarised_results[summarised_results['FullName']==driver].loc[:,'Abbreviation'].values[0],
                                summarised_results[summarised_results['FullName']==driver].loc[:,'BroadcastName'].values[0],
                                summarised_results[summarised_results['FullName']==driver].loc[:,'TeamName'].values[0],
                                summarised_results[summarised_results['FullName']==driver].loc[:,'TeamColor'].values[0],
                                ]

        driver_dict[driver] = temp_list
    

    preference = st.sidebar.select_slider('Preference', [ 'Summarise', 'Get Nerdy!' ],key='mode of information')

    if preference == 'Summarise':

        st.markdown(f'''<h6 style="font-family:formula1, syne;">{session_select} Summary</h6>''',unsafe_allow_html=True)
        select_choice = st.selectbox('Data Summary', ['Summarise?','Q1 Grid-Positions', 'Q2 Grid-Positions', 'Q3 Grid-Positions', 'Knocked Out', 'Final Grid-Positions'])
            

        if not select_choice == 'Summarise?':
            
            if select_choice == 'Knocked Out':

                # latice of if-else structure 
                knocked_q1 = summarised_session('Qualifying',summarised_results, 'Knocked', round_number='Q1')
                knocked_q2 = summarised_session('Qualifying',summarised_results, 'Knocked', round_number='Q2')
                knocked_q3 = summarised_session('Qualifying',summarised_results, 'Knocked', round_number='Q3')

                
                tq2 = copy.deepcopy(knocked_q2)
                tq3 = copy.deepcopy(knocked_q3)

                # Repeating knockout drivers 
                if not knocked_q1 == None:
                    tq2['driver_data'] = dict(pd.DataFrame(tq2['driver_data']).set_index('DriverNumber').drop(knocked_q1['driver_data']['DriverNumber'].to_list(),axis=0).reset_index())
                    tq3['driver_data'] = dict(pd.DataFrame(tq3['driver_data']).set_index('DriverNumber').drop(knocked_q1['driver_data']['DriverNumber'].to_list(),axis=0).reset_index())
                
                tq3['driver_data'] = dict(pd.DataFrame(tq3['driver_data']).set_index('DriverNumber').drop(tq2['driver_data']['DriverNumber'].to_list(),axis=0).reset_index())
                

                if knocked_q1 == None:
                    st.markdown(f'''> <h5 style="font-family:formula1, syne;">Everyone Qualified Q1</h5>''',unsafe_allow_html=True)
                else:
                    st.markdown(f'''> <h5 style="font-family:formula1, syne;"><u>Q1 Knockouts</u></h5>''',unsafe_allow_html=True)
                    display_qualifying_summary(knocked_q1, mode='Knocked')
                
                
                
                if knocked_q2 == None:
                    st.markdown(f'''<h5 style="font-family:formula1, syne;">Everyone Qualified Q2</h5>''',unsafe_allow_html=True)
                else:
                    st.markdown(f'''> <h5 style="font-family:formula1, syne;"><u>Q2 Knockouts</u></h5>''',unsafe_allow_html=True)
                    display_qualifying_summary(tq2, mode='Knocked')
                    
                

                if knocked_q3 == None:
                    st.markdown(f'''<h5 style="font-family:formula1, syne;">Everyone Qualified Q3</h5>''',unsafe_allow_html=True)
                else:
                    st.markdown(f'''> <h5 style="font-family:formula1, syne;"><u>Q3 Knockouts</u></h5>''',unsafe_allow_html=True)
                    display_qualifying_summary(tq3,mode='Knocked')
                    

            elif select_choice == 'Final Grid-Positions':

                
                full_grid = summarised_session('Qualifying',summarised_results, 'Final Grid-Positions')
                display_qualifying_summary(full_grid, mode='Final Grid-Positions')

            elif select_choice == 'Q1 Grid-Positions':

                q1_grid = summarised_session('Qualifying',summarised_results, 'Q1 Grid-Positions', round_number='Q1')
                display_qualifying_summary(q1_grid, mode='Q1 Grid-Positions')
            
            elif select_choice == 'Q2 Grid-Positions':

                q1_grid = summarised_session('Qualifying',summarised_results, 'Q2 Grid-Positions', round_number='Q2')
                display_qualifying_summary(q1_grid, mode='Q2 Grid-Positions')

            elif select_choice == 'Q3 Grid-Positions':

                q1_grid = summarised_session('Qualifying',summarised_results, 'Q3 Grid-Positions', round_number='Q3')
                display_qualifying_summary(q1_grid, mode='Q3 Grid-Positions')

    else: 
        st.markdown(f'''<h6 style="font-family:formula1, syne;">{session_select} Comprehensive Analysis</h6>''',unsafe_allow_html=True)

        cols = st.columns([6,3])
        placeholder = cols[1].empty()
        analysis_type = cols[0].selectbox('Select to Investigate', ['Analysis Type?','Driver Performance Analysis', 'Team Performance Analysis'],key='key-analysis')
        placeholder.selectbox('?',[])

        if year >= 2018:

            if analysis_type == 'Driver Performance Analysis':

                delta_required = [ 'LapTime',
                                    'Sector1Time',
                                    'Sector2Time',
                                    'Sector3Time',   ]  
                
                # overwrite placeholder selectbox

                
                analysis_mode = placeholder.selectbox('Type of Analysis', ['Individual','Comparative'])
                
                    
                if analysis_mode == 'Individual':

                    driver = st.selectbox('Choose Driver', driver_dict.keys())
                    fastest_driver = list(driver_dict.keys())[0]

                    st.markdown('***')
                    st.markdown(f'''<h6 style="font-family:formula1, syne;"><u> Driver Performance Investigation</u></h6>''',unsafe_allow_html=True)
                    AB, BN, TN, TC  = driver_dict[driver]
                    st.markdown(f'''<h6 style="font-family:formula1, syne;">Fastest Lap Analysis</h6>''',unsafe_allow_html=True)
                    st.markdown(f'''<h4 style="font-family:formula1, syne;">{BN} ({AB})<sub style='color:#{TC}'>{TN}</sub></h4>''',unsafe_allow_html=True)
            
                    
                    # session data 
                    session = return_session_object(year, event, session_select)

                    # laps
                    laps = session.laps.reset_index(drop=True)

                    # weather data
                    weather_data = session.laps.get_weather_data()
                    weather_data = weather_data.reset_index(drop=True)

                    # club data
                    joined = pd.concat([laps, weather_data.loc[:, ~(weather_data.columns == 'Time')]], axis=1) #from the fastf1 documentation
                    # compounds
                    driver_data = joined.pick_driver(AB)

                    


                    if driver_data.empty or pd.isnull(driver_data.pick_fastest()).all():

                        st.error(f'''No Time recorded, is either Knocked out or Data is Invalidated.''')

                    else:
                        
                        compounds_used = list(driver_data['Compound'].dropna().unique())
                        
                        # Fastest Lap
                        driver_data = joined.pick_driver(AB).pick_fastest()

                        

                        compound = driver_data['Compound'].lower()
            
                    
                        # fastest time in that session
                        fastest = joined.pick_fastest()
                        fcompound = fastest['Compound'].lower() 
                        # st.write(fastest)



                        # Lap and Sector Times
                        # positive and negative delta
                        delta_dict = {}
                        for item in delta_required: 
                            delta_dict[item] = delta_variation(driver_data[item], fastest[item])
                        
                        # st.write(delta_dict)                                         
                        
                        if driver_data['IsPersonalBest']:
                            st.markdown(f'''> <h5 style="font-family:formula1, syne; color:purple;">Lap Time - {timedelta_conversion(driver_data['LapTime'])} <sub style='color:black;'>Personal Best</sub></h5>''',unsafe_allow_html=True)
                        else:
                            st.markdown(f'''> <h5 style="font-family:formula1, syne; color:purple;">Lap Time - {timedelta_conversion(driver_data['LapTime'])}</h5>''',unsafe_allow_html=True)
                            
                                                        
                        st.markdown(f'''<h6 style="font-family:formula1, syne;"><u>Sector Times</u></h6>''',unsafe_allow_html=True)
                        for i,item in enumerate(delta_required[1:],1):
                            st.markdown(f'''<h6 style="font-family:formula1, syne; ">S{i} - {timedelta_conversion(driver_data[item])} <sub>{delta_dict[item]} ({fastest['Driver']})</sub></h6>''',unsafe_allow_html=True)

                        st.markdown('***')
                        # compound 
                        st.markdown(f'''<h6 style="font-family:formula1, syne"><u>Compounds Used</u></h6>''',unsafe_allow_html=True)                                                                                
                        st.markdown(f'''<h6 style="font-family:formula1, syne; "><img src='data:image/png;base64,{img_to_bytes(f'./assets/{compound}.png')}' class='img-fluid' width=50 ><sub style='padding-left:10px;'>{compound} Compound, </sub> Tyre Life <span style='font-size:28px'>{driver_data['TyreLife']}</span> Laps</h6>''',unsafe_allow_html=True)
                        st.markdown('***')
                        # Speed Traps
                        st.markdown(f'''<h6 style="font-family:formula1, syne;"><u>Speed Traps</u></h6>''',unsafe_allow_html=True)
                        st.markdown(f'''<h6 style="font-family:formula1, syne; ">Sector1 Speed - <span style='font-size:28px'>{driver_data['SpeedI1']}<sup>km/h</sup></span> <sub>{fastest['SpeedI1']} km/h ({fastest['Driver']}) <img src='data:image/png;base64,{img_to_bytes(f'./assets/{fcompound}.png')}' class='img-fluid' width=40></sub></h6>''',unsafe_allow_html=True)
                        st.markdown(f'''<h6 style="font-family:formula1, syne; ">Sector2 Speed - <span style='font-size:28px'>{driver_data['SpeedI2']} <sup>km/h</sup></span> <sub>{fastest['SpeedI2']} km/h ({fastest['Driver']}) <img src='data:image/png;base64,{img_to_bytes(f'./assets/{fcompound}.png')}' class='img-fluid' width=40></sub></h6>''',unsafe_allow_html=True)
                        st.markdown(f'''<h6 style="font-family:formula1, syne; ">Finish Line Speed - <span style='font-size:28px'>{driver_data['SpeedFL']} <sup>km/h</sup></span> <sub>{fastest['SpeedFL']} km/h ({fastest['Driver']}) <img src='data:image/png;base64,{img_to_bytes(f'./assets/{fcompound}.png')}' class='img-fluid' width=40></sub></h6>''',unsafe_allow_html=True)
                        st.markdown(f'''<h6 style="font-family:formula1, syne; ">Longest Straight Speed - <span style='font-size:28px'>{driver_data['SpeedST']} <sup>km/h</sup></span> <sub>{fastest['SpeedST']} km/h ({fastest['Driver']}) <img src='data:image/png;base64,{img_to_bytes(f'./assets/{fcompound}.png')}' class='img-fluid' width=40></sub></h6>''',unsafe_allow_html=True)

                        st.markdown('***')
                        
                        st.markdown(f'''<h6 style="font-family:formula1, syne;"><u>Weather Data</u></h6>''',unsafe_allow_html=True) 
                        st.markdown(f'''<h6 style="font-family:formula1, syne;">Air Temperature - {driver_data['AirTemp']} °C</h6>''',unsafe_allow_html=True)            
                        st.markdown(f'''<h6 style="font-family:formula1, syne;">Track Temperature - {driver_data['TrackTemp']} °C</h6>''',unsafe_allow_html=True)     
                        st.markdown(f'''<h6 style="font-family:formula1, syne;">Humidity - {driver_data['Humidity']}</h6>''',unsafe_allow_html=True)       
                        st.markdown(f'''<h6 style="font-family:formula1, syne;">Pressure - {driver_data['Pressure']}</h6>''',unsafe_allow_html=True)                                                                                                                                                     
                        st.markdown(f'''<h6 style="font-family:formula1, syne;">Wind Direction - {driver_data['WindDirection']}</h6>''',unsafe_allow_html=True)
                        st.markdown(f'''<h6 style="font-family:formula1, syne;">Wind Speed - {driver_data['WindSpeed']}</h6>''',unsafe_allow_html=True)
                        if driver_data['Rainfall']:
                            st.markdown(f'''<h6 style="font-family:formula1, syne;"> Rainfall - <img src='data:image/png;base64, {img_to_bytes('./assets/rain.png')}' class='img-fluid', width=35> </h6>''',unsafe_allow_html=True)
                        else:
                            st.markdown(f'''<h6 style="font-family:formula1, syne;"> Rainfall - <img src='data:image/png;base64, {img_to_bytes('./assets/no-rain.png')}' class='img-fluid', width=35> </h6>''',unsafe_allow_html=True)

                        
                        st.markdown('***')

                        st.markdown(f'''<h6 style="font-family:formula1, syne;"><u>Speed Chart</u></h6>''',unsafe_allow_html=True)
                        st.markdown(f'''<h6 style="font-family:formula1, syne;">Speed Vs Distance Visualisation , <br> {driver} - {event} {year}</h6>''',unsafe_allow_html=True)
                        
                        if driver == fastest_driver:
                            driver_lap = session.laps.pick_driver(AB).pick_fastest()
                            driver_tel = driver_lap.get_car_data().add_distance()
                            driver_color = '#'+TC
                            x = driver_tel['Distance']
                            y = driver_tel['Speed']
                            context = f'{AB} Speed'
                            package = [(x, y, driver_color, AB)]
                            fig = speed_visualisation(package)
                            st.plotly_chart(fig)
                        
                        else:
                            driver_lap = session.laps.pick_driver(AB).pick_fastest()
                            driver_tel = driver_lap.get_car_data().add_distance()
                            try:
                                driver_color = '#'+TC
                            except:
                                driver_color = 'gold'
                            x1 = driver_tel['Distance']
                            y1 = driver_tel['Speed']

                            fdriver_lap = session.laps.pick_fastest()
                            fdriver_tel = fdriver_lap.get_car_data().add_distance()
                            fdriver_color = fastf1.plotting.team_color(fdriver_lap['Team'])
                            fdriver_AB = fdriver_lap['Driver']
                            x2 = fdriver_tel['Distance']
                            y2 = fdriver_tel['Speed']

                            package = [(x1, y1, driver_color, AB),(x2, y2, fdriver_color, fdriver_AB)]


                            fig = speed_visualisation(package)


                            context = f'{AB} Speed'
                            st.plotly_chart(fig)


                        st.markdown(f'''<h6 style="font-family:formula1, syne;"><u>Gear Shifts</u></h6>''',unsafe_allow_html=True)

                        try:
                            lap = session.laps.pick_driver(AB).pick_fastest()
                            tel = lap.get_telemetry()
                            x = np.array(tel['X'].values)
                            y = np.array(tel['Y'].values)   
                            st.markdown(f'''<h6 style="font-family:formula1, syne;">Lap Gear Shift Visualization, <br> {driver} - {event} {current_year}</h6>''',unsafe_allow_html=True)
                    
                            gear_heatmap(x,y,tel,driver,event,current_year)
                        
                            
                        except:
                            st.warning("Data Descrepancy!")

                        
                else:
                    pass
                    # COMPARATIVE
                    
                    
                    

        else:

            st.warning("The API doesn't hold the telemetry data for the years before 2018.")
            

                                            

                                        



    

    


def display_qualifying_summary(data, mode):

    # st.write(data)
    round_dict = {'Q1':'Q1', 'Q2':'Q1','Q3':'Q2'}
    # round_number = data['driver_data']

    # assigning data
    driver_numbers = data['driver_data']['DriverNumber']
    broadcast_names = data['driver_data']['BroadcastName']
    abbreviations = data['driver_data']['Abbreviation']
    team_names = data['driver_data']['TeamName']
    team_colors = data['driver_data']['TeamColor']
    full_names = data['driver_data']['FullName']
    positions = data['driver_data']['Position']

    if mode == 'Knocked':
        # displaying data -- knocked
        Q_times = data['driver_data'][round_dict[data['round_number']]]
        for dn, bn, ab, tn, tc, p, qt in zip(driver_numbers, broadcast_names, abbreviations, team_names, team_colors, positions, Q_times):

            cols = st.columns([4,2])
            cols[0].markdown(f'''<h5 style="font-family:formula1, syne;"><span style="font-size:37px; background-color: #{tc}; border: 2px solid black; color:black; ">{dn}</span> {bn} ({ab}) <sub style=''>{tn}</sub></h5>''',unsafe_allow_html=True)
            cols[1].markdown(f'''<h6 style="font-family:formula1, syne;">P{int(p)} - {timedelta_conversion(qt)}</h6>''',unsafe_allow_html=True)

    elif mode in ['Final Grid-Positions', 'Q1 Grid-Positions', 'Q2 Grid-Positions', 'Q3 Grid-Positions']:

        # st.write(data)
        Q_times = data['Times']
        
        st.markdown('***')
        st.markdown(f'''<center><h5 style="font-family:formula1, syne;">{mode} <img src='data:image/png;base64,{img_to_bytes('./assets/grid.png')}' class='img-fluid' width=35> </h5></center>''',unsafe_allow_html=True)
        st.markdown('***')
        st.markdown('')
        cols = st.columns([5,5])
        len_drivers = np.arange(len(driver_numbers))
        
        for x, dn, bn, ab, tn, tc, p, qt in zip(len_drivers,driver_numbers, broadcast_names, abbreviations, team_names, team_colors, positions, Q_times):
            
            if x%2==0:
                if x == 0:
                    cols[0].markdown(f'''<center><h5 style="font-family:formula1, syne; border: 2px solid black;"> {dn} <u>{bn}</u> <span style='font-size:16px;'>({ab})</span> <p style='font-size:16px; font-weight:bold; color: #{tc}; background-color:slategray;'>{tn}</p> <p style='background-color:purple; color:white;'>{timedelta_conversion(qt)}</p> </h5></center>''',unsafe_allow_html=True)
                    cols[0].markdown(f'''<h6 style="font-family:formula1, syne;">P{int(p)}</h6>''',unsafe_allow_html=True)
                    cols[1].markdown('')
                    cols[1].markdown('')
                    cols[1].markdown('')
                    cols[1].markdown('')
                else:
                    cols[0].markdown(f'''<center><h5 style="font-family:formula1, syne; border: 2px solid black;"> {dn} <u>{bn}</u> <span style='font-size:16px;'>({ab})</span> <p style='font-size:16px; font-weight:bold; color: #{tc}; background-color:slategray;'>{tn}</p> <p>{timedelta_conversion(qt)}</p> </h5></center>''',unsafe_allow_html=True)
                    cols[0].markdown(f'''<h6 style="font-family:formula1, syne;">P{int(p)}</h6>''',unsafe_allow_html=True)
                    cols[1].markdown('')
                    cols[1].markdown('')
                    cols[1].markdown('')

             
            else:
                cols[0].markdown('')
                cols[0].markdown('')
                cols[0].markdown('')
                cols[1].markdown(f'''<center><h5 style="font-family:formula1, syne; border: 2px solid black;"> {dn} <u>{bn}</u> <span style='font-size:16px;'>({ab})</span> <p style='font-size:16px; font-weight:bold; color: #{tc}; background-color:slategray;'>{tn}</p> <p>{timedelta_conversion(qt)}</p> </h5></center>''',unsafe_allow_html=True)
                cols[1].markdown(f'''<h6 style="font-family:formula1, syne;">P{int(p)}</h6>''',unsafe_allow_html=True)
                

        st.markdown('***')


            
            
            

if __name__ == '__main__':

    # page configuration
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.set_page_config(page_title="F1-Web-Paddock",layout="wide",initial_sidebar_state="expanded",)

    # font-base64
    with open('./assets/formula1-regular-base64.txt', 'r') as f:
        regular = f.read()

    with open('./assets/formula1-bold-base64.txt', 'r') as f:
        bold = f.read()

    with open('./assets/formula1-black-base64.txt', 'r') as f:
        black = f.read()


    # font-style
    font_url = '''
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Syne:wght@700&display=swap" rel="stylesheet">

    '''
    st.markdown(f"{font_url}",unsafe_allow_html=True)

    # fonts-style preset
    style = '''
    <style>
    @font-face {
    font-family: 'formula1';'''+f'''
    src: url(data:application/x-font-woff;charset=utf-8;base64,{regular} )'''+''' format('woff');
    font-weight: normal;
    font-style: normal;

    }

    html, body, [class*="css"]  {
    font-family: 'formula1';
    }
    

    @font-face {
    font-family: 'formula1';'''+f'''
    src: url(data:application/x-font-woff;charset=utf-8;base64,{bold} )'''+''' format('woff');
    font-weight: normal;
    font-style: normal;
    }

    @font-face {
    font-family: 'formula1';'''+f'''
    src: url(data:application/x-font-woff;charset=utf-8;base64,{black} )'''+''' format('woff');
    font-weight: normal;
    font-style: normal;

    }

    </style>
    '''
    st.markdown(f'{style}',unsafe_allow_html=True)

    

    # Sidebar title 
    st.sidebar.markdown(f'''<h2 style="font-family:formula1, syne;font-weight:bold; font-size:27px;">The Control Deck <img src='data:image/png;base64,{img_to_bytes('./assets/steering-1.png')}' class='img-fluid' width=50 ></h2>''',unsafe_allow_html=True)
    st.sidebar.markdown('***')


    # title 
    st.markdown(f'''<center><h1 style='font-family:formula1, syne; font-size:40px;'>The Formula-1 Web-Paddock <img src='data:image/png;base64,{img_to_bytes('./assets/f1-car.png')}' class='img-fluid' width=80></h1></center>''',unsafe_allow_html=True)
    st.markdown('***')


    # Fetch Dependables
    api_elements = instantiate_API_keys()
    yearwise_rounds = load_miscellaneous_data()

    # st.write(yearwise_rounds.loc[1950,'Rounds'])


    # Categories -- Change Order to, About, Current Season, Previous Season, The F1 Glossary
    # category = st.sidebar.selectbox('Select', ['Current Season', 'Previous Seasons', 'About', 'The F1 Glossary'])
    category = st.sidebar.selectbox('Select Timeline', ['Current Season', 'Previous Seasons'])
    

    if category == 'Previous Seasons':
        st.sidebar.markdown('***')

        # YEAR -- input
        
        st.sidebar.markdown(f'''<p style='font-weight:bold;'><u>Parameters</u></p>''',unsafe_allow_html=True)
        year = st.sidebar.slider('Select Year', min_value=2006, max_value=2021,value=2021)

        # fetch circuits
        circuits_cdf, circuits_rdf = fetch_circuits_data(year)
            
        # collecting data
        event_names, event_schedule = fetch_event_schedule(int(year))
        

        

        radios = st.sidebar.radio('Data', ['Schedule', 'Grand Prix Analysis'])


        if radios == 'Schedule':

            st.markdown(f'''<p style="font-size:30px; font-weight:bold; font-family:formula1, syne;"> <span style='color:darkblue;'>{year}</span> Race Schedule | Grand Prix List </p>''',unsafe_allow_html=True)
            st.markdown('***')

            # Events for the year 
            st.markdown(f'''<p style='font-weight:bold;'></p>''',unsafe_allow_html=True)
            display_schedule(year, circuits_cdf, circuits_rdf)


        else:

            # Events for the year 
            st.sidebar.markdown(f'''<p style='font-weight:bold;'>Grand Prix List | <span style='color:darkblue;'>{year}</span></p>''',unsafe_allow_html=True)

            
            # EVENT NAME
            event = st.sidebar.selectbox('Select Event', event_names)

            if not event == 'List of GPs':
                
                # slicing data 
                event_data = event_schedule[event_schedule['EventName'] == event].T

                # packaging event summarised-information
                package = {}
                items = event_schedule.columns[:-1]
                for item in items:
                    package[item] = event_data.loc[item].values[0]

                sessions_list = [package[x] for x in ['Session'+str(i) for i in range(1,6)]]
                # st.write(package)

                summary_type = st.sidebar.radio('Select Category', ['Session Analysis', 'Weekend Analysis'])

                if summary_type == 'Weekend Analysis':

                    # title
                    st.markdown(f'''<center><h4 style="font-family:formula1, syne;">{summary_type}</h4></center>''',unsafe_allow_html=True)
                    st.markdown('***')

                    # Grand Prix Title
                    st.markdown(f'''<p style="font-size:30px; font-weight:bold; font-family:formula1, syne;"> <img src="https://countryflagsapi.com/png/{package["Country"]}" width="50">  <u>{package["EventName"]}</u>  |  <span style="font-size:23px;">{date_modifier(package["EventDate"])}</span></p>''',unsafe_allow_html=True)
                
                
                    # Race Type
                    st.markdown(f'<p style="font-size:22px;font-family:formula1, syne;">Race Format - {package["EventFormat"].capitalize()}</p>',unsafe_allow_html=True)

                    # Sessions
                    cols = st.columns(len(sessions_list))
                    for i, session_name in enumerate(sessions_list, start=0) :
                        cols[i].markdown('> <p style="font-size:17px; font-weight:bold; font-family:formula1, syne;"><u>{}</u><p>'.format(session_name),unsafe_allow_html=True)
                        cols[i].markdown('> <p style="font-size:13px; font-weight:bold; font-family:formula1, syne;">{}<p>'.format(date_modifier(package['Session'+str(i+1)+"Date"])),unsafe_allow_html=True)
                    
                    st.markdown('***')


                else:
                    # title
                    st.markdown(f'''<center><h4 style="font-family:formula1, syne;">{summary_type}</h4></center>''',unsafe_allow_html=True)
                    st.markdown('***')

                    # Grand Prix Title
                    st.markdown(f'''<p style="font-size:28px; font-weight:bold; font-family:formula1, syne;"> <img src="https://countryflagsapi.com/png/{package["Country"]}" width="50">  <u>{package["EventName"]}</u>  |  <span style="font-size:23px;">{date_modifier(package["EventDate"])}</span></p>''',unsafe_allow_html=True)
                


                    st.markdown(f'<p style="font-size:15px;font-family:formula1, syne; font-weight:bold;">Race Format - {package["EventFormat"].capitalize()}</p>',unsafe_allow_html=True)
                    sessions_list.insert(0,'Select Session')
                    session_select = st.selectbox('Select Session', sessions_list, key='sessions')
                    st.markdown('***')
                    
                    # bypassing the first element problem
                    if session_select != 'Select Session':

                        session_results = load_session_data(year, event, session_select)
                        
                        

                    if session_select == 'Qualifying':

                        # Data Collection
                        summarised_results = session_results.copy(deep=True)
                        summarised_results = pd.DataFrame(summarised_results)
                        summarised_results = summarised_results.drop(['Time','Status','Points'], axis=1)
                        summarised_results = summarised_results.fillna('0')

                        qualifying(summarised_results, year)


                    elif session_select in ['Practice 1','Practice 2','Practice 3','Sprint']:
    
                        st.warning('In Development ⌛')
                        st.markdown('***')

                       
                            


    
    elif category == 'Current Season':
        st.sidebar.markdown('***')

        current_year = datetime.now().strftime('%Y')
        st.markdown(f'''<p style="font-size:30px; font-weight:bold; font-family:formula1, syne;"> <span style='color:darkblue;'>{current_year}</span> Race Schedule | Grand Prix List </p>''',unsafe_allow_html=True)
        st.markdown("***")

        circuits_cdf, circuits_rdf = fetch_circuits_data(current_year)

        radios = st.sidebar.radio('Data',['Schedule','Grand Prix Analysis'], key='current')

        if radios == 'Schedule':
            # Events for the year 
            st.markdown(f'''<p style='font-weight:bold;'></p>''',unsafe_allow_html=True)
            display_schedule(int(current_year), circuits_cdf, circuits_rdf)

        else:
            # GP analysis
    
            # Analytics for the races that happened
            current_event = fastf1.get_event_schedule(int(current_year))
            conditional = (current_event['EventDate'] <= datetime.now()) | (current_event['Session1Date'] <= datetime.now()) | (current_event['Session2Date'] <= datetime.now()) | (current_event['Session3Date'] <= datetime.now()) | (current_event['Session4Date'] <= datetime.now()) | (current_event['Session4Date'] <= datetime.now())
            index = current_event[conditional].index
            current_event = current_event.loc[index,:]

            event_names = current_event['EventName'].to_list()
            event_names.insert(0, "List of Completed Grand Prixs'")

            
            event = st.sidebar.selectbox('Select Event', event_names)

            if not event == "List of Completed Grand Prixs'":

                event_data = current_event[current_event['EventName'] == event].T

                # packaging event summarised-information
                package = {}
                items = current_event.columns[:-1]
                for item in items:
                    package[item] = event_data.loc[item].values[0]

                sessions_list = [package[x] for x in ['Session'+str(i) for i in range(1,6)]]

                st.markdown(f'''<center><h4 style="font-family:formula1, syne;">Session Analysis</h4></center>''',unsafe_allow_html=True)
                st.markdown('***')

                # Grand Prix Title
                st.markdown(f'''<p style="font-size:28px; font-weight:bold; font-family:formula1, syne;"> <img src="https://countryflagsapi.com/png/{package["Country"]}" width="50">  <u>{package["EventName"]}</u>  |  <span style="font-size:23px;">{date_modifier(package["EventDate"])}</span></p>''',unsafe_allow_html=True)
                st.markdown(f'<p style="font-size:15px;font-family:formula1, syne; font-weight:bold;">Race Format - {package["EventFormat"].capitalize()}</p>',unsafe_allow_html=True)

                # session select
                sessions_list.insert(0,'Select Session')
                session_select = st.selectbox('Select Session', sessions_list, key='sessions')
                st.markdown('***')


                # Bypassing the first element problem
                if session_select != 'Select Session':
                    
                    if session_select == 'Qualifying':

                        session_results = load_session_data(int(current_year), event, session_select)

                        # Data Collection
                        summarised_results = session_results.copy(deep=True)
                        summarised_results = pd.DataFrame(summarised_results)
                        summarised_results = summarised_results.drop(['Time','Status','Points'], axis=1)
                        summarised_results = summarised_results.fillna('0')

                        
                        qualifying(summarised_results,int(current_year))

            
                    elif session_select in ['Practice 1','Practice 2','Practice 3','Sprint']:

                        st.warning('In Development ⌛')
                        st.markdown('***')


                


    elif category == 'About':
        about_cs()






    st.sidebar.markdown('***')
    
    st.sidebar.markdown(f'''<h4 style='font-family:formula1, syne;'><span style='font-weight:normal;'>Engineered by</span> <u style='font-size:25px'>r0han</u> <a href='https://github.com/r0han99'> <img src='data:image/png;base64,{img_to_bytes('./assets/github.png')}' class='img-fluid' width=35> </a></h4>''',unsafe_allow_html=True)
    
