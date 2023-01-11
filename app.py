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
from src.about import about_cs
from src.carimages import fetch_carimgs
from pathlib import Path
import base64
from dateutil import parser
import plotly.graph_objects as go
import copy
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
import seaborn as sns
from src.attributions import attribute
import pycountry
from itertools import cycle

import pickle


# fastf1.Cache.enable_cache('./cache')  

missing_endpoints = {'Abu Dhabi':"United Arab Emirates", 'UAE':"United Arab Emirates"}


def load_model():
    reconstructed_model = load_model("data_mining_LSTM.h5")
    return reconstructed_model


# default subroutines
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

@st.cache(persist=True)
def load_carspecs():
    with open('./data/CAR_SPECIFICATIONS_v3.pickle', 'rb') as f:
        return pickle.load(f)

def fabricate_dict(dictionary):
    if dictionary == []:
        return "No Data"
    else:
        manifactured = {}
        for string in dictionary:

            if len(string.split(':')) == 1:
                try:
                    key, value = string.split(';')
                except:
                    key = string.split(':')[0]
                    value = 'No Data'
            
            elif len(string.split(':'))>2:
                ex = string.split(':')
                l = [ex[0]]
                l.append(','.join(ex[1:]))
                key, value = l
            else:
                key, value = string.split(':')
            
            manifactured[key] = value.strip()

        return manifactured

def load_carimage_data():
    df = pd.read_csv('./data/FINAL-CAR-IMAGES_2012-2022.csv')
    return df


def load_miscellaneous_data():

    rounds = pd.read_csv('./data/year_wise_rounds.csv').set_index('Unnamed: 0')
    
    
    return rounds


@st.cache(persist=True,suppress_st_warning=True)
def load_rounds():
    rounds = pd.read_csv('./data/year_wise_rounds.csv')
    rounds.columns = ['year','rounds']  
    years = rounds['year'].to_list()
    round_v = rounds['rounds'].to_list()
    rounds = {}
    for year, r in zip(years, round_v):
        rounds[year] = r
        

    return rounds

def instantiate_API_keys():


    API_elements = {

            'drivers_wr': 'http://ergast.com/api/f1/{}/{}/drivers.json',
            'drivers_wor': 'http://ergast.com/api/f1/{}/drivers.json'

    }

    return API_elements


def fetch_position_rank(const=None, driver=None,team=None,year=2022, individual=False):
    if not individual:
        pos = const.loc[year].loc[rounds[year]].reset_index()
        position = pos[pos['Constructor']==team]['position'].values[0]
        points = pos[pos['Constructor']==team]['points'].values[0]
        
        return position, points
    else:
        pos = driver.loc[year].loc[rounds[year]].reset_index()
        dr_standings = {}
        for dr in pos[pos['Constructors'] == team].values:
            pos, name, _, code, team, points, wins = dr
            dr_standings[name] = [(pos, code, points, wins)]
        return dr_standings


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
        



@st.cache(persist=True, show_spinner=True)
def fetch_constructorStandings(range_list=[2014,2023], roundwise=False,rounds=None,verbose=False):
    
    if roundwise == False:
        dfs = []
        years = list(range(*range_list))
        for year in range(*range_list):
            url = f'http://ergast.com/api/f1/{year}/constructorStandings.json'
            response = requests.get(url)
            constructor_standings = response.json()

            itemlist = constructor_standings['MRData']['StandingsTable']['StandingsLists'][0]['ConstructorStandings']
            teams = []
            for item in itemlist:
                teams.append(item['Constructor']['name'])

            constructor_data = pd.DataFrame(itemlist)
            constructor_data['Constructor'] = teams
            constructor_data = constructor_data.set_index('position',drop=True)
            constructor_data = constructor_data.drop('positionText',axis=1)

            dfs.append(constructor_data)

        return pd.concat(dfs, keys=years)
    elif rounds != None and roundwise==True:

        # roundwise constructors
        dfs_y = {}
        for year in range(*range_list):
            dfs_r = []
            for r in range(1,rounds[year]+1):
                if verbose:
                    print(f'Constructors: fetching {year}, round:{r}')
                url = f'http://ergast.com/api/f1/{year}/{r}/constructorStandings.json'
                res = requests.get(url)
                res = res.json()
                item_list = res['MRData']['StandingsTable']['StandingsLists'][0]['ConstructorStandings']

                teams = []
                for item in item_list:
                    teams.append(item['Constructor']['name'])

                constructor_data = pd.DataFrame(item_list)
                constructor_data['Constructor'] = teams
                constructor_data = constructor_data.set_index('position',drop=True)
                constructor_data = constructor_data.drop('positionText',axis=1)

                dfs_r.append(constructor_data)

            dfs_y[year] = dfs_r

        year_wise_data = []
        for key in zip(dfs_y.keys()):
        #     print(key[0])
            year_wise_data.append(pd.concat(dfs_y[key[0]], keys=range(1,rounds[year]+1)))

        const = pd.concat(year_wise_data, keys=range(*range_list))
        return const
        
        
@st.cache(persist=True, show_spinner=True)
def fetch_driverstandings(range_list=[2014,2023], rounds=None, roundwise=False,verbose=False):
    
    
    if roundwise == False:
        dfs = []
        years = list(range(*range_list))

        for year in range(*range_list):
            url = f'http://ergast.com/api/f1/{year}/driverStandings.json'
            response = requests.get(url)
            drivers_standings = response.json()

            item_list = drivers_standings['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings']

            teams = []
            for item in item_list:
                teams.append(item['Constructors'][0]['name'])

            driver_dict_list = []
            for item in item_list:
                driver_dict_list.append(item['Driver'])

            ds = pd.DataFrame(item_list)
            ds['Constructors'] = teams
            ds = pd.concat([ds,pd.DataFrame(driver_dict_list)],axis=1)
            ds['fullname'] = ds['givenName'] + ' ' + ds['familyName']
            ds = ds.drop(['positionText','Driver','driverId','url','dateOfBirth','givenName','familyName','nationality'],axis=1)
            ds = ds.set_index('position',drop=True)
            col_order = ['fullname','permanentNumber','code','Constructors','points','wins']
            dfs.append(ds[col_order])
        return pd.concat(dfs,keys=years)
    
    elif roundwise==True and rounds != None:
        # Drivers Round Wise
        dfs_y = {}
        for year in range(*[2014,2023]):
            dfs_r = []
            for r in range(1,rounds[year]+1):
                print(f'fetching {year}, round:{r}')
                url = f'http://ergast.com/api/f1/{year}/{r}/driverStandings.json'
                res = requests.get(url)
                res = res.json()
                item_list = res['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings']

                teams = []
                for item in item_list:
                    teams.append(item['Constructors'][0]['name'])

                driver_dict_list = []
                for item in item_list:
                    driver_dict_list.append(item['Driver'])

                ds = pd.DataFrame(item_list)
                ds['Constructors'] = teams
                ds = pd.concat([ds,pd.DataFrame(driver_dict_list)],axis=1)
                ds['fullname'] = ds['givenName'] + ' ' + ds['familyName']
                ds = ds.drop(['positionText','Driver','driverId','url','dateOfBirth','givenName','familyName','nationality'],axis=1)
                ds = ds.set_index('position',drop=True)
                col_order = ['fullname','permanentNumber','code','Constructors','points','wins']
                ds = ds[col_order]
                dfs_r.append(ds)

            dfs_y[year] = dfs_r

        year_wise_data = []
        for key in zip(dfs_y.keys()):
        #     print(key[0])
            year_wise_data.append(pd.concat(dfs_y[key[0]], keys=range(1,rounds[year]+1)))

        drivers = pd.concat(year_wise_data, keys=range(*range_list))

        return drivers

            
    



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
        return session_results, session_obj
    
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
    event_names.insert(0, "List of Grand Prix's")

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
        
        
        try:
            flag = pycountry.countries.search_fuzzy(package["Country"].lower())[0].flag
        except:
            flag = pycountry.countries.search_fuzzy(missing_endpoints[package["Country"]])[0].flag


        if current_date > package['EventDate']:
            
            st.markdown(f'''<p style="font-size:30px; font-weight:bold; font-family:formula1, syne;"> {flag}  <u>{package["EventName"]}</u>  |  <span style="font-size:23px;">{date_modifier(package["EventDate"])} <br>{circuit}, {locality} <img src='data:image/png;base64,{img_to_bytes('./assets/checkered-flag.png')}' class='img-fluid' width=50 ></span> </p>''',unsafe_allow_html=True)
        
        else:
            st.markdown(f'''<p style="font-size:30px; font-weight:bold; font-family:formula1, syne;"> {flag}  <u>{package["EventName"]}</u>  |  <span style="font-size:23px;">{date_modifier(package["EventDate"])} <br>{circuit}, {locality} </span></p>''',unsafe_allow_html=True)


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
        return '-{}'.format(str(negative_delta).split('.')[-1:][0][:4])
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


def speed_visualisation(package, mode):


    if len(package) == 2:
        iterations = 2

    else:
        iterations = 1

    fig = go.Figure()
    for i in range(iterations):

        x, y, color, AB = package[i]

        if mode == 'different':
            fig.add_trace(go.Scatter(x=x, y=y,
                                    line = dict(color=color),
                                mode='lines',
                                name=f'{AB} Speed'))
        else:
            
            fig.add_trace(go.Scatter(x=x, y=y,
                                mode='lines',
                                name=f'{AB} Speed'))



    fig.update_layout(paper_bgcolor="#e5e9f0", template='seaborn', showlegend=True)


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


def bump_plot(driver, team=None, mode='overall', year=2022, team_color=None ):
    
    if mode == 'team':
        if team_color == None:
            team_color = 'orange'
        plt.style.use('seaborn-darkgrid')
        test = driver.loc[year].reset_index()
        plt.style.use('seaborn-whitegrid')
        plt.figure(figsize=(11,8))
        sns.lineplot(data=test[test['Constructors']==team], x='level_0',y='position',
                     style='code',
                     markers=True,linewidth=3, color=team_color )
        plt.xticks(ticks=list(range(1,rounds[year])))
        plt.legend(bbox_to_anchor =(1.25,1), loc='lower right')
        plt.xlabel('Rounds',size=15)
        plt.ylabel('Position',size=15)
        plt.title(f'{team}, Drivers-Standing: Position Vs Rounds for the Year {year}')
        
        
        
    elif mode == 'overall':
        plt.style.use('seaborn-darkgrid')
        test = driver.loc[year].reset_index()
        plt.style.use('seaborn-whitegrid')
        plt.figure(figsize=(11,8))
        sns.lineplot(data=test, x='level_0',y='position',
                     style='code', hue='Constructors',
                     markers=True,linewidth=3, palette='Set2' )
        plt.xticks(ticks=list(range(1,rounds[year])))
        plt.legend(bbox_to_anchor =(1.25,-0.1), loc='lower right')
        plt.xlabel('Rounds',size=15)
        plt.ylabel('Position',size=15)
        plt.title(f'Drivers Standing Position Vs Rounds for the Year {year}')




def qualifying_comparison(driver1, driver2, joined, driver_dict, delta_required, session_obj, event, year, mode):



    AB2, BN2, TN2, TC2  = driver_dict[driver2]
    driver2_data = joined.pick_driver(AB2).pick_fastest()
    driver2_compound = driver2_data['Compound']
    
    
    Comparison = [driver1, driver2]
    cols = st.columns([6,6])
    for i, driver in enumerate(Comparison, 0):

        cols[i].markdown('***')                            
        AB, BN, TN, TC  = driver_dict[driver]
        
        if TN == 'Haas F1 Team':
            TC = 'bcbcbc'
        
        # cols[1].markdown('')
        if mode == 'different':
            cols[i].markdown(f'''<h5 style="font-family:formula1, syne; font-weight:800;">{BN} <br><sub style='color:#{TC};'> {TN}</sub></h4>''',unsafe_allow_html=True)
            # cols[0].markdown('')
            # cols[0].markdown('')
        else:
            cols[i].markdown(f'''<h5 style="font-family:formula1, syne; font-weight:800;">{BN} <sub>({AB})</sub></h5>''',unsafe_allow_html=True)
            
        # cols[1].markdown('')
        
        cols[i].markdown('***')
        
        # driver_data
        driver_data = joined.pick_driver(AB)
        
        # fastest-lap and compound
        if driver_data.empty or pd.isnull(driver_data.pick_fastest()).all():
            
            st.error(f'''{BN}, has no Lap Records!.''')

        else:
        
            # Fastest Lap
            driver_data = joined.pick_driver(AB).pick_fastest()

            # compounds
            compound = driver_data['Compound'].lower()

        # flags 
        if pd.isnull(driver2_data).all():
            
            control = 'halt'
        elif pd.isnull(driver_data).all():
            # cols[0].warning("The Driver has No Lap Records!")
            control = 'halt'
        else:
            control = 'continue'

        
        
        if control == 'continue':
            # delta driver1 in-contrast to driver2
            delta_dict = {}
            for item in delta_required: 
                delta_dict[item] = delta_variation(driver_data[item], driver2_data[item])

        
            if driver_data['IsPersonalBest']:
                cols[i].markdown(f'''> <h5 style="font-family:formula1, syne; color:purple;">Lap Time - {timedelta_conversion(driver_data['LapTime'])} <sub style='color:black;'>Personal Best</sub></h5>''',unsafe_allow_html=True)
            else:
                cols[i].markdown(f'''> <h5 style="font-family:formula1, syne; color:purple;">Lap Time - {timedelta_conversion(driver_data['LapTime'])}</h5>''',unsafe_allow_html=True)
                
            
            cols[i].markdown(f'''<h6 style="font-family:formula1, syne;"><u>Sector Times</u></h6>''',unsafe_allow_html=True)
            # for j, item in enumerate(delta_required[1:],1):

            # st.write(delta_dict)
            
            cols[i].markdown(f'''<h6 style="font-family:formula1, syne; ">S1 - {timedelta_conversion(driver_data["Sector1Time"])} <sub>{delta_dict['Sector1Time']} ({driver2_data['Driver']})</sub></h6>''',unsafe_allow_html=True)
            cols[i].markdown(f'''<h6 style="font-family:formula1, syne; ">S2 - {timedelta_conversion(driver_data["Sector2Time"])} <sub>{delta_dict['Sector2Time']} ({driver2_data['Driver']})</sub></h6>''',unsafe_allow_html=True)
            cols[i].markdown(f'''<h6 style="font-family:formula1, syne; ">S3 - {timedelta_conversion(driver_data["Sector3Time"])} <sub>{delta_dict['Sector3Time']} ({driver2_data['Driver']})</sub></h6>''',unsafe_allow_html=True)

            cols[i].markdown('***')
            # compound 
            cols[i].markdown(f'''<h6 style="font-family:formula1, syne"><u>Compounds Used</u></h6>''',unsafe_allow_html=True)                                                                                
            cols[i].markdown(f'''<h6 style="font-family:formula1, syne; "><img src='data:image/png;base64,{img_to_bytes(f'./assets/{compound}.png')}' class='img-fluid' width=70 > {compound.upper()} Compound, <br> Tyre Life <span style='font-size:28px'>{driver_data['TyreLife']}</span> Laps</h6>''',unsafe_allow_html=True)
            cols[i].markdown('***')

            # Speed Traps
            
            cols[i].markdown(f'''<h6 style="font-family:formula1, syne;"><u>Speed Traps</u></h6>''',unsafe_allow_html=True)
            cols[i].markdown(f'''<h6 style="font-family:formula1, syne; ">Sector1 Speed - <span style='font-size:28px'>{driver_data['SpeedI1']}<sup>km/h</sup></span> </h6>''',unsafe_allow_html=True)
            cols[i].markdown(f'''<h6 style="font-family:formula1, syne; ">Sector2 Speed - <span style='font-size:28px'>{driver_data['SpeedI2']} <sup>km/h</sup></span> </h6>''',unsafe_allow_html=True)
            cols[i].markdown(f'''<h6 style="font-family:formula1, syne; ">Finish Line Speed - <span style='font-size:28px'>{driver_data['SpeedFL']} <sup>km/h</sup></span> </h6>''',unsafe_allow_html=True)
            cols[i].markdown(f'''<h6 style="font-family:formula1, syne; ">Longest Straight Speed - <span style='font-size:28px'>{driver_data['SpeedST']} <sup>km/h</sup></span> </h6>''',unsafe_allow_html=True)
            

            cols[i].markdown('***')
            cols[i].markdown(f'''<h6 style="font-family:formula1, syne;"><u>Weather Data</u></h6>''',unsafe_allow_html=True) 
            cols[i].markdown(f'''<h6 style="font-family:formula1, syne;">Air Temperature - {driver_data['AirTemp']} °C</h6>''',unsafe_allow_html=True)            
            cols[i].markdown(f'''<h6 style="font-family:formula1, syne;">Track Temperature - {driver_data['TrackTemp']} °C</h6>''',unsafe_allow_html=True)     
            cols[i].markdown(f'''<h6 style="font-family:formula1, syne;">Humidity - {driver_data['Humidity']}%</h6>''',unsafe_allow_html=True)       
            cols[i].markdown(f'''<h6 style="font-family:formula1, syne;">Pressure - {driver_data['Pressure']} Pa</h6>''',unsafe_allow_html=True)                                                                                                                                                     
            cols[i].markdown(f'''<h6 style="font-family:formula1, syne;">Wind Direction - {driver_data['WindDirection']}°</h6>''',unsafe_allow_html=True)
            cols[i].markdown(f'''<h6 style="font-family:formula1, syne;">Wind Speed - {driver_data['WindSpeed']} Kmph</h6>''',unsafe_allow_html=True)
            if driver_data['Rainfall']:
                cols[i].markdown(f'''<h6 style="font-family:formula1, syne;"> Rainfall - <img src='data:image/png;base64, {img_to_bytes('./assets/rain.png')}' class='img-fluid', width=35> </h6>''',unsafe_allow_html=True)
            else:
                cols[i].markdown(f'''<h6 style="font-family:formula1, syne;"> Rainfall - <img src='data:image/png;base64, {img_to_bytes('./assets/no-rain.png')}' class='img-fluid', width=35> </h6>''',unsafe_allow_html=True)
            
            cols[i].markdown('***') 
            

            
            try:
                lap = session_obj.laps.pick_driver(AB).pick_fastest()
                tel = lap.get_telemetry()
                x = np.array(tel['X'].values)
                y = np.array(tel['Y'].values)   
                
                cols[i].markdown(f'''<h6 style="font-family:formula1, syne;"><u>Lap Gear Shift Visualization</u></h6>''',unsafe_allow_html=True)
        

                expander = cols[i].expander(f'{BN}',expanded=True)
                with expander:
                    gear_heatmap(x,y,tel,driver,event,year)
            except:
                st.warning('Data Descrepancy!, No Telemetry Records Found. ')


            if i ==0:
                st.markdown('***')

                # st.markdown(f'''<h6 style="font-family:formula1, syne;"><u>Speed Chart Comparison</u></h6>''',unsafe_allow_html=True)
                st.markdown(f'''<h6 style="font-family:formula1, syne;"><u>Speed Vs Distance Visualisation</u>, <br> {AB} Vs {AB2} - {event} {year}</h6>''',unsafe_allow_html=True)
                
                driver_lap = session_obj.laps.pick_driver(AB).pick_fastest()
                driver_tel = driver_lap.get_car_data().add_distance()

                driver2_lap = session_obj.laps.pick_driver(AB2).pick_fastest()
                driver2_tel = driver2_lap.get_car_data().add_distance()
                
                try:
                    driver_color = '#'+TC
                    driver2_color = '#'+TC2
                except:
                    driver_color = 'gold'
                    driver2_color = 'purple'

                x1 = driver_tel['Distance']
                y1 = driver_tel['Speed']

                x2 = driver2_tel['Distance']
                y2 = driver2_tel['Speed']

                package = [(x1, y1, driver_color, AB),(x2, y2, driver2_color, AB2)]

                fig = speed_visualisation(package, mode=mode)

                # context = f'{AB} Speed'
                st.plotly_chart(fig)
                st.markdown('***') 



def qualifying(summarised_results, year, session_obj):

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
    

    preference = st.select_slider('Preference', [ 'Summarise', 'Get Nerdy!' ],key='mode of information')

    if preference == 'Summarise':

        st.markdown(f'''<h6 style="font-family:formula1, syne;">{session_select} Summary</h6>''',unsafe_allow_html=True)
        select_choice = st.selectbox('Data Summary', ['Choose','Q1 Grid-Positions', 'Q2 Grid-Positions', 'Q3 Grid-Positions', 'Knocked Out', 'Final Grid-Positions'])
            

        if not select_choice == 'Choose':
            
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

        if year >= 2018:
            st.markdown(f'''<h6 style="font-family:formula1, syne;">{session_select} Comprehensive Analysis</h6>''',unsafe_allow_html=True)

            cols = st.columns([6,3])
            placeholder = cols[1].empty()
            analysis_type = cols[0].selectbox('Select to Investigate', ['Analysis Type?','Driver Performance Analysis', 'Team Performance Analysis'],key='key-analysis')
            placeholder.selectbox('?',[])


            # laps
            laps = session_obj.laps.reset_index(drop=True)

            # weather data
            weather_data = session_obj.laps.get_weather_data()
            weather_data = weather_data.reset_index(drop=True)

            # club data
            joined = pd.concat([laps, weather_data.loc[:, ~(weather_data.columns == 'Time')]], axis=1) #from the fastf1 documentation
            
            delta_required = [ 'LapTime',
                                        'Sector1Time',
                                        'Sector2Time',
                                        'Sector3Time',   ]  
                    

            

            if analysis_type == 'Driver Performance Analysis':

            
                # overwrite placeholder selectbox

                
                analysis_mode = placeholder.selectbox('Type of Analysis', ['Individual','Comparative'])
                
                    

                if analysis_mode == 'Individual':

                    driver = st.selectbox('Choose Driver', driver_dict.keys())
                    fastest_driver = list(driver_dict.keys())[0]

                    st.markdown('***')
                    st.markdown(f'''<h6 style="font-family:formula1, syne;"><u> Driver Performance Investigation</u></h6>''',unsafe_allow_html=True)
                    AB, BN, TN, TC  = driver_dict[driver]
                    st.markdown(f'''<h6 style="font-family:formula1, syne;">Fastest Lap Analysis</h6>''',unsafe_allow_html=True)
                    st.markdown(f'''<h4 style="font-family:formula1, syne; font-weight:800;">{BN} ({AB})<sub style='color:#{TC}'>{TN}</sub></h4>''',unsafe_allow_html=True)
            
                    
                    # session data 
                    # session = return_session_object(year, event, session_select)

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
                        fdriver = fastest['Driver']

                        # st.write(fdriver)
                        # st.write()


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
                        st.markdown(f'''<h6 style="font-family:formula1, syne; "><img src='data:image/png;base64,{img_to_bytes(f'./assets/{compound}.png')}' class='img-fluid' width=70 > {compound.upper()} Compound, Tyre Life <span style='font-size:28px'>{driver_data['TyreLife']}</span> Laps</h6>''',unsafe_allow_html=True)
                        st.markdown('***')

                        if not driver_data['Driver'] == fdriver:
                        # Speed Traps
                            st.markdown(f'''<h6 style="font-family:formula1, syne;"><u>Speed Traps</u></h6>''',unsafe_allow_html=True)
                            st.markdown(f'''<h6 style="font-family:formula1, syne; ">Sector1 Speed - <span style='font-size:28px'>{driver_data['SpeedI1']}<sup>km/h</sup></span> <sub>{fastest['SpeedI1']} km/h ({fastest['Driver']}) <img src='data:image/png;base64,{img_to_bytes(f'./assets/{fcompound}.png')}' class='img-fluid' width=40></sub></h6>''',unsafe_allow_html=True)
                            st.markdown(f'''<h6 style="font-family:formula1, syne; ">Sector2 Speed - <span style='font-size:28px'>{driver_data['SpeedI2']} <sup>km/h</sup></span> <sub>{fastest['SpeedI2']} km/h ({fastest['Driver']}) <img src='data:image/png;base64,{img_to_bytes(f'./assets/{fcompound}.png')}' class='img-fluid' width=40></sub></h6>''',unsafe_allow_html=True)
                            st.markdown(f'''<h6 style="font-family:formula1, syne; ">Finish Line Speed - <span style='font-size:28px'>{driver_data['SpeedFL']} <sup>km/h</sup></span> <sub>{fastest['SpeedFL']} km/h ({fastest['Driver']}) <img src='data:image/png;base64,{img_to_bytes(f'./assets/{fcompound}.png')}' class='img-fluid' width=40></sub></h6>''',unsafe_allow_html=True)
                            st.markdown(f'''<h6 style="font-family:formula1, syne; ">Longest Straight Speed - <span style='font-size:28px'>{driver_data['SpeedST']} <sup>km/h</sup></span> <sub>{fastest['SpeedST']} km/h ({fastest['Driver']}) <img src='data:image/png;base64,{img_to_bytes(f'./assets/{fcompound}.png')}' class='img-fluid' width=40></sub></h6>''',unsafe_allow_html=True)
                        else:
                            st.markdown(f'''<h6 style="font-family:formula1, syne;"><u>Speed Traps</u></h6>''',unsafe_allow_html=True)
                            st.markdown(f'''<h6 style="font-family:formula1, syne; ">Sector1 Speed - <span style='font-size:28px'>{driver_data['SpeedI1']}<sup>km/h</sup></span> </h6>''',unsafe_allow_html=True)
                            st.markdown(f'''<h6 style="font-family:formula1, syne; ">Sector2 Speed - <span style='font-size:28px'>{driver_data['SpeedI2']} <sup>km/h</sup></span> </h6>''',unsafe_allow_html=True)
                            st.markdown(f'''<h6 style="font-family:formula1, syne; ">Finish Line Speed - <span style='font-size:28px'>{driver_data['SpeedFL']} <sup>km/h</sup></span> </h6>''',unsafe_allow_html=True)
                            st.markdown(f'''<h6 style="font-family:formula1, syne; ">Longest Straight Speed - <span style='font-size:28px'>{driver_data['SpeedST']} <sup>km/h</sup></span> </h6>''',unsafe_allow_html=True)
                            

                
                        st.markdown('***')
                        
                        st.markdown(f'''<h6 style="font-family:formula1, syne;"><u>Weather Data</u></h6>''',unsafe_allow_html=True) 
                        st.markdown(f'''<h6 style="font-family:formula1, syne;">Air Temperature - {driver_data['AirTemp']} °C</h6>''',unsafe_allow_html=True)            
                        st.markdown(f'''<h6 style="font-family:formula1, syne;">Track Temperature - {driver_data['TrackTemp']} °C</h6>''',unsafe_allow_html=True)     
                        st.markdown(f'''<h6 style="font-family:formula1, syne;">Humidity - {driver_data['Humidity']}%</h6>''',unsafe_allow_html=True)       
                        st.markdown(f'''<h6 style="font-family:formula1, syne;">Pressure - {driver_data['Pressure']} Pa</h6>''',unsafe_allow_html=True)                                                                                                                                                     
                        st.markdown(f'''<h6 style="font-family:formula1, syne;">Wind Direction - {driver_data['WindDirection']}°</h6>''',unsafe_allow_html=True)
                        st.markdown(f'''<h6 style="font-family:formula1, syne;">Wind Speed - {driver_data['WindSpeed']} Kmph</h6>''',unsafe_allow_html=True)
                        if driver_data['Rainfall']:
                            st.markdown(f'''<h6 style="font-family:formula1, syne;"> Rainfall - <img src='data:image/png;base64, {img_to_bytes('./assets/rain.png')}' class='img-fluid', width=35> </h6>''',unsafe_allow_html=True)
                        else:
                            st.markdown(f'''<h6 style="font-family:formula1, syne;"> Rainfall - <img src='data:image/png;base64, {img_to_bytes('./assets/no-rain.png')}' class='img-fluid', width=35> </h6>''',unsafe_allow_html=True)

                        
                        st.markdown('***')

                        st.markdown(f'''<h6 style="font-family:formula1, syne;"><u>Speed Chart</u></h6>''',unsafe_allow_html=True)
                        st.markdown(f'''<h6 style="font-family:formula1, syne;">Speed Vs Distance Visualisation , <br> {driver} - {event} {year}</h6>''',unsafe_allow_html=True)
                        
                        if driver == fastest_driver:
                            driver_lap = session_obj.laps.pick_driver(AB).pick_fastest()
                            driver_tel = driver_lap.get_car_data().add_distance()
                            driver_color = '#'+TC
                            x = driver_tel['Distance']
                            y = driver_tel['Speed']
                            context = f'{AB} Speed'
                            package = [(x, y, driver_color, AB)]
                            fig = speed_visualisation(package, mode='individual')
                            st.plotly_chart(fig)
                        
                        else:
                            driver_lap = session_obj.laps.pick_driver(AB).pick_fastest()
                            driver_tel = driver_lap.get_car_data().add_distance()
                            try:
                                driver_color = '#'+TC
                            except:
                                driver_color = 'gold'
                            x1 = driver_tel['Distance']
                            y1 = driver_tel['Speed']

                            fdriver_lap = session_obj.laps.pick_fastest()
                            fdriver_tel = fdriver_lap.get_car_data().add_distance()
                            fdriver_color = fastf1.plotting.team_color(fdriver_lap['Team'])
                            fdriver_AB = fdriver_lap['Driver']
                            x2 = fdriver_tel['Distance']
                            y2 = fdriver_tel['Speed']

                            package = [(x1, y1, driver_color, AB),(x2, y2, fdriver_color, fdriver_AB)]


                            fig = speed_visualisation(package, mode='individual')


                            context = f'{AB} Speed'
                            st.plotly_chart(fig)


                        st.markdown(f'''<h6 style="font-family:formula1, syne;"><u>Gear Shifts</u></h6>''',unsafe_allow_html=True)

                        try:
                            lap = session_obj.laps.pick_driver(AB).pick_fastest()
                            tel = lap.get_telemetry()
                            x = np.array(tel['X'].values)
                            y = np.array(tel['Y'].values)   
                            st.markdown(f'''<h6 style="font-family:formula1, syne;">Lap Gear Shift Visualization, <br> {driver} - {event} {year}</h6>''',unsafe_allow_html=True)
                    
                            gear_heatmap(x,y,tel,driver,event,year)
                        except:
                            st.warning('Data Descrepancy!, No Telemetry Records Found. ')
                                                
                        
                else:
                    # COMPARATIVE
                    st.markdown(f'''<center><h6 style="font-family:formula1, syne;"><u>Comparative Driver Performance Investigation</u></h6></center>''',unsafe_allow_html=True)
                    st.markdown('')
                    cols = st.columns([5,5])
                    driver1 = cols[0].selectbox('Choose Driver 1', driver_dict.keys())
                    driver2 = cols[1].selectbox('Choose Driver 2', driver_dict.keys())
                    # fastest_driver = list(driver_dict.keys())[0]
                    
                    
                    st.markdown('')
                    st.markdown(f'''<center><h6 style="font-family:formula1, syne;">Fastest Lap Analysis</h6></center>''',unsafe_allow_html=True)
                    
                    # precomputing few aspects of the driver2 for Comparison 
                    AB2, BN2, TN2, TC2  = driver_dict[driver2]
                    driver2_data = joined.pick_driver(AB2).pick_fastest()
                    driver2_compound = driver2_data['Compound']
                    
            
                    if driver1 != driver2:

                        qualifying_comparison(driver1, driver2, joined, driver_dict, delta_required, session_obj, event, year, mode='different')
                        

                    else:
                        st.warning('Select Different Drivers to Compare.')
                        
            elif analysis_type == 'Team Performance Analysis':

                team_data = session_obj.results
                team_data = team_data.set_index('TeamName')
                team_data = team_data.drop(team_data.columns[7:], axis=1)

                team = placeholder.selectbox('Select Team', team_data.index.unique(),key='teams')   
                DN1, BN1, AB1, TC1, FN1, LN1, FuN1 = team_data.loc[team].reset_index(drop=True).loc[0,:]
                DN2, BN2, AB2, TC2, FN2, LN2, FuN2 = team_data.loc[team].reset_index(drop=True).loc[1,:]
                TN = team

                st.markdown(f'''<h3 style="font-family:formula1, syne; font-weight:800; color:#{TC1};"><center>{TN}</center></h3>''',unsafe_allow_html=True)

                qualifying_comparison(FuN1, FuN2, joined, driver_dict, delta_required, session_obj, event, year, mode='same')


                        
                        

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

def fetch_compounds(driver_data):
    compounds = {}
    initial = list(driver_data['Compound'].dropna().unique())[0]
    compounds[1] = initial

    lapnumbers = list(driver_data[~driver_data['PitInTime'].isnull()].loc[:,'LapNumber'].values)
    nextcompounds = []
    nextlaps = []
    for lapnumber in lapnumbers:
        try:
            nextcompounds.append(driver_data[driver_data['LapNumber']==lapnumber+1].loc[:,'Compound'].values[0])
            nextlaps.append(driver_data[driver_data['LapNumber']==lapnumber+1].loc[:,'LapNumber'].values[0])
        except:
            nextcompounds.append(driver_data[driver_data['LapNumber']==lapnumber].loc[:,'Compound'].values[0])
            nextlaps.append(driver_data[driver_data['LapNumber']==lapnumber].loc[:,'LapNumber'].values[0])
    
    for lap, compound in zip(nextlaps, nextcompounds):

        compounds[int(lap)] = compound
        
    return compounds

def fetch_strategy(driver_data, total=None,mode='previous'):
    
    if mode == 'previous':
        dfs = []
        for compound in driver_data['Compound'].dropna().unique():
            dfs.append(driver_data[driver_data['Compound']==compound])
        lapschanged = []
        for df in dfs:
            lapschanged.append(df.iloc[-1:,:]['LapNumber'].values[0])

        if driver_data.shape[0] == 1 or driver_data.empty:
            return None, None
        else:
            lap_retired = driver_data.shape[0]
            for i in range(len(lapschanged)):
                try:
                    if driver_data[driver_data['LapNumber'] == lapschanged[i]+1].empty:
                        print('first if')
                        lap_retired = lapschanged[i]
                    else:
                        # second check
                        if driver_data[driver_data['LapNumber'] == lapschanged[i+1]+1].empty:
                            print('second if')
                            lap_retired = lapschanged[i+1]

                        else:
                            compounds = fetch_compounds(driver_data)
                            return compounds, lap_retired
                except:
                    compounds = fetch_compounds(driver_data)
                    return compounds, lap_retired
                
            compounds = fetch_compounds(driver_data)
            return compounds, lap_retired

    else:
        
        if driver_data.shape[0] == 1 or driver_data.empty:
                print ('No Data Records, check knockout list.')
        else:
            dfs = []
            for compound in driver_data['Compound'].dropna().unique():
                dfs.append(driver_data[driver_data['Compound']==compound])
            lapschanged = []
            for df in dfs:
                lapschanged.append(df.iloc[-1:,:]['LapNumber'].values[0])

            lapschanged = list(map(int, lapschanged))

            firstphase = driver_data[driver_data['LapNumber']<lapschanged[0]]['Compound'].value_counts().idxmax()
            lapschanged.insert(0,1)

            retired_flag = False
            if lapschanged[-1:][0] < 64 and not (lapschanged[-1:][0] - total) == 1:
                retired_flag = True
                pairs = []
                for lap, i in zip(lapschanged, range(len(lapschanged)) ):
                    try:
                        pairs.append((lapschanged[i],lapschanged[i+1]-1))
                    except:
                        pass
            else:
                pairs = []
                for lap, i in zip(lapschanged, range(len(lapschanged)) ):
                    try:
                        pairs.append((lapschanged[i],lapschanged[i+1]))
                    except:
                        pass


            print(lapschanged)
            print(pairs)

            compounds = {}
            for df, laprange in zip(dfs, pairs):
                compound = df['Compound'].value_counts().idxmax()
            #     print(df['Compound'].value_counts().idxmax())
                compounds[compound] = laprange

            if retired_flag:
                compounds['Retired'] = (lapschanged[-1:][0], total)

            print(compounds)
            return compounds, lapschanged, pairs

                
def display_strategy(presets, mode='previous', team=False):


    # color_dict
    color_dict = {'soft':'red','supersoft':'red','ultrasoft':'purple','hypersoft':'purple','hard':'grey','medium':'gold','intermediate':'limegreen','wet':'dodgerblue'} 


     # Tyre Information
    information_dict = {'soft':'A soft tyre is stickier, allowing the driver to: Accelerate faster without spinning the rear wheels, because the car has more traction; Brake harder without locking; Take corners at a higher speed.',
    'supersoft':'It is a very adaptable tire that can be used as the softest compound at a high-severity track as well as the hardest compound at a low-severity track or street circuit. It is one of the most commonly used compounds of all.',
    'ultrasoft':'As the very softest tire in the range, designed to sit below the supersoft, it has a very rapid warm-up and huge peak performance, but the other side of this is its relatively limited overall life.',
    'hypersoft':'Is the heir to the universally-popular hyper-soft: the fastest compound that Pirelli has ever made. This tire is suitable for all circuits that demand high levels of mechanical grip, but the trade-off for this extra speed and adhesion is a considerably shorter lifespan than the other tires in the range. It is not a qualifying tire, but it comes closest.',
    'hard':'Hardest Tyre from the lot, It is designed for circuits that put the highest energy loadings through the tires, which will typically feature fast corners, abrasive surfaces, or high ambient temperatures. The compound takes longer to warm up but offers maximum durability and provides low degradation.',
    'medium':'Strikes a very good balance between performance and durability, with the accent on performance. It is a very adaptable tire that can be used as the softest compound at a high-severity track as well as the hardest compound at a low-severity track or street circuit. It is one of the most commonly used compounds of all.',
    'intermediate':'The intermediates are the most versatile of the rain tires. They can be used on a wet track with no standing water, as well as a drying surface. This tire evacuates 30 litres of water per second per tire at 300kph. The compound has been designed to expand the working range, as seen at a number of races last year, guaranteeing a wide crossover window both with the slicks and the full wets.',
    'wet':'The full wet tires are the most effective for heavy rain. These tires can evacuate 85 litres of water per second per tire at 300kph: when it rains heavily, visibility rather than grip causes issues. The profile has been designed to increase resistance to aquaplaning, which gives the tire more grip in heavy rain. The diameter of the full wet tire is 10mm wider than the slick tire.'}
                    


    if mode == 'previous':

        # unwarp 
        strategy, lap_retired, total_laps = presets

        if not strategy == None and not lap_retired == None:

            last_lap_error = False
            if not int(lap_retired) == total_laps:
                strategy[lap_retired] = 'Retired' 
                if (total_laps - lap_retired) == 1:
                    last_lap_error = True
                        
                               
            
            
            laps0 = list(strategy.keys())
            laps0.append(total_laps)
            laps1 = list(strategy.keys())


            # range of tyre usage
            pairs = []
            for lap, i in zip(laps1, range(len(laps1)) ):
                try:
                    pairs.append((laps1[i],laps1[i+1]-1))
                except:
                    pass

            pairs.append((laps1[-1:][0], total_laps))                                                                  
            
            cols = st.columns(laps0[1:])
            for lap, i in zip(laps0 ,range(len(laps0[1:]))):
                range_of_usage = pairs[i]
                cols[i].markdown(f'''<center><span style='font-size:28px;'>{int(range_of_usage[0])} - {int(range_of_usage[1])}</span></center>''',unsafe_allow_html=True)
                try:
                    cols[i].markdown(f'''<hr style="height:10px; width:100%; border-width:0; color:{color_dict[strategy[lap].lower()]}; background-color:{color_dict[strategy[lap].lower()]}">''',unsafe_allow_html=True)
                    cols[i].markdown(f'''<center><span style='font-size:28px;'><img src='data:image/png;base64,{img_to_bytes(f'./assets/{strategy[lap].lower()}.png')}' class='img-fluid' width=50></span></center>''',unsafe_allow_html=True)
                except:
                    cols[i].markdown(f'''<hr style="height:10px; width:100%; border-width:0; color:black; background-color:black;">''',unsafe_allow_html=True)
                    if last_lap_error:
                        cols[i].markdown(f'''<center><span style='font-size:28px;'>+1</span></center>''',unsafe_allow_html=True)
                    else:
                        cols[i].markdown(f'''<center><span style='font-size:28px;'>Retired</span></center>''',unsafe_allow_html=True)


            # baseline 
            st.markdown(f'''<hr style="height:5px; width:100%; border-width:0; color:black; background-color:black">''',unsafe_allow_html=True)
            st.markdown(f'''<center><span style='font-size:35px;'>{total_laps} Laps</span></center>''',unsafe_allow_html=True)
            st.markdown('')

            if not team:
                tyreexp = st.expander('Tyre Information', expanded=True)
                # st.write(strategy)
                if 'Retired' in strategy.values():
                    strategy.popitem()
                # st.write(strategy)

                compounds = set(strategy.values())
                for compound in compounds: 
                    tyreexp.markdown(f'''* <span> <img src='data:image/png;base64,{img_to_bytes(f'./assets/{compound.lower()}.png')}' class='img-fluid' width=35> {compound}</span>''',unsafe_allow_html=True)
                    tyreexp.markdown(f'''> <p style='font-size:15px;text-align:justify;'>{information_dict[compound.lower()]}</p>''',unsafe_allow_html=True)



                disclaimer = st.expander('Disclaimer?')
                disclaimer_info = '''
                * The Retired Lap Range representation might not be accurate to the real-life scenario. 
                * '+1' Represents a Missing Data record, in this scenario the driver isn't retired.'''
                disclaimer.info(disclaimer_info)
                # st.markdown('***')
                

        else:
            st.warning(f'No Lap Records Found, Check the Retired List.')
            st.markdown('***')

    else:

        compounds, lapschanged, pairs, total = presets
        # st.code(compounds)
        # st.code(lapschanged)
        # st.code(pairs)
        # st.code(total)
        column_lengths = []

        for pair in pairs:
            l, u = pair
            value = int(u) - int(l)
            column_lengths.append(value)

        # st.code(column_lengths)
        cols = st.columns(column_lengths)

        for compound, lrange, i in zip(compounds.keys(), compounds.values(), range(len(compounds))):

            lower, upper = lrange
            cols[i].markdown(f'''<center><span style='font-size:28px;'>{lower} - {upper}</span></center>''',unsafe_allow_html=True)
            
            cols[i].markdown(f'''<hr style="height:10px; width:100%; border-width:0; color:{color_dict[compound.lower()]}; background-color:{color_dict[compound.lower()]}">''',unsafe_allow_html=True)
            cols[i].markdown(f'''<center><span style='font-size:28px;'><img src='data:image/png;base64,{img_to_bytes(f'./assets/{compound.lower()}.png')}' class='img-fluid' width=50></span></center>''',unsafe_allow_html=True)
            # except:
            #     cols[i].markdown(f'''<hr style="height:10px; width:100%; border-width:0; color:black; background-color:black;">''',unsafe_allow_html=True)
            #     # if last_lap_error:
                #     cols[i].markdown(f'''<center><span style='font-size:28px;'>+1</span></center>''',unsafe_allow_html=True)
                # else:
                #     cols[i].markdown(f'''<center><span style='font-size:28px;'>Retired</span></center>''',unsafe_allow_html=True)


        # baseline 
        st.markdown(f'''<hr style="height:5px; width:100%; border-width:0; color:black; background-color:black">''',unsafe_allow_html=True)
        st.markdown(f'''<center><span style='font-size:35px;'>{total} Laps</span></center>''',unsafe_allow_html=True)
        st.markdown('')

        if not team:
            tyreexp = st.expander('Tyre Information', expanded=True)
            # st.write(strategy)
            if 'Retired' in compounds.keys():
                strategy.popitem()
            # st.write(strategy)

            compounds = set(compounds.keys())
            for compound in compounds: 
                tyreexp.markdown(f'''* <span> <img src='data:image/png;base64,{img_to_bytes(f'./assets/{compound.lower()}.png')}' class='img-fluid' width=35> {compound}</span>''',unsafe_allow_html=True)
                tyreexp.markdown(f'''> <p style='font-size:15px;text-align:justify;'>{information_dict[compound.lower()]}</p>''',unsafe_allow_html=True)


            disclaimer = st.expander('Disclaimer?')
            disclaimer_info = '''
            * The Retired Lap Range representation might not be accurate to the real-life scenario. 
            * '+1' Represents a Missing Data record, in this scenario the driver isn't retired.'''
            disclaimer.info(disclaimer_info)
            st.markdown('***')
        
@st.cache(persist=True)
def get_race_lap_data(_driverlaps, driver_AB):

    laps_data = driverlaps.copy(deep=True)
    laps_data = laps_data.reset_index(drop=True)

    weather_data = laps_data.get_weather_data()
    weather_data = weather_data.reset_index(drop=True)

    joined = pd.concat([laps_data, weather_data.loc[:, ~(weather_data.columns == 'Time')]], axis=1) #from the fastf1 documentation
    driver_data = joined.pick_driver(driver_AB)

    return joined, driver_data

@st.cache(persist=True)
def parse_race_points(year):

    driver_url = f'http://ergast.com/api/f1/{year}/driverStandings.json'
    const_url = f'http://ergast.com/api/f1/{year}/constructorStandings.json'

    result_driver =  requests.get(driver_url)
    result_const =  requests.get(const_url) 

    # constructors
    const_list = result_const.json()['MRData']['StandingsTable']['StandingsLists'][0]['ConstructorStandings']
    required = ['position','points','wins']
    constructors = {}
    for i,item in enumerate(const_list,0):
        temp = {}
        for element in required:
            temp[element] = item[element]
        temp['Team'] = item['Constructor']['name']
        temp['Nationality'] = item['Constructor']['nationality']
        constructors[i] = temp

    # Drivers
    driver_pos_list = result_driver.json()['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings']
    required, driver_details, loc = ['position','points','wins'],['permanentNumber','code','givenName','familyName'],['name','nationality']
    driver_standings = {}
    for i, item in enumerate(driver_pos_list,0):
        temp = {}
        for element in required:
            temp[element] = item[element]
        for dd in driver_details:
            temp[dd] = item['Driver'][dd]
        for ll in loc:
            temp[ll] = item['Constructors'][0][ll]
        driver_standings[i] = temp

    return constructors, driver_standings



def driver_race_analysis(year, event, driverlaps, driver_AB, driver_data, total_laps,mode, team=False, preset=None):

    
    if not team:
        # fetch strategy 
        st.markdown('***')
        st.markdown('''<center><span style='font-weight:800; font-size:28px;'>Race Strategy</span></center>''', unsafe_allow_html=True)
        st.markdown('***')

        driver_info = session_obj.get_driver(driver_AB)
        dn, bn, ab, tn, tc  = driver_info[:5]
        st.markdown(f'''<h4 style="font-family:formula1, syne; font-weight:800;">{bn} ({ab}) <sub style='color:#{tc}'>{tn}</sub></h4>''',unsafe_allow_html=True)
        # st.subheader(driver_AB) # subheader -- racer name 

        if mode == 'previous':
            strategy, lap_retired = fetch_strategy(driver_data)
            presets = [strategy, lap_retired, total_laps]  
            display_strategy(presets, mode='previous')
            

            
        else:
            compounds, lapschanged, pairs = fetch_strategy(driver_data,total_laps, mode='current') 
            strategy, lap_retired = fetch_strategy(driver_data)
            if event == 'Monaco Grand Prix':
                presets = [compounds, lapschanged, pairs, total_laps] 
                display_strategy(presets, mode='current')
            else:
                presets = [strategy, lap_retired, total_laps]  
                display_strategy(presets, mode='previous')

    if team:
        TC, AB = preset
        st.markdown(f'''<center><span style='font-weight:800; font-size:28px;'>Stints <span style='color:#{TC};'>{AB}</span></span></center>''', unsafe_allow_html=True)
        st.markdown('***')
    else:
        st.markdown(f'''<center><span style='font-weight:800; font-size:28px;'>Stints</span></center>''', unsafe_allow_html=True)
        st.markdown('***')


    # cols = st.columns([6,2])
    # category = cols[0].select_slider('Lap Data Presentation',['Detailed','Summarised'],key=f'{mode}')
    # cols[1].markdown('> <- Select Mode')
    # st.markdown('***')

    

    st.markdown('**Stintwise, Individual Lap Data**')

    required = ['LapTime','LapNumber','Sector1Time','Sector2Time','Sector3Time',
    'SpeedI1','SpeedI2','SpeedFL','SpeedST','Compound','TyreLife','AirTemp',
    'Humidity','Pressure','Rainfall','TrackTemp','WindDirection','WindSpeed']

    joined, driver_data = get_race_lap_data(driverlaps, driver_AB)
    # st.markdown(driver_AB)

    # st.write(driver_data.shape[0])
    
    if driver_data.empty or int(driver_data.shape[0]) == 1:
        
        st.warning(f'No Lap Records Found for **{bn}**, Check the Retired List.')
    
    else:
        
        stints = driver_data['Stint'].unique()
        stints = list(map(int, stints))
        stint_dfs = []
        for stint in stints:
            stint_dfs.append(driver_data[driver_data['Stint']==stint])

        stint_minmax = []
        for stint_df in stint_dfs:
            stint_minmax.append((stint_df['LapNumber'].min(),stint_df['LapNumber'].max()))


        expander_objects = []
        for stint in stints:
            expander_objects.append(st.expander(f'Stint {stint}'))

        
        
        for stint_df,expander, stint_range in zip(stint_dfs, expander_objects, stint_minmax):
            min_val, max_val = stint_range
            min_val, max_val = int(min_val), int(max_val)
            # st.write(min_val, max_val)
            compound_title = expander.empty()
            random_num = np.random.randint(0,1000,1)[0]
            lapnum = expander.number_input('Laps', min_value=min_val,max_value=max_val, key=f'{driver_AB}'+str(random_num))
            lapnum = int(lapnum)
            # st.write(lapnum)
            # stint_df = stint_df.set_index('LapNumber')
            # expander.write(stint_df.iloc[lapnum,:]['IsPersonalBest'])           
            # st.write(stint_df['Driver'])
            
            stint_df = stint_df.fillna('0')
            
            selector = stint_df[stint_df['LapNumber']==lapnum]                                    

                

            expander.markdown('***')
            # compound 
            compound_title.markdown(f'''<center><h6 style="font-family:formula1, syne; "><img src='data:image/png;base64,{img_to_bytes(f"./assets/{selector['Compound'].values[0].lower()}.png")}' class='img-fluid' width=70 >  {selector['Compound'].values[0].lower()} Compound</h6> <sub><h6 style="font-family:formula1, syne; ">Tyre Life <span style='font-size:28px'>{selector['TyreLife'].values[0]}</span> Laps</h6></sub></center>''',unsafe_allow_html=True)
            # expander.markdown(f'''> <h6 style="font-family:formula1, syne; ">Tyre Life <span style='font-size:28px'>{selector['TyreLife'].values[0]}</span> Laps</h6>''',unsafe_allow_html=True)                                                                              
            # expander.markdown('***')

            #Time 
            if selector['IsPersonalBest'].values[0]:

                if not int(lapnum) == min_val:
                    expander.markdown(f'''> <h5 style="font-family:formula1, syne; color:black;">Lap Time - {timedelta_conversion(pd.Timedelta(selector['LapTime'].T.values[0]))} <sub style='color:black;'>Personal Best</sub></h5>''',unsafe_allow_html=True)
                else:
                    expander.markdown(f'''> <h5 style="font-family:formula1, syne; color:black;">Out Lap</h5>''',unsafe_allow_html=True)

            else:
                if not int(lapnum) == min_val:
                    expander.markdown(f'''> <h5 style="font-family:formula1, syne; color:black;">Lap Time - {timedelta_conversion(pd.Timedelta(selector['LapTime'].T.values[0]))}</h5>''',unsafe_allow_html=True)
                else:
                    expander.markdown(f'''> <h5 style="font-family:formula1, syne; color:black;">Out Lap</h5>''',unsafe_allow_html=True)

            expander.markdown('***')

            # Sector Times
            expander.markdown(f'''<h6 style="font-family:formula1, syne;"><u>Sector Times</u></h6>''',unsafe_allow_html=True)
            expander.markdown(f'''<h6 style="font-family:formula1, syne; ">Sector1 Time - <span style='font-size:28px'>{timedelta_conversion(pd.Timedelta(selector['Sector1Time'].values[0]))}</span></h6>''',unsafe_allow_html=True)
            expander.markdown(f'''<h6 style="font-family:formula1, syne; ">Sector2 Time - <span style='font-size:28px'>{timedelta_conversion(pd.Timedelta(selector['Sector2Time'].values[0]))}</span></h6>''',unsafe_allow_html=True)
            expander.markdown(f'''<h6 style="font-family:formula1, syne; ">Sector3 Time - <span style='font-size:28px'>{timedelta_conversion(pd.Timedelta(selector['Sector3Time'].values[0]))}</span></h6>''',unsafe_allow_html=True)
            
                # Speed Traps
            expander.markdown('***')
            expander.markdown(f'''<h6 style="font-family:formula1, syne;"><u>Speed Traps</u></h6>''',unsafe_allow_html=True)
            expander.markdown(f'''<h6 style="font-family:formula1, syne; ">Sector1 Speed - <span style='font-size:28px'>{selector['SpeedI1'].values[0]}<sup>km/h</sup></span></h6>''',unsafe_allow_html=True)
            expander.markdown(f'''<h6 style="font-family:formula1, syne; ">Sector2 Speed - <span style='font-size:28px'>{selector['SpeedI2'].values[0]} <sup>km/h</sup></span></h6>''',unsafe_allow_html=True)
            expander.markdown(f'''<h6 style="font-family:formula1, syne; ">Finish Line Speed - <span style='font-size:28px'>{selector['SpeedFL'].values[0]} <sup>km/h</sup></span></h6>''',unsafe_allow_html=True)
            expander.markdown(f'''<h6 style="font-family:formula1, syne; ">Longest Straight Speed - <span style='font-size:28px'>{selector['SpeedST'].values[0]} <sup>km/h</sup></span> </h6>''',unsafe_allow_html=True)
            


            expander.markdown('***')
            if expander.checkbox('Show Weather Data',key=f'value{driver_AB}{min_val}'):
                expander.markdown(f'''<h6 style="font-family:formula1, syne;"><u>Weather Data</u></h6>''',unsafe_allow_html=True) 
                expander.markdown(f'''<h6 style="font-family:formula1, syne;">Air Temperature - {selector['AirTemp'].values[0]} °C</h6>''',unsafe_allow_html=True)            
                expander.markdown(f'''<h6 style="font-family:formula1, syne;">Track Temperature - {selector['TrackTemp'].values[0]} °C</h6>''',unsafe_allow_html=True)     
                expander.markdown(f'''<h6 style="font-family:formula1, syne;">Humidity - {selector['Humidity'].values[0]}%</h6>''',unsafe_allow_html=True)       
                expander.markdown(f'''<h6 style="font-family:formula1, syne;">Pressure - {selector['Pressure'].values[0]} Pa</h6>''',unsafe_allow_html=True)                                                                                                                                                     
                expander.markdown(f'''<h6 style="font-family:formula1, syne;">Wind Direction - {selector['WindDirection'].values[0]}°</h6>''',unsafe_allow_html=True)
                expander.markdown(f'''<h6 style="font-family:formula1, syne;">Wind Speed - {selector['WindSpeed'].values[0]} Kmph</h6>''',unsafe_allow_html=True)
                if selector['Rainfall'].values[0]:
                    expander.markdown(f'''<h6 style="font-family:formula1, syne;"> Rainfall - <img src='data:image/png;base64, {img_to_bytes('./assets/rain.png')}' class='img-fluid', width=35> </h6>''',unsafe_allow_html=True)
                else:
                    expander.markdown(f'''<h6 style="font-family:formula1, syne;"> Rainfall - <img src='data:image/png;base64, {img_to_bytes('./assets/no-rain.png')}' class='img-fluid', width=35> </h6>''',unsafe_allow_html=True)

    
    
    

    st.markdown('***')


def display_race_standings(results, top3):

    st.markdown('***')
    st.markdown('''<center><span style='font-weight:800; font-size:28px;'>Race Standings</span></center>''', unsafe_allow_html=True)
    st.markdown('***')

    # podium
    p1 = top3.iloc[0,:]
    p2 = top3.iloc[1,:]
    p3 = top3.iloc[2,:]

    # reference
    #  cols[1].markdown(f'''<center><span style='font-size:70px; font-weight:bold;  border-bottom:3px solid #000;'>P1<sup>{points}</sup> </span><br><span style='font-size:32px;'>{fn} <b>{ln}</b></span> <br><sub style='color:#{tc}'><b>{tn}</b></sub> <br><sub>positions {image} {abs(gained)}</sub></center>''',unsafe_allow_html=True)
                                
    cols = st.columns([9,9,9])
    dn, bn, ab, tn, tc, fn, ln,  pn, gp, _, status, points = p1
    gained = gp-pn
    if gained < 0:
        image = f'''<img src='data:image/png;base64,{img_to_bytes(f"./assets/red_down.png")}' class='img-fluid' width=25 >'''
    else:
        image =  f'''<img src='data:image/png;base64,{img_to_bytes(f"./assets/green_up.png")}' class='img-fluid' width=25 >'''
    cols[1].markdown(f'''<center><span style='font-size:70px; font-weight:bold;  border-bottom:3px solid #000;font-weight:800;' >P1<sup>{points}</sup> </span><br><span style='font-size:32px;'>{fn} <b style='font-weight:800;'>{ln}</b></span> <br><sub style='color:#{tc}'><b>{tn}</b></sub> </center>''',unsafe_allow_html=True)
    cols[0].markdown('')
    cols[0].markdown('')
    cols[0].markdown('')               
    
    dn, bn, ab, tn, tc, fn, ln,  pn, gp, _, status, points = p2
    cols[0].markdown(f'''<center><span style='font-size:70px; font-weight:bold;  border-bottom:3px solid #000;font-weight:800;'>P2<sup>{points}</sup> </span><br><span style='font-size:32px;'>{fn} <b style='font-weight:800;'>{ln}</b></span> <br><sub style='color:#{tc}'><b>{tn}</b></sub> </center>''',unsafe_allow_html=True)
    

    cols[2].markdown('')
    cols[2].markdown('')
    cols[2].markdown('') 
    dn, bn, ab, tn, tc, fn, ln,  pn, gp, _, status, points = p3
    cols[2].markdown(f'''<center><span style='font-size:70px; font-weight:bold;  border-bottom:3px solid #000;font-weight:800;'>P3<sup>{points}</sup> </span><br><span style='font-size:32px;'>{fn} <b style='font-weight:800;'>{ln}</b></span> <br><sub style='color:#{tc}'><b>{tn}</b></sub> </center>''',unsafe_allow_html=True)  

    st.markdown('***')

    cols = st.columns([5,5])
    for index in range(len(results)):
        dn, bn, ab, tn, tc, fn, ln, pn, gp, time, status, points = results.iloc[index,:]
        gained = gp-pn
        if gained < 0:
            image = f'''<img src='data:image/png;base64,{img_to_bytes(f"./assets/red_down.png")}' class='img-fluid' width=25 >'''
        else:
            image =  f'''<img src='data:image/png;base64,{img_to_bytes(f"./assets/green_up.png")}' class='img-fluid' width=25 >'''
        
        if index%2==0:
            if status == 'Finished' and status == '+1 Lap':
                cols[0].markdown(f'''<center><span style='font-size:45px; font-weight:bold;  border-bottom:3px solid #000;font-weight:800;'>P{int(pn)}<sup>{points}</sup> </span><br><span style='font-size:32px;'>{fn} <b style='font-weight:800;'>{ln}</b></span> <br><sub style='color:#{tc}'><b>{tn}</b></sub> <br><sub>positions {image} {abs(gained)}</sub></center>''',unsafe_allow_html=True)
            else:
                cols[0].markdown(f'''<center><span style='font-size:45px; font-weight:bold;  border-bottom:3px solid #000;font-weight:800;'>P{int(pn)}<sup>{points}</sup> </span><br><span style='font-size:32px;'>{fn} <b style='font-weight:800;'>{ln}</b></span> <br><sub style='color:#{tc}'><b>{tn}</b></sub> <br><sub><b>Status</b> - {status}</sub></center>''',unsafe_allow_html=True)

            cols[1].markdown('')
            cols[1].markdown('')

        else:
            cols[0].markdown('')
            cols[0].markdown('')
            if status == 'Finished' and status == '+1 Lap':
                cols[1].markdown(f'''<center><span style='font-size:45px; font-weight:bold;  border-bottom:3px solid #000;font-weight:800;'>P{int(pn)}<sup>{points}</sup> </span><br><span style='font-size:32px;'>{fn} <b style='font-weight:800;'>{ln}</b></span> <br><sub style='color:#{tc}'><b>{tn}</b></sub> <br><sub>positions {image} {abs(gained)}</sub></center>''',unsafe_allow_html=True)
            else:
                cols[1].markdown(f'''<center><span style='font-size:45px; font-weight:bold;  border-bottom:3px solid #000;font-weight:800;'>P{int(pn)}<sup>{points}</sup> </span><br><span style='font-size:32px;'>{fn} <b style='font-weight:800;'>{ln}</b></span> <br><sub style='color:#{tc}'><b>{tn}</b></sub> <br><sub><b>Status</b> - {status}</sub></center>''',unsafe_allow_html=True)

    st.markdown('***')








if __name__ == '__main__':

    # page configuration
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.set_page_config(page_title="F1 Web Paddock",layout="wide",initial_sidebar_state="expanded",page_icon='./assets/tab-logo.png')
    

    # # font-base64
    # with open('./assets/formula1-regular-base64.txt', 'r') as f:
    #     regular = f.read()

    # with open('./assets/formula1-bold-base64.txt', 'r') as f:
    #     bold = f.read()

    # with open('./assets/formula1-black-base64.txt', 'r') as f:
    #     black = f.read()


    # font-style
    font_url = '''
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Syne:wght@700;800&display=swap" rel="stylesheet">
    '''
    st.markdown(f"{font_url}",unsafe_allow_html=True)

    # fonts-style preset
    style = '''
    <style>
    html, body, [class*="css"] {
        font-family: syne; 
        
    }

    </style>
    '''
    # }

    # @font-face {
    # font-family: 'formula1';'''+f'''
    # src: url(data:application/x-font-woff;charset=utf-8;base64,{bold} )'''+''' format('woff');
    # font-weight: normal;
    # font-style: normal;
    # }

    # @font-face {
    # font-family: 'formula1';'''+f'''
    # src: url(data:application/x-font-woff;charset=utf-8;base64,{black} )'''+''' format('woff');
    # font-weight: normal;
    # font-style: normal;

    # }

    st.markdown(style,unsafe_allow_html=True)

    # title 
    st.markdown(f'''<center><h1 style="font-family:syne; font-size:50px; font-weight:800;text-shadow: 4px 2px white; background-color: #e00400; border-radius: 10px;">The <img src='data:image/png;base64,{img_to_bytes('./assets/f1.png')}' class='img-fluid' width=120>   Web-Paddock </h1></center>''',unsafe_allow_html=True)
    st.markdown('***')

    # Sidebar title 
    st.sidebar.markdown(f'''<h2 style="font-family:syne, syne; font-weight:bold; font-size:27px;">The Control Deck <img src='data:image/png;base64,{img_to_bytes('./assets/steering-wheel.png')}' class='img-fluid' width=35 ></h2>''',unsafe_allow_html=True)
    st.sidebar.markdown('***')




    # Fetch Dependables
    api_elements = instantiate_API_keys()
    yearwise_rounds = load_miscellaneous_data()

    # st.write(yearwise_rounds.loc[1950,'Rounds'])


    # Categories -- Change Order to, About, Current Season, Previous Season, The F1 Glossary
    # category = st.sidebar.selectbox('Select', ['Current Season', 'Previous Seasons', 'About', 'The F1 Glossary'])

    # dashboard_type = st.sidebar.selectbox('Who are you?', ['Home Page','Real-Time', 'Historic Summary','The Machinery!','Fun Trivia','Testing Zone'],key='dashboard-type')
    dashboard_type = st.sidebar.selectbox('Control the Data.', ['Home Page','Real-Time', 'Historic Summary','The Machinery!','Fun Trivia'],key='dashboard-type')
    # st.sidebar.info('*This decides how the dashboard is organised with the results. A Fan can see all the session level details, sponsor will be able to get analytical reports summarising the performance of a team for them to take decisions on whether to sponsor or pass.*')
    st.sidebar.markdown('***')

    if dashboard_type == 'Real-Time':

        category = st.sidebar.selectbox('Select Timeline', ['Current Season', 'Previous Seasons'])
        
        
        if category == 'Previous Seasons':


            # st.sidebar.markdown('***')

            # YEAR -- input

            
            
            st.sidebar.markdown(f'''<p style='font-weight:bold;'><u>Parameters</u></p>''',unsafe_allow_html=True)
            year = st.sidebar.slider('Select Year', min_value=2006, max_value=datetime.now().year - 1, value=datetime.now().year - 1)

            # fetch circuits
            circuits_cdf, circuits_rdf = fetch_circuits_data(year)
                
            # collecting data
            event_names, event_schedule = fetch_event_schedule(int(year))
            

            

            radios = st.sidebar.radio('Data', ['Schedule', 'Grand Prix Analysis','Points Table'])
            st.sidebar.markdown('***')


            if radios == 'Schedule':

                st.markdown(f'''<p style="font-size:30px; font-weight:bold; font-family:syne;"> <span style='color:darkblue; font-weight:900;'>{year}</span> Race Schedule | Grand Prix List </p>''',unsafe_allow_html=True)
                st.markdown('***')

                # Events for the year 
                st.markdown(f'''<p style='font-weight:bold;'></p>''',unsafe_allow_html=True)
                display_schedule(year, circuits_cdf, circuits_rdf)


            elif radios == 'Grand Prix Analysis':

                # Events for the year 
                st.sidebar.markdown(f'''<p style='font-weight:bold;'>Grand Prix List | <span style='color:darkblue;'>{year}</span></p>''',unsafe_allow_html=True)

                
                # EVENT NAME
                event = st.sidebar.selectbox('Select Event', event_names)

                if not event == "List of Grand Prix's":
                    
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
                        st.markdown(f'''<center><h4 style="font-family:syne;">{summary_type}</h4></center>''',unsafe_allow_html=True)
                        st.markdown('***')

                        # Grand Prix Title
                        try:
                            flag = pycountry.countries.search_fuzzy(package["Country"].lower())[0].flag
                        except:
                            flag = pycountry.countries.search_fuzzy(missing_endpoints[package["Country"]])[0].flag

                        st.markdown(f'''<p style="font-size:30px; font-weight:800; font-family:syne;"> {flag} <u>{package["EventName"]}</u>  |  <span style="font-size:23px;">{date_modifier(package["EventDate"])}</span></p>''',unsafe_allow_html=True)
                    
                    
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
                        try:
                            flag = pycountry.countries.search_fuzzy(package["Country"].lower())[0].flag
                        except:
                            flag = pycountry.countries.search_fuzzy(missing_endpoints[package["Country"]])[0].flag

                        st.markdown(f'''<p style="font-size:28px; font-weight:800; font-family:formula1, syne;"> {flag} <u>{package["EventName"]}</u>  |  <span style="font-size:23px;">{date_modifier(package["EventDate"])}</span></p>''',unsafe_allow_html=True)
                    


                        st.markdown(f'<p style="font-size:15px;font-family:formula1, syne; font-weight:bold;">Race Format - {package["EventFormat"].capitalize()}</p>',unsafe_allow_html=True)
                        sessions_list.insert(0,'Select Session')
                        session_select = st.selectbox('Select Session', sessions_list, key='sessions')
                        st.markdown('***')
                        
                        # bypassing the first element problem
                        if session_select != 'Select Session':

                            session_results, session_obj = load_session_data(year, event, session_select)
                            
                            
                            if session_select == 'Qualifying':

                                # Data Collection
                                summarised_results = session_results.copy(deep=True)
                                summarised_results = pd.DataFrame(summarised_results)
                                summarised_results = summarised_results.drop(['Time','Status','Points'], axis=1)
                                summarised_results = summarised_results.fillna('0')

                                qualifying(summarised_results, year, session_obj)


                            elif session_select in ['Practice 1','Practice 2','Practice 3','Sprint']:
            
                                st.warning('In Development ⌛')
                                st.markdown('***')

                            elif session_select == 'Race':


                                cols = st.columns([6,3])
                                placeholder = cols[1].empty()
                                # placeholder.selectbox('?',['?'])
                                mode = cols[0].selectbox('Select Mode', ['Mode?','Driver Analysis','Team Analysis', 'Race Standings','Retirements'])

                                if year >= 2018:

                                    # result summary
                                    results = session_obj.results
                                    results = results.drop(['Q1','Q2','Q3','FullName'],axis=1)
                                    results = results.fillna('0')
                                    retired = results[(results['Status'] != 'Finished') & (results['Status'] != '+1 Lap') ]
                                    driverlaps = session_obj.laps
                                
                                    if mode == 'Driver Analysis':

                                        
                                            
                                            ab_list = list(driverlaps['Driver'].unique())
                                            driver_AB = placeholder.selectbox('Driver',ab_list) # selection
                                            driver_data = driverlaps.pick_driver(driver_AB)
                                            driver_data = driver_data.reset_index()
                                            driver_data = driver_data.drop(0)
                                            total_laps = int(driverlaps['LapNumber'].max())

                                            driver_race_analysis(year, event, driverlaps, driver_AB, driver_data, total_laps, 'previous')

                                    elif mode == 'Team Analysis':

                                        team_data = session_obj.results
                                        team_data = team_data.set_index('TeamName')
                                        team_data = team_data.drop(team_data.columns[7:], axis=1)

                                        team = placeholder.selectbox('Select Team', team_data.index.unique(),key='teams')   
                                        DN1, BN1, AB1, TC1, FN1, LN1, FuN1 = team_data.loc[team].reset_index(drop=True).loc[0,:]
                                        DN2, BN2, AB2, TC2, FN2, LN2, FuN2 = team_data.loc[team].reset_index(drop=True).loc[1,:]
                                        TN = team

                                        st.markdown(f'''<h3 style="font-family:formula1, syne; font-weight:800; color:#{TC1};"><center>{TN}</center></h3>''',unsafe_allow_html=True)

                                        

                                        st.markdown('***')
                                        st.markdown('''<center><span style='font-weight:800; font-size:28px;'>Race Strategy</span></center>''', unsafe_allow_html=True)
                                        st.markdown('***')
                                        total_laps = int(driverlaps['LapNumber'].max())

                                        Boolean = [True,False]
                                        data_gen = (y for y in Boolean)

                                        for driver_AB in [AB1, AB2]:
                                            driver_data = driverlaps.pick_driver(driver_AB)
                                            driver_info = session_obj.get_driver(driver_AB)
                                            dn, bn, ab, tn, tc  = driver_info[:5]
                                            st.markdown(f'''<h4 style="font-family:formula1, syne; font-weight:800;">{bn} ({ab}) <sub style='color:#{tc}'>{tn}</sub></h4>''',unsafe_allow_html=True)
                                            # st.subheader(driver_AB) # subheader -- racer name 


                                            strategy, lap_retired = fetch_strategy(driver_data)
                                            presets = [strategy, lap_retired, total_laps]  
                                            display_strategy(presets, mode='previous',team=next(data_gen))
                                                

                                        
                                        st.markdown('***')

                                        col1, col2 = st.columns(2)
                                        with col1:
                                            driver_data = driverlaps.pick_driver(AB1)
                                            driver_data = driver_data.reset_index()
                                            driver_race_analysis(year, event, driverlaps, AB1, driver_data, total_laps,'previous',team=True,preset=[TC1,AB1])
                                        
                                        with col2:
                                            driver_data = driverlaps.pick_driver(AB2)
                                            driver_data = driver_data.reset_index()
                                            driver_race_analysis(year, event, driverlaps, AB2, driver_data, total_laps,'previous',team=True,preset=[TC2,AB2])
                                        

                                    
                                    
                                    elif mode == 'Race Standings':
                                        
                                    
                                        top3 = results.iloc[:3,:]
                                        results = results.iloc[3:,:]
                                        results = results.reset_index(drop=True)

                                        display_race_standings(results, top3)

                                    elif mode == 'Retirements':

                                        total_laps = session_obj.laps['LapNumber'].max()


                                        st.markdown('***')
                                        st.markdown('''<center><span style='font-weight:800; font-size:28px;'>Retirements</span></center>''', unsafe_allow_html=True)
                                        st.markdown('***')

                                        if retired.empty:
                                            st.info('No Retirements')

                                        else:
                                            for index in range(len(retired)):
                                                dn, bn, ab, tn, tc, fn, ln, pn, gp, time, status, points = retired.iloc[index,:]
                                                retired_lap = session_obj.laps.pick_driver(ab).iloc[-1:,:]['LapNumber'].values[0]
                                                if  retired_lap < total_laps and (total_laps-retired_lap) !=1:
                                                    pass
                                                st.markdown(f'''> <h2><span style='font-family:syne; font-size:70px; font-weight:800;'>{dn} </span>{fn} <b>{ln}</b> <sub style='color:#{tc}'>{tn}</sub> <p><b>{status}</b> (Problem / Failure / Occured) — Retired at <b>Lap {retired_lap}</b></p></h2>''',unsafe_allow_html=True)

                                        st.markdown('***')



                                    
                                else:
                                    st.warning("The API doesn't hold the telemetry data for the years before 2018.")

                                        
            
            
            elif radios == 'Points Table':
                st.markdown('''<center><span style='font-weight:800; font-size:28px;'>Driver Standings & Constructors</span></center>''', unsafe_allow_html=True)
                st.markdown('***')
                try:
                    constructors, driver_standings = parse_race_points(year)
                    
                    # driver standings
                    st.markdown(f'''<span style="font-family:syne; font-size:25px">Driver Standings <span style='font-weight:900;color:darkblue;'>{year}</span></span>''',unsafe_allow_html=True)
                    y = pd.DataFrame(driver_standings).T.copy(deep=True)
                    y['Full Name'] = pd.DataFrame(driver_standings).T['givenName'] +' '+ pd.DataFrame(driver_standings).T['familyName']
                    y['Driver Code'] = y['code'] + ' ' + y['permanentNumber']
                    y = y.drop(['givenName','familyName','code','permanentNumber'],axis=1)
                    columns = ['Position','Points','Wins','Team','Nationality','Full Name','Driver Code']
                    y.columns = columns
                    y = y[['Full Name','Driver Code','Team','Points','Wins','Nationality','Position']]
                    y = y.set_index('Position')
                    
                    st.dataframe(y,2000, 2000)

                    st.markdown('***')
                    st.markdown(f'''<span style="font-family:syne; font-size:25px">Constructors <span style='font-weight:900;color:darkblue;'>{year}</span></span>''',unsafe_allow_html=True)
                    y = constructors = pd.DataFrame(constructors).T.copy(deep=True)
                    columns = ['Position','Points','Wins','Team','Nationality']
                    y.columns = columns
                    y = y[['Position','Team','Points','Wins','Nationality']].set_index('Position')
                    
                    cols = st.columns([2,6,2])
                    cols[1].dataframe(y, 1000,1000)

                except:
                    st.warning('Data Descrepancy!, Certain elements of the Data are not Preserved by the API')
                    constructors_url = f'https://www.formula1.com/en/results.html/{year}/team.html'
                    driver_standing_url = f'https://www.formula1.com/en/results.html/{year}/drivers.html'
                    st.markdown(f''' <span style='font-weight:900;color:darkblue;font-size:30px;'> {year}</span>, <span style="font-family:syne; font-size:25px">Driver Standings <a href='{driver_standing_url}'>Refer this Source.</a></span>''',unsafe_allow_html=True)
                    st.markdown(f'''<span style='font-weight:900;color:darkblue;font-size:30px;'> {year}</span>, <span style="font-family:syne; font-size:25px">Constructors Standings <a href='{constructors_url}'>Refer this Source.</a></span>''',unsafe_allow_html=True)
                    
                
                
            

                                
                            
                        
                    


        
        elif category == 'Current Season':

            




            st.sidebar.markdown('***')

            current_year = datetime.now().strftime('%Y')
            current_year = int(current_year)
            

            circuits_cdf, circuits_rdf = fetch_circuits_data(current_year)

            radios = st.sidebar.radio('Data',['Schedule','Grand Prix Analysis','Points Table'], key='current')
            st.sidebar.markdown('***')

            if radios == 'Schedule':
                # Events for the year 
                st.markdown(f'''<p style="font-size:30px; font-weight:bold; font-family:syne;"> <span style='color:darkblue; font-weight:900;'>{current_year}</span> Race Schedule | Grand Prix List </p>''',unsafe_allow_html=True)
                st.markdown("***") 
                display_schedule(current_year, circuits_cdf, circuits_rdf)

            elif radios == 'Grand Prix Analysis':
                # GP analysis
        
                # Analytics for the races that happened
                current_event = fastf1.get_event_schedule(current_year)
                conditional = (current_event['EventDate'] <= datetime.now()) | (current_event['Session1Date'] <= datetime.now()) | (current_event['Session2Date'] <= datetime.now()) | (current_event['Session3Date'] <= datetime.now()) | (current_event['Session4Date'] <= datetime.now()) | (current_event['Session4Date'] <= datetime.now())
                index = current_event[conditional].index
                current_event = current_event.loc[index,:]

                event_names = current_event['EventName'].to_list()
                event_names = event_names[::-1]
                event_names.insert(0, "List of Completed Grand Prixs'")

                
                event = st.sidebar.selectbox('Select Event', event_names)

                if not event == "List of Completed Grand Prixs'":

                    event_data = current_event[current_event['EventName'] == event].T
                    
                    # country name
                    country = event_data.loc['Country'].values[0]

                    try:
                        circuit = circuits_rdf.loc[event, 'Circuits']
                        locality = circuits_rdf.loc[event, 'Localities']
                    except:
                        circuit = circuits_cdf.loc[country, 'Circuits']
                        locality = circuits_cdf.loc[country, 'Localities']

                    # packaging event summarised-information
                    package = {}
                    items = current_event.columns[:-1]
                    for item in items:
                        package[item] = event_data.loc[item].values[0]

                    sessions_list = [package[x] for x in ['Session'+str(i) for i in range(1,6)]]

                    st.markdown(f'''<center><h4 style="font-family:formula1, syne;">Session Analysis</h4></center>''',unsafe_allow_html=True)
                    st.markdown('***')

                    # Grand Prix Title
                    try:
                        flag = pycountry.countries.search_fuzzy(package["Country"].lower())[0].flag
                    except:
                        flag = pycountry.countries.search_fuzzy(missing_endpoints[package["Country"]])[0].flag

                    st.markdown(f'''<p style="font-size:30px; font-weight:bold; font-family:formula1, syne;"> {flag}  <u>{package["EventName"]}</u>  |  <span style="font-size:23px;">{date_modifier(package["EventDate"])} <br><span style='font-family:syne; '>{circuit}, {locality}</span> </span> </p>''',unsafe_allow_html=True)
                    # st.markdown(f'''<p style="font-size:28px; font-weight:bold; font-family:formula1, syne;"> <img src="https://countryflagsapi.com/png/{package["Country"]}" width="50">  <u>{package["EventName"]}</u>  |  <span style="font-size:23px;">{date_modifier(package["EventDate"])}</span></p>''',unsafe_allow_html=True)
                    st.markdown(f'<p style="font-size:15px;font-family:formula1, syne; font-weight:bold;">Race Format - {package["EventFormat"].capitalize()}</p>',unsafe_allow_html=True)

                    # session select
                    sessions_list.insert(0,'Select Session')
                    session_select = st.selectbox('Select Session', sessions_list, key='sessions')
                    st.markdown('***')


                    # Bypassing the first element problem
                    if session_select != 'Select Session':

                        session_results, session_obj = load_session_data(current_year, event, session_select)
                        
                        if session_select == 'Qualifying':

                            # Data Collection
                            summarised_results = session_results.copy(deep=True)
                            summarised_results = pd.DataFrame(summarised_results)
                            summarised_results = summarised_results.drop(['Time','Status','Points'], axis=1)
                            summarised_results = summarised_results.fillna('0')

                            
                            qualifying(summarised_results,current_year, session_obj)

                        elif session_select == 'Race':


                            cols = st.columns([6,3])
                            placeholder = cols[1].empty()
                            # placeholder.markdown('<-')
                            mode = cols[0].selectbox('Select Mode', ['Mode?','Driver Analysis','Team Analysis', 'Race Standings','Retirements'])

                            # result summary
                            results = session_obj.results
                            results = results.drop(['Q1','Q2','Q3','FullName'],axis=1)
                            results = results.fillna('0')
                            retired = results[(results['Status'] != 'Finished') & (results['Status'] != '+1 Lap') ]
                            driverlaps = session_obj.laps

                            if current_year >= 2018:
                                if mode == 'Driver Analysis':
                                    
                                    
                                    ab_list = list(driverlaps['Driver'].unique())
                                    driver_AB = placeholder.selectbox('Driver',ab_list)
                                    # analysis_type = st.checkbox('Comparative?')
                                    # selection
                                    driver_data = driverlaps.pick_driver(driver_AB)
                                    driver_data = driver_data.reset_index()
                                    total_laps = int(driverlaps['LapNumber'].max())

                                    driver_race_analysis(current_year, event, driverlaps, driver_AB, driver_data, total_laps,'current')

                            
                                elif mode == 'Team Analysis':

                                    team_data = session_obj.results
                                    team_data = team_data.set_index('TeamName')
                                    team_data = team_data.drop(team_data.columns[7:], axis=1)

                                    team = placeholder.selectbox('Select Team', team_data.index.unique(),key='teams')   
                                    DN1, BN1, AB1, TC1, FN1, LN1, FuN1 = team_data.loc[team].reset_index(drop=True).loc[0,:]
                                    DN2, BN2, AB2, TC2, FN2, LN2, FuN2 = team_data.loc[team].reset_index(drop=True).loc[1,:]
                                    TN = team

                                    st.markdown(f'''<h3 style="font-family:formula1, syne; font-weight:800; color:#{TC1};"><center>{TN}</center></h3>''',unsafe_allow_html=True)

                                    

                                    st.markdown('***')
                                    st.markdown('''<center><span style='font-weight:800; font-size:28px;'>Race Strategy</span></center>''', unsafe_allow_html=True)
                                    st.markdown('***')
                                    total_laps = int(driverlaps['LapNumber'].max())

                                    Boolean = [True,False]
                                    data_gen = (y for y in Boolean)

                                    for driver_AB in [AB1, AB2]:
                                        driver_data = driverlaps.pick_driver(driver_AB)
                                        driver_info = session_obj.get_driver(driver_AB)
                                        dn, bn, ab, tn, tc  = driver_info[:5]
                                        st.markdown(f'''<h4 style="font-family:formula1, syne; font-weight:800;">{bn} ({ab}) <sub style='color:#{tc}'>{tn}</sub></h4>''',unsafe_allow_html=True)
                                        # st.subheader(driver_AB) # subheader -- racer name 

                                    

                                        if mode == 'previous':
                                            strategy, lap_retired = fetch_strategy(driver_data)
                                            presets = [strategy, lap_retired, total_laps]  
                                            display_strategy(presets, mode='previous',team=next(data_gen))
                                            
                                            
                                        else:
                                            compounds, lapschanged, pairs = fetch_strategy(driver_data,total_laps, mode='current') 
                                            strategy, lap_retired = fetch_strategy(driver_data)
                                            if event == 'Monaco Grand Prix':
                                                presets = [compounds, lapschanged, pairs, total_laps] 
                                                display_strategy(presets, mode='current',team=next(data_gen))
                                            else:
                                                presets = [strategy, lap_retired, total_laps]  
                                                display_strategy(presets, mode='previous',team=next(data_gen))

                                    
                                    st.markdown('***')

                                    col1, col2 = st.columns(2)
                                    with col1:
                                        driver_data = driverlaps.pick_driver(AB1)
                                        driver_data = driver_data.reset_index()
                                        driver_race_analysis(current_year, event, driverlaps, AB1, driver_data, total_laps,'current',team=True,preset=[TC1,AB1])
                                    
                                    with col2:
                                        driver_data = driverlaps.pick_driver(AB2)
                                        driver_data = driver_data.reset_index()
                                        driver_race_analysis(current_year, event, driverlaps, AB2, driver_data, total_laps,'current',team=True,preset=[TC2,AB2])
                                    

                                    

                                elif mode == 'Race Standings':
                                    
                                    
                                    top3 = results.iloc[:3,:]
                                    results = results.iloc[3:,:]
                                    results = results.reset_index(drop=True)

                                    display_race_standings(results, top3)
                                    
                                
                                elif mode == 'Retirements':
                                
                                    total_laps = session_obj.laps['LapNumber'].max()
                                    st.markdown('***')
                                    st.markdown('''<center><span style='font-weight:800; font-size:28px;'>Retirements</span></center>''', unsafe_allow_html=True)
                                    st.markdown('***')

                                    if retired.empty:
                                        st.info('No Retirements')

                                    else:
                                        for index in range(len(retired)):
                                            dn, bn, ab, tn, tc, fn, ln, pn, gp, time, status, points = retired.iloc[index,:]
                                            retired_lap = session_obj.laps.pick_driver(ab).iloc[-1:,:]['LapNumber'].values[0]
                                            if  retired_lap < total_laps and (total_laps-retired_lap) !=1:
                                                pass
                                            st.markdown(f'''> <h2><span style='font-family:syne; font-size:70px; font-weight:800;'>{dn} </span>{fn} <b>{ln}</b> <sub style='color:#{tc}'>{tn}</sub> <p><b>{status}</b> (Problem / Failure / Occured) — Retired at <b>Lap {retired_lap}</b></p></h2>''',unsafe_allow_html=True)

                                    st.markdown('***')
                            
                            else:
                                st.warning("The API doesn't hold the telemetry data for the years before 2018.")

                
                        elif session_select in ['Practice 1','Practice 2','Practice 3','Sprint']:

                            st.warning('In Development ⌛')
                            st.markdown('***')

                
                
                
                
            elif radios == 'Points Table':
                st.markdown('''<center><span style='font-weight:800; font-size:28px;'>Driver Standings & Constructors</span></center>''', unsafe_allow_html=True)
                st.markdown('***')

                try:
                    constructors, driver_standings = parse_race_points(current_year)
                    # driver standings
                    st.markdown(f'''<span style="font-family:syne; font-size:25px">Driver Standings <span style='font-weight:900;color:darkblue;'>{current_year}</span></span>''',unsafe_allow_html=True)
                    y = pd.DataFrame(driver_standings).T.copy(deep=True)
                    y['Full Name'] = pd.DataFrame(driver_standings).T['givenName'] +' '+ pd.DataFrame(driver_standings).T['familyName']
                    y['Driver Code'] = y['code'] + ' ' + y['permanentNumber']
                    y = y.drop(['givenName','familyName','code','permanentNumber'],axis=1)
                    columns = ['Position','Points','Wins','Team','Nationality','Full Name','Driver Code']
                    y.columns = columns
                    y = y[['Full Name','Driver Code','Team','Points','Wins','Nationality','Position']]
                    y = y.set_index('Position')
                    
                    st.dataframe(y,2000, 2000)

                    st.markdown('***')
                    st.markdown(f'''<span style="font-family:syne; font-size:25px">Constructors <span style='font-weight:900;color:darkblue;'>{current_year}</span></span>''',unsafe_allow_html=True)
                    y = constructors = pd.DataFrame(constructors).T.copy(deep=True)
                    columns = ['Position','Points','Wins','Team','Nationality']
                    y.columns = columns
                    y = y[['Position','Team','Points','Wins','Nationality']].set_index('Position')
                    
                    cols = st.columns([2,6,2])
                    cols[1].dataframe(y, 1000,1000)


                except:
                    st.warning('Data Descrepancy!, Certain elements of the Data are not Preserved by the API')
                    

               



                                
                            

                
    elif dashboard_type == 'Historic Summary':    


        era = st.sidebar.selectbox('Select a Formual One Era to begin Analysis',['Turbo Hybrid Era','Turbo Era'],key='eras')

       

        if era == 'Turbo Hybrid Era':


            expander = st.sidebar.expander('Info', expanded=True)
            info = '''The turbo-hybrid era, which is where Formula 1 has gone, has been dubbed since 2014. And it's appropriate since this is the first time in the history of the sport that internal combustion engine and hybrid technology have been combined (ICE).'''
            expander.markdown(f'_{info}_')        
            
            team_colors = fastf1.plotting.TEAM_COLORS

            team_name_corr = {'Apine F1 Team':'alpine'}

            # st.write(team_colors)

            st.markdown(f'''<center><h2 style="font-family:formula1, syne; font-weight:800">Turbo Hybrid Era</h2></center>''',unsafe_allow_html=True)
            slot = st.empty()
            st.markdown('***')
            
        
            if st.checkbox('Current Season',True):
                year = 2022
                slot.markdown(f'''<center><h4 style="font-family:formula1, syne;">2022</h4></center>''',unsafe_allow_html=True)
            else:
                year = st.number_input('Select Year:',max_value=2022, min_value=2014)
                slot.markdown(f'''<center><h3 style="font-family:formula1, syne;">{year}</h3></center>''',unsafe_allow_html=True)

            cars_dict = fetch_carimgs()
            cars_names = pd.read_html('https://racingnews365.com/f1-2022-car-names')[0]
            car_names = dict(cars_names[['Team','Car Name']].values)
            rounds = load_rounds()

            perf_category = st.selectbox('Select Summary Type', ['Choose','Entire Timeline','Yearly'], key='datetype-standings')

            if not perf_category == 'Choose':

                if perf_category == "Entire Timeline":
                    
                   
                    constructor_standings = fetch_constructorStandings()
                    driver_standings = fetch_driverstandings()

                    # st.write(constructor_standings)
                    # st.write(driver_standings)

                    teams = constructor_standings.loc[year]['Constructor'].unique()


                    team = st.selectbox('Choose a Team', teams, key='team-selection')
                    
                    try:
                        st.markdown(f'''<center><h2 style="font-family:formula1, syne; font-weight:800; color:{team_colors[team.lower()]}">{team}</h2></center>''',unsafe_allow_html=True)
                    
                    except:
                        st.markdown(f'''<center><h2 style="font-family:formula1, syne; font-weight:800; color:black">{team}</h2></center>''',unsafe_allow_html=True)
                    st.markdown('***')
                    
                    all_drivers = {}
                    for year in range(2014, 2023):
                        x = driver_standings.loc[year]
                        all_drivers[year] = list(x[x['Constructors']==team]['fullname'].values)

                    
                    #ADDRESS THE EXITING TEAMS PROBLEM

                    st.markdown(f'''<center><h3 style="font-family:formula1, syne; color:"><span style='font-weight:800;'>{team}</span> Drivers Over the Years</h3></center>''',unsafe_allow_html=True)
                    cols = st.columns(len(list(range(2014,2023))))
                    for col, year in zip(cols, range(2014, 2023)):
                        try:
                            dr1, dr2 = all_drivers[year]
                        except:
                            try:
                                dr1, dr2, dr3 = all_drivers[year]
                            except:
                                dr1, dr2 = 'b','b'
                        if not dr1 == 'b' and not dr2 =='b':
                            col.markdown(f'<h3>{year}</h3>',unsafe_allow_html=True)
                            col.markdown(f'''> {dr1} {dr2}''')
                        else:
                            break

                    points = []
                    for year in range(2014,2023):
                        x = constructor_standings.loc[year].reset_index()
                        points.append(x[x['Constructor']==team]['points'].values[0])
                    points = list(map(int, list(map(float, points))))

                    years= list(range(2014,2023))

                    wins = []
                    for year in range(2014,2023):
                        x = constructor_standings.loc[year].reset_index()
                        wins.append(x[x['Constructor']==team]['wins'].values[0])
                    wins = list(map(int, list(map(float, wins))))

                   
                    fig = go.Figure([go.Bar(x=years, y=points,  )])
                    fig.update_traces(marker_color=team_colors[team.lower()], marker_line_color='black',
                                    marker_line_width=1.5, )
                    

                    fig.update_layout(
                        title=f"Points attained in the Turbo Hybrid Era - {team}",
                        title_x=0.5,
                        xaxis_title="Years",
                        yaxis_title="Points",
                        legend_title="Legend Title",
                        font=dict(
                            family="Syne",
                            size=11,
                            color="black"
                        )
                    )

                    st.plotly_chart(fig)

                    fig = go.Figure([go.Bar(x=years, y=wins,  )])
                    fig.update_traces(marker_color=team_colors[team.lower()], marker_line_color='black',
                                    marker_line_width=1.5, )
                    

                    fig.update_layout(
                        title=f"All Podiums in the Turbo Hybrid Era - {team}",
                        title_x=0.5,
                        xaxis_title="Years",
                        yaxis_title="Podiums",
                        legend_title="Legend Title",
                        font=dict(
                            family="Syne",
                            size=11,
                            color="black"
                        )
                    )

                    st.plotly_chart(fig)

                    


                
                elif perf_category == 'Yearly':

                            
                    constructor_standings = fetch_constructorStandings(rounds=rounds, roundwise=True)
                    driver_standings = fetch_driverstandings(rounds=rounds, roundwise=True)

                    # st.write(constructor_standings)
                    # st.write(driver_standings)

                    teams = constructor_standings.loc[year].loc[1]['Constructor'].to_list()
                    drivers = driver_standings.loc[year].loc[1]['fullname'].to_list()
                    codes = driver_standings.loc[year].loc[1]['code'].to_list()
                    
                    cols = st.columns([4,2])
                    slot = cols[1].empty()
                    diff = cols[0].selectbox('Differentiator',['Team','Driver'],key='differentiator')

                    if diff == 'Team':
                        team = slot.selectbox('Select Team',teams,key='teams-diff')

                        st.markdown('***')
                        

                        if year == 2022:
                            cols = st.columns([5,5])
                            cols[0].markdown(f'''<h2 style="font-family:formula1, syne; font-weight:800; color:{team_colors[team.lower()]}">{team}</h2>''',unsafe_allow_html=True)
                            cols[0].image(cars_dict[team],caption=f'{team} - {car_names[team]}')
                            
                            pos, points = pos, points = fetch_position_rank(constructor_standings, team=team, year=year)
                            cols[1].markdown(f'''> <h2><span style='font-family:syne; font-size:70px; font-weight:800;'>Position {pos} </span> <b></b> <sub style='color:#{team_colors[team.lower()]}'></sub> Points {points}</b></p></h2>''',unsafe_allow_html=True)

                            dr_standings = fetch_position_rank(driver=driver_standings, team=team,individual=True,year=year)
                            names = list(dr_standings.keys())
                                
                            pos0, code0, points0, wins0 = dr_standings[names[0]][0]
                            pos1, code1, points1, wins1 = dr_standings[names[1]][0]
                            
                            cols[0].markdown(f'''> <h2><span style='font-family:syne; font-size:30px; font-weight:800; color:{team_colors[team.lower()]}'> {names[0]} <sub>{code0}</sub> </span> <b></b></sub> <br> Points {points0}</b><br> Wins {wins0}</p></h2>''',unsafe_allow_html=True)
                            cols[1].markdown(f'''> <h2><span style='font-family:syne; font-size:30px; font-weight:800; color:{team_colors[team.lower()]}'> {names[1]} <sub>{code1}</sub> </span> <b></b></sub> <br> Points {points1}</b><br> Wins {wins1}</p></h2>''',unsafe_allow_html=True)



                            try:
                                st.pyplot(bump_plot(driver_standings, mode='team',team=team,year=year,team_color=team_colors[team.lower()]))
                            except:
                                st.pyplot(bump_plot(driver_standings, mode='team',team=team,year=year,team_color='black'))
                            
                            st.pyplot(bump_plot(driver_standings, mode='overall',year=year))


                
                        else:
                            cols = st.columns([5,5])
                            cols[0].markdown(f'''<h2 style="font-family:formula1, syne; font-weight:800; color:">{team}</h2>''',unsafe_allow_html=True)
                            team_color='black'


                            pos, points = pos, points = fetch_position_rank(constructor_standings, team=team, year=year)
                            cols[1].markdown(f'''> <h2><span style='font-family:syne; font-size:70px; font-weight:800;'>Position {pos} </span> <b></b> <sub style='color:black'></sub> Points {points}</b></p></h2>''',unsafe_allow_html=True)

                            dr_standings = fetch_position_rank(driver=driver_standings, team=team,individual=True,year=year)
                            names = list(dr_standings.keys())
                                
                            pos0, code0, points0, wins0 = dr_standings[names[0]][0]
                            pos1, code1, points1, wins1 = dr_standings[names[1]][0]
                            
                            cols[0].markdown('')
                            cols[0].markdown('')
                            cols[0].markdown('')
                            cols[0].markdown('')
                            cols[0].markdown('')
                            cols[0].markdown('')
                            cols[0].markdown('')
                            cols[0].markdown('')
                            cols[0].markdown('')
                            cols[0].markdown(f'''> <h2><span style='font-family:syne; font-size:30px; font-weight:800; color:black'> {names[0]} <sub>{code0}</sub> </span> <b></b></sub> <br> Points {points0}</b><br> Wins {wins0}</p></h2>''',unsafe_allow_html=True)
                           
                            cols[1].markdown(f'''> <h2><span style='font-family:syne; font-size:30px; font-weight:800; color:black'> {names[1]} <sub>{code1}</sub> </span> <b></b></sub> <br> Points {points1}</b><br> Wins {wins1}</p></h2>''',unsafe_allow_html=True)


                            try:
                                st.pyplot(bump_plot(driver_standings, mode='team',team=team,year=year,team_color=team_colors[team.lower()]))
                            except:
                                st.pyplot(bump_plot(driver_standings, mode='team',team=team,year=year,team_color='black'))
                            st.pyplot(bump_plot(driver_standings, mode='overall',year=year))

                        const_overall = fetch_constructorStandings()
                        driver_overall = fetch_driverstandings()

                        cols = st.columns(2)
                        cols[0].markdown(f'*Driver Standings {year}*')
                        cols[0].dataframe(driver_overall.loc[year])
                        cols[1].markdown(f'*Constructor Standings {year}*')
                        cols[1].dataframe(const_overall.loc[year])

                    else:
                        st.warning('*Beta Testing*')


            else:
                pass

        else:
            st.markdown(f'''<center><h2 style="font-family:formula1, syne; font-weight:800">Pre-Turbo Hybrid Era (2006 to 2013)</h2></center>''',unsafe_allow_html=True)
            st.markdown('***')
            st.warning('In Development ⌛')
    


    elif dashboard_type == "The Machinery!":
        st.markdown(f'''<center><h2 style="font-family:formula1, syne; font-weight:800">The Evolution of Formula 1 Machinery</h2></center>''',unsafe_allow_html=True)
        st.markdown('***')

        cimg_df = load_carimage_data()
        cspecs_df = load_carspecs()
        # st.write(cspecs_df)

        year = st.number_input('Control the Timeline', 2012,2021,key='year-slider')

        car, link, team = cimg_df[cimg_df['Unnamed: 0']==year]['Car'], cimg_df[cimg_df['Unnamed: 0']==year]['Image-Link'], cimg_df[cimg_df['Unnamed: 0']==year]['team']
        y_dict = cspecs_df[year]
        st.markdown(f'''<center><h2 style="font-family:formula1, syne; font-weight:800">{year}</h2></center>''',unsafe_allow_html=True)
        st.markdown('***')
        # st.markdown(f'''<center><h4 style="font-family:formula1, syne; font-weight:800">Gallery</h24></center>''',unsafe_allow_html=True)
        for c, l, t in zip(car, link, team):
            st.image(l)
            st.markdown(f'<center><b>{c}, {t}</b><c/enter>',unsafe_allow_html=True)
            st.markdown('')
            for char in y_dict[c].keys():
                exp = st.expander(char)
                t_dict = fabricate_dict(y_dict[c][char])
                if type(t_dict) == str:
                    exp.markdown(f'**{t_dict}**')
                else:
                    for key, value in zip(t_dict.keys(),t_dict.values()):
                        exp.markdown(f'<span style="font-weight:800;font-family:menlo;">{key}</span>: <span style="font-family:menlo;">{value}</span>',unsafe_allow_html=True)
            st.markdown('***')
                

    elif dashboard_type == "Fun Trivia":
        st.markdown(f'''<center><h2 style="font-family:formula1, syne; font-weight:800">Fun Trivia!</h2></center>''',unsafe_allow_html=True)
        st.markdown('***')
        cols = st.columns([7,3])
        cols[0].markdown(f'''<center><h4 style="font-family:formula1, syne; font-weight:800">Who won the Year you were born?</h4></center>''',unsafe_allow_html=True)
        cols[1].number_input('Year of Birth?',1950,datetime.now().year,key='year-of-birth')

        st.markdown("***")
        st.warning('In Development ⌛')

    elif dashboard_type == 'Home Page':
        about_cs()
        attribute()
    
    # elif dashboard_type == 'Testing Zone':
    #     rounds = load_rounds()
    #     aggregate = fetch_constructorStandings([2012, 2023], roundwise=True, rounds=rounds)

    #     st.write(aggregate)
    #     aggregate.to_csv('ForCarPERC.csv')



            



    
    
   






   
