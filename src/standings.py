
import pandas as pd 
import numpy as np
import requests

    
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
        
        
# const = fetch_constructorStandings(rounds=rounds, roundwise=True)
        

# const = fetch_constructorStandings()
        

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

            

            
        

# drivers = fetch_driverstandings(rounds=rounds,roundwise=True)

# fetch_driverstandings()