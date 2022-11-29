import requests 
from bs4 import BeautifulSoup 
import pandas as pd 


    
def getdata(url): 
    r = requests.get(url) 
    return r.text 

def fetch_carimgs():
    
    
    htmldata = getdata("https://www.formula1.com/en/teams.html") 
    soup = BeautifulSoup(htmldata, 'html.parser') 
    data_src = []
    for item in soup.find_all('img'):
        try:
            item['src']
        except:
            data_src.append(item['data-src'])
    
    # Teamsm
    url = f'http://ergast.com/api/f1/2022/constructorStandings.json'
    response = requests.get(url)
    constructor_standings = response.json()
    itemlist = constructor_standings['MRData']['StandingsTable']['StandingsLists'][0]['ConstructorStandings']
    teams = []
    for item in itemlist:
        teams.append(item['Constructor']['name'])
    
    data_src = data_src[11:-6]

    multiples = [x*4-1 for x in range(1,11)]
    cars = pd.Series(data_src)[multiples]
    cars.reset_index(drop=True)
    cars = cars.to_list()
    
    cars_dict = {}
    for car, team in zip(cars,teams):
        cars_dict[team] = car
        
    return cars_dict


fetch_carimgs()
    
    
    



