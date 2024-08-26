import requests
from bs4 import BeautifulSoup
import pandas as pd

for year in range(2020, 2025):

    url = f"https://www.basketball-reference.com/players/j/jokicni01/gamelog/{year}"

    response = requests.get(url)

    soup = BeautifulSoup(response.content, 'lxml')
    
    table = soup.find('table', id='pgl_basic')

    #Extract headers
    headers = [th.get_text() for th in table.find('thead').find_all('th')][1:]  # Skipping the first header (row numbers)
    
    #Extract rows
    rows = table.find('tbody').find_all('tr', class_=lambda x: x != 'thead')
    
    game_data = []
    for row in rows:
        columns = row.find_all('td') #Attribute columns are all tagged with td
        if columns:
            game_data.append([col.get_text() for col in columns])
    
    df = pd.DataFrame(game_data, columns=headers)
    
    df = df[(df['GS'] != "")]
    df = df[(df['GS'] != "Inactive") & (df['GS'] != "Did Not Play") & (df['GS'] != "Did Not Dress")] #Removing all DNPs
    df.to_csv(f"Data/{year}.csv")