import requests
import datetime
import math
import feedparser
import os
import shutil
from requests.packages import urllib3
import pandas as pd
import numpy as np
import seaborn as sns
from functools import reduce
from itertools import product
import matplotlib.pyplot as plt
urllib3.disable_warnings()

########################################## API'S INFORMATION ##########################################

# Pro-publica
## Congress
headers = {'X-API-Key': 'Li9aCR2BjLZviTE2vhRXQ9QDqq1XU8NxXUXkQDbA',}
pp_base = 'https://api.propublica.org/congress/v1/'
## Campaign Finance
headers_cf = {'X-API-Key': 'VYYgLHhbV4VOedGEXkQVSHw8M4bfeynzuhMuHNTl',}
cf_base = 'https://api.propublica.org/campaign-finance/v1/'

# Open Secrets
apikey = 'd1d012a1e374fbd4a7fff80d7b38215f'
os_base = 'http://www.opensecrets.org/api/?method='
# email : usmca1@economia.gob.mx --- password : pasantia2022

# American Community Survey Data - ACS - Census Bureau
key = '5249ce7f4a7ad2e553fe1572a4c6cdd2b2e4a3da'
last_year = '2020'
acs_base = f'https://api.census.gov/data/{last_year}/acs/acs5'

# Harvard Dataverse
# user: usmca_1-. --- password : pasantia2022
api_token = '645bedbd-2dab-47eb-a8ad-517bd1399411'
# Expiration Date: 2023-08-05

########################################## GENERAL VARIABLES ##########################################

party_codes = {100: 'Democrats', 200: 'Republicans'}

keywords = ['advanced pharmaceutical ingredients','american jobs','buy america','cool','canada','china',
            'country of origin labeling','department of agriculture','department of commerce',
            'department of energy','foreign direct investment','international trade commission','mexico',
            'nafta','section 201','section 232','section 301','section 332','trade','trade agreement',
            'usmca','vaquita marina','world trade organization','advanced pharmaceutical ingredients',
            'agriculture','american-made','antidumping','api','auto','automobile','automotive','back end',
            'batteries','bea','beef','berries','budget reconciliation','build back better','buy america',
            'canada','china','chips','clouthier','commerce','competition','competitionautomobile',
            'competitive business','competitiveness','cool','corn','cotton','country of origin','doc',
            'doe','dol','domestic','domestic manufacturing','domestic production','dot','dumping','ebrard',
            'electric vehicles','exports','fertilizers','foreign','front end','global trade',"gmo's",
            'grain oriented steel','granholm','guzman','import tax','imports','incentives',
            'infrastructure bill','infrastructure plan','itc','katherine tai','labeling','local',
            'machinery','manufacture','manufactures','maquila','maquiladoras','market','medical devices',
            'medical equipment','mexico','mineral fuels','nearshoring','nec','nom','oce','offshoring',
            'origin','personal protection equipment','pharmaceutical','picte','pork','potatoe','ppe',
            'protect american','protect national','raimondo','regulations','regulator','retaliation',
            'roundup','rubber','rules of origin','section 201','section 232','section 301','section 332',
            'semiconductor','shortage','solar panel','soybeans','steel','strengthening america',
            'subsidies','substantial transformation','sugar','sugar cane','supply chains','tai','tariff',
            'tariffs','tomato','tomatoe','tomatoes','trade','trade agreement',
            'small business administration','turtle excluded device','unregulated fishing','usda',
            'usica','usmca','uyghur','vaquita marina','vehicles','vilsack','walsh','wto']

us_state_to_abbrev = {"Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
                      "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
                      "Florida": "FL", "Georgia": "GA", "Hawaii": "HI","Idaho": "ID", "Illinois": "IL",
                      "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
                      "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI",
                      "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", 
                      "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
                      "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND",
                      "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
                      "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
                      "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA",
                      "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
                      "District of Columbia": "DC", "American Samoa": "AS", "Guam": "GU",
                      "Northern Mariana Islands": "MP", "Puerto Rico": "PR",
                      "United States Minor Outlying Islands": "UM", "U.S. Virgin Islands": "VI",}

abbrev_to_us_state = dict(map(reversed, us_state_to_abbrev.items()))  # invert the dictionary
CURRENT_YEAR = datetime.datetime.now().year
CURRENT_CONGRESS = int(math.floor((CURRENT_YEAR - 1789) / 2 + 1))
bill_type = ['introduced','updated','cosponsored','withdrawn']
os_get = ['office','phone','fax','webform','congress_office']
os_sum = ['cycle','first_elected','next_election','total','spent',
          'cash_on_hand','debt','source',"last_updated"]

# Information from Census
names_1 = requests.get(f'{acs_base}/profile/variables.json').json()['variables']
names_2 = requests.get(f'{acs_base}/subject/variables.json').json()['variables']
names_3 = requests.get(f'{acs_base}/variables.json').json()['variables']
names_4 = requests.get(f'{acs_base}/cprofile/variables.json').json()['variables']
url_states = f'{acs_base}/profile?get=NAME,DP02_0001E&for=state:*&key={key}'
states_codes = {state[0] : state[2] for state in requests.get(url_states).json()[1:]}

# Colors
COLOR = 'lightgray'
EDGECOLOR = 'black'
DEMOCRAT = 'blue'
REPUBLICAN = 'red'
LIBERTARIAN = 'yellow'

########################################## HELPER FUNCTIONS ##########################################

def build_dic(lst, value):
    '''
    Esta funcion toma una lista de atributos y un valor, y regresa un diccionario anidado
    con el valor al final del 'arbol'.
    '''
    if len(lst) > 1:
        return {lst[0]: build_dic(lst[1:], value)}
    return {lst[0]: value}

def mergedicts(dict1, dict2):
    '''
    Esta funcion ayuda a combinar diccionarios
    '''
    for k in set(dict1.keys()).union(dict2.keys()):
        if k in dict1 and k in dict2:
            if isinstance(dict1[k], dict) and isinstance(dict2[k], dict):
                yield (k, dict(mergedicts(dict1[k], dict2[k])))
            else:
                yield (k, dict2[k])
        elif k in dict1:
            yield (k, dict1[k])
        else:
            yield (k, dict2[k])

def relevant(bill):
    text = f"{bill['title']} {bill['short_title']} {bill['summary']} {bill['summary_short']}"
    for ch in [',', '.', ':', ';', '(', ')']:
        if ch in text:
            text = text.replace(ch,'')
    text = text.replace('  ',' ').lower()
    lst = text.split()
    for pos, word in enumerate(lst):
        if word in keywords:
            return True
        if pos != len(lst) - 1:
            tuple = f'{word} {lst[pos + 1]}'
            if tuple in keywords:
                return True
            if pos != len(lst) - 2:
                tuple += f' {lst[pos + 2]}'
                if tuple in keywords:
                    return True
    return False

def relevant_bill_to_df(bills_relevant):
    lst = []
    for k in bills_relevant.keys():
        for b in bills_relevant[k]:
            b['status'] = k
            b['Republican_sponsors'] = b['cosponsors_by_party'].get('R', 0)
            b['Democrat_sponsors'] = b['cosponsors_by_party'].get('D', 0)
            b['Other_sponsors'] = b['cosponsors'] - b['Republican_sponsors'] - b['Democrat_sponsors']
            b['sponsor'] = f"{b['sponsor_title']} {b['sponsor_name']} ({b['sponsor_party']})"
            b['id'] = f"{b['number']}. Congreso: {b['congress']}. Estado: {'activo' if b['active'] else 'inactivo'}"
            [b.pop(key) for key in ['bill_type','bill_uri','sponsor_title','sponsor_id','sponsor_name',
                                    'sponsor_state','sponsor_party','sponsor_uri','gpo_pdf_uri','govtrack_url',
                                    'last_vote','house_passage','senate_passage','enacted','vetoed','active',
                                    'cosponsors','cosponsors_by_party','congress','bill_id','number']]
        lst.extend(bills_relevant[k])
    df = pd.DataFrame(lst)
    df = df.reindex(columns=['status', 'id','title','short_title','introduced_date','primary_subject','committees','summary',
                            'summary_short','sponsor','Democrat_sponsors','Republican_sponsors','Other_sponsors',
                            'congressdotgov_url'])
    df.rename(columns = {'status':'estatus','title':'título','short_title':'título_corto','introduced_date':'fecha',
                         'primary_subject':'tópico','committees':'comité','summary':'resumen','summary_short':'resumen_corto',
                         'sponsor':'responsable','Democrat_sponsors':'promotores_demócratas',
                         'Republican_sponsors':'promotores_republicanos','Other_sponsors':'otros_promotores',
                         'congressdotgov_url':'url'}, inplace = True)
    return df

# Basic information from delegate
def basic(member_id):
    '''
    A partir del ID seleccionado, contruye un diccionario utilizando la informacion
    tanto de pro-publica como de open-secrets con los datos basicos del representante,
    incluyendo sus roles e informacion de contacto
    '''
    data = requests.get(f'{pp_base}members/{member_id}.json', headers = headers).json()
    basic_info = data['results'][0]
    
    for role in basic_info['roles']:
        for subcommittee in role['subcommittees']:
            try:
                path = "{congress}/{chamber}/committees/{committee}.json".format(
                        congress = role['congress'], chamber = role['chamber'],
                        committee = subcommittee['parent_committee_id'])
                data = requests.get(pp_base + path, headers = headers).json()
                subcommittee['parent_committee_name'] = data['results'][0]['name']            
                subcommittee['parent_committee_url'] = data['results'][0]['url']
            except:
                subcommittee['parent_committee_name'] = [commitee['name'] for commitee in\
                    role['committees'] if commitee['code'] == subcommittee['parent_committee_id']][0]
                subcommittee['parent_committee_url'] = ''
            
    # Contact information from Open Secrets
    path = f"{os_base}getLegislators&id={basic_info['crp_id']}&apikey={apikey}&output=json"
    data = requests.get(path).json()
    
    for attribute in data['response']['legislator']['@attributes']:
        if attribute in os_get:
            basic_info[attribute] = data['response']['legislator']['@attributes'][attribute]

    r = requests.get(f'https://bioguide.congress.gov/search/bio/{member_id}.json').json()
    basic_info['text'] = r['data']['profileText']
    
    if len(basic_info['roles'][0]['committees']) > 0:
        basic_info['commite_lst'] = [com['name'] for com in basic_info['roles'][0]['committees']]
    else:
        basic_info['commite_lst'] = []
        
    if len(basic_info['roles'][0]['committees']) > 0:
        basic_info['subcommittees_lst'] = [com['name'] for com in basic_info['roles'][0]['subcommittees']]
    else:
        basic_info['subcommittees_lst'] = []

    return basic_info

# Political Ideology
def political_ideology(basic_info):
    '''
    Primary liberal-conservative axis describes preferences over fundamental issues of taxation, spending and
    redistribution (ECONOMIC/REDISTRIBUTIVE). The secondary axis that describes preferences over social issues, such
    as civil rights (SOCIAL/RACIAL). {LIBERAL : -1 - CONSERVATIVE : 1}
    
    The frst kind of DW-NOMINATE scores that we estimate—those used in the web visualizations and the main estimates
    provided in the downloadable datasets—are CommonSpace Constant DW-NOMINATE scores. These are the scores that Poole
    reported at the top of his website. For these scores, House and Senate members are scaled in a single space
    (Common Space) and individual legislators have a constant ideal point throughout their time in the Congress (Constant).
    
    The second kind of DW-NOMINATE score that we provide are known as Nokken-Poole scores. These scores allow legislators'
    ideal points to move freely over time, and thus make less restrictive assumptions about legislator ideological fixedness.
    '''
    cat_lst = ['x_nom', 'y_nom', 'x_nok', 'y_nok']
    dim_lst = ['nominate_dim1', 'nominate_dim2', 'nokken_poole_dim1', 'nokken_poole_dim2']
    key_lst = ['Objective', 'Democrats', 'Republicans', 'Others']
    
    ideology = {key: {cat: [] for cat in cat_lst} for key in key_lst}
    url = 'https://voteview.com/static/data/out/members/{chamber}{congress}_members.json'.format(
            chamber = basic_info['roles'][0]['chamber'][0], congress = CURRENT_CONGRESS)
    data = requests.get(url).json()
    
    for i in data:
        if i['chamber'] != 'President':
            if i['bioguide_id'] == basic_info['id']:
                target = 'Objective'
            else:
                if i['party_code'] in party_codes:
                    target = party_codes[i['party_code']]
                else:
                    target = 'Others'
            for x, y in zip(cat_lst, dim_lst):
                ideology[target][x].append(i[y])
                        
    return ideology

# topics
def topics(basic_info):
    path = 'https://fetch-bill-statuses.appspot.com/membersearch'
    return requests.get(path, {'congress' : CURRENT_CONGRESS,
                               'member' : basic_info['id']}).json()
    
# Bills
def bills(member_id):
    '''
    De pro-publica, extrae todos los bills relacionados con el representante seleccionado
    '''
    bills = {}
    bills_relevant = {}

    for type in bill_type:
        data = requests.get(f'{pp_base}members/{member_id}/bills/{type}.json',
                            headers = headers).json()
        bills[type] = data['results'][0]['bills']
        bills_relevant[type] = [bill for bill in bills[type] if relevant(bill)]

    return bills, bills_relevant

# Funding
def funding(crp_id):
    '''
    De Open-Secrets extrae la informacion de fondeo del representante seleccionado
    '''
    funding = {}

    ## general funding profile
    data = requests.get(f'{os_base}candSummary&cid={crp_id}&apikey={apikey}&output=json').json()
    funding['summary'] = data['response']['summary']['@attributes']

    ## top contributors
    url = f"{os_base}candContrib&cid={crp_id}&cycle={funding['summary']['cycle']}&apikey={apikey}&output=json"
    data = requests.get(url).json()
    funding['contributors'] = [i['@attributes'] for i in data['response']['contributors']['contributor']]

    ## top contributors profiles
    for contributor in funding['contributors']:
        try:
            url = f"{os_base}getOrgs&org={contributor['org_name']}&apikey={apikey}&output=json"
            data = requests.get(url).json()
            orgid = data['response']['organization']['@attributes']['orgid']
            data = requests.get(f'{os_base}orgSummary&id={orgid}&apikey={apikey}&output=json').json()
            contributor['full_profile'] = data['response']['organization']['@attributes']
        except:
            continue

    ## top industries
    url = f"{os_base}candIndustry&cid={crp_id}&cycle={funding['summary']['cycle']}&apikey={apikey}&output=json"
    data = requests.get(url).json()
    funding['industries'] = [i['@attributes'] for i in data['response']['industries']['industry']]
    
    ## top sectors
    path = f"{os_base}candSector&cid={crp_id}&cycle={funding['summary']['cycle']}&apikey={apikey}&output=json"
    data = requests.get(path).json()
    funding['sectors'] = [i['@attributes'] for i in data['response']['sectors']['sector']]

    return funding

def funding_helper(funding, type):
    lst = []
    names = {'contributors':'org_name', 'industries':'industry_name', 'sectors':'sector_name'}
    for i in funding[type]:
        cont = {'nombre': i[names[type]], 'aportes': i['total'], 'aporte_individual': i['indivs'], 'aporte_PAC': i['pacs']}
        lst.append(cont)
    df = pd.DataFrame(lst).fillna(0)
    for i in ['aportes', 'aporte_individual', 'aporte_PAC']:
        df[i] = pd.to_numeric(df[i]) / 1000
    df = df.sort_values(by = 'aportes', ascending = False)
    return df

def econ_demo_info(role):
    state = states_codes[abbrev_to_us_state[role['state']]]
    econ = {}
    econ['other'] = {}
    econ = get_info_area(econ, state)
    if role['chamber'] == 'House' and role['district'] != 'At-Large':
        if len(role['district']) == 1:
            district = str(0) + role['district']
        else:
            district = role['district']
        econ = get_info_area(econ, state, district)
    econ.pop('other')
    econ['COMMERCIAL'] = {}
    econ['COMMERCIAL']['state_census'] = get_commerce_state_census(role['state'])
    econ['COMMERCIAL']['state_nafta'] = nafta_helper(role['state'])
    econ['COMMERCIAL']['state_nafta']['Mex_Share'] = econ['COMMERCIAL']['state_nafta']['Mexico'] /\
        econ['COMMERCIAL']['state_nafta']['World']
    if role['chamber'] == 'House' and role['district'] != 'At-Large':
        econ['COMMERCIAL']['district'] = nafta_helper(role['state'], district)
        econ['COMMERCIAL']['district']['Mex_Share'] = econ['COMMERCIAL']['district']['Mexico'] /\
        econ['COMMERCIAL']['district']['World']
    return econ

NAME_VBLES = dict(mergedicts(dict(mergedicts(names_1, names_2)),
                             dict(mergedicts(names_3, names_4))))

def get_info_area(econ, state, district = None):
    if district is None:
        chamber = 'state'
        comp = ''
    else:
        chamber = 'district'
        comp = f'congressional%20district:{district}&in='
    path1 = f'{acs_base}/subject?get=NAME,group(S2902)&for={comp}state:{state}&key={key}'
    path2 = f'{acs_base}?get=NAME,group(B03002)&for={comp}state:{state}&key={key}'
    path3 = f'{acs_base}/profile?get=group(DP03)&for={comp}state:{state}&key={key}'
    for path in [path1, path2, path3]:
        data = requests.get(path).json()
        for a, b in zip(data[0], data[1]):
            if a in NAME_VBLES:
                lst = NAME_VBLES[a]['label'].replace(":", "").split("!!")
                if len(lst) == 1:
                    if lst[0] in econ['other']:
                        econ['other'][lst[0]][chamber] = b
                    else:
                        econ['other'][lst[0]] = {chamber : b}
                else:
                    lst.append(chamber)
                    temp = {}
                    temp[lst[0]] = build_dic(lst[1:], b)
                    econ = dict(mergedicts(econ, temp))
            else:
                if a in econ['other']:
                    econ['other'][a][chamber] = b
                else:
                    econ['other'][a] = {chamber : b}
    return econ

def get_commerce_state_census(state):
    df = update_census(state, comm_lvl = 'HS2')
    df = df.sum(axis = 0).to_frame().to_dict()[0]
    [df.pop(key) for key in ['COMMODITY_ID', 'COMMODITY']]
    return df

def nafta_helper(state, district = None):
    df = pd.read_csv('NAFTA.csv')
    if district:
        df = df.loc[(df['state'] == state) & (df['district'] == district)]
    df = df.loc[(df['state'] == state)]
    df = df.sum(axis = 0).to_frame().to_dict()[0]
    [df.pop(key) for key in ['state', 'district', 'NAICS4', 'Description']]
    return df

# Funciones para recolectar las fotos
def save_image(bioguide_id, data):
    '''
    Helper que guarda la imagen como archivo binario
    '''
    
    # Make directory if necessary
    full_dir = os.path.dirname("static/bio_pics/")
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)
    
    # Write the binary data.
    with open(f'static/bio_pics/{bioguide_id}.jpg', 'wb') as out_file:
        shutil.copyfileobj(data, out_file)

def individual_lookup(bioguide_id):
    '''
    Toma el bioguide_id y obtiene la imagen del directorio biografico del congreso
    '''
    if not os.path.exists("static/bio_pics/{bioguide_id}.jpg".format(bioguide_id = bioguide_id)):
        lookup_url = 'https://bioguide.congress.gov/bioguide/photo/'
        image_url = f'{bioguide_id[0]}/{bioguide_id}.jpg'
        # Download image if it exists
        file_exists = requests.head(lookup_url + image_url).status_code
        if file_exists == 200:
            binary_download = requests.get(lookup_url + image_url, stream = True)
            save_image(bioguide_id, binary_download.raw)
            
# Info Comercial por Estado
def commerce_helper(comm_type, comm_lvl, country, year = CURRENT_YEAR - 1, state = None):
    columns = {'imp': 'CTY_NAME,I_COMMODITY,I_COMMODITY_SDESC,GEN_VAL_YR',
               'exp': 'CTY_NAME,E_COMMODITY,E_COMMODITY_SDESC,ALL_VAL_YR'}
    base = 'https://api.census.gov/data/timeseries/intltrade/'
    aux1 = 'state' if state else ''
    aux2 = f'&STATE={state}' if state else ''
    flow = f'imports/{aux1}hs' if comm_type == 'imp' else f'exports/{aux1}hs'
    countries = {'-': 'TOT', '2010': 'MEX'}
    param = columns[comm_type]
    url = f'{base}{flow}?get={param}&COMM_LVL={comm_lvl}&CTY_CODE={country}&YEAR={year}&MONTH=12{aux2}&key={key}'
    r = requests.get(url).json()
    data = pd.DataFrame(r[1:], columns = r[0])
    
    if comm_type == 'imp':
        data['GEN_VAL_YR'] = pd.to_numeric(data['GEN_VAL_YR'])
        data.rename(columns = {'CTY_NAME':'CONTRAPARTE','I_COMMODITY':'COMMODITY_ID','I_COMMODITY_SDESC':'COMMODITY',
                               'GEN_VAL_YR': f'{comm_type.upper()}_{countries[country]}'}, inplace = True)
        return data

    data['ALL_VAL_YR'] = pd.to_numeric(data['ALL_VAL_YR'])
    data.rename(columns = {'CTY_NAME':'CONTRAPARTE','E_COMMODITY':'COMMODITY_ID','E_COMMODITY_SDESC':'COMMODITY',
                          'ALL_VAL_YR': f'{comm_type.upper()}_{countries[country]}'}, inplace = True)
    return data

def update_census(state = None, comm_lvl = 'HS6', year = CURRENT_YEAR - 1):
    col_drop = ['CONTRAPARTE', 'COMM_LVL', 'CTY_CODE', 'YEAR', 'MONTH']
    if state:
        col_drop.append('STATE')
    keys = ['COMMODITY_ID', 'COMMODITY']
    data_frames = [commerce_helper(comm_type, comm_lvl, country, year, state).drop(columns = col_drop)\
        for comm_type, country in [(comm_type, country)\
            for comm_type in ['imp', 'exp'] for country in ['-', '2010']]]
    
    df = reduce(lambda l, r: pd.merge(l, r, on = keys, how = 'outer'), data_frames).fillna(0)
    df['MEX_BOT'] = df['EXP_MEX'] - df['IMP_MEX']
    df['SHARE_IMP'] = df['IMP_MEX'] / df['IMP_TOT'] * 100
    df['SHARE_EXP'] = df['EXP_MEX'] / df['EXP_TOT'] * 100
    df['TOT_BOT'] = df['EXP_TOT'] - df['IMP_TOT']
    df = df.sort_values(by = 'COMMODITY_ID').fillna(0)
    return df

def extract_votes(chamber):
    '''Esta funcion extrae toda la informacion de votantes '''
    dic = {'house' : '6157134', 'senate' : '4300300', 'president' : '4299753'}
    data = requests.get(f"https://dataverse.harvard.edu/api/access/datafile/{dic[chamber]}").content.decode('utf8')
    df = pd.DataFrame([x.split('\t') for x in data.split('\n')[1: ]], columns = [x for x in data.split('\n')[0].split('\t')])
    df = df.replace({'"': ''}, regex = True).reset_index()
    df.drop(columns = 'index', inplace = True)
    df = df.dropna(axis = 0, how = 'all')
    df = df.dropna(axis = 1, how = 'all')
    
    if 'special' in df.columns:
        df = df[df.special != 'TRUE']

    for i in ['year', 'candidatevotes', 'totalvotes']:
        df[i] = pd.to_numeric(df[i])

    if 'party_simplified' not in df.columns:
        df['party_simplified'] = np.where(df['party'].isin(['DEMOCRAT', 'REPUBLICAN', 'LIBERTARIAN']), df['party'], 'OTHER')

    df['share_votes'] = df['candidatevotes'] / df['totalvotes'] * 100
    
    return df

def vot_df_hist(state, district = None, years = CURRENT_YEAR - 20, president = False):
    '''Esta funcion devuleve datos electorales de los ultimos N años (20 si no se especifica)'''
    if district and district != 'At-Large':
        tlt = f'Evolucion Votaciones Representante\n(Estado: {state} - Distrito: {district})'
        df = extract_votes('house')
        df = df.loc[(df['state_po'] == state) & (df['year'] >= years) & (df['district'] == '0' + district)]
    else:
        if president:
            tlt = f'Evolucion Votaciones Presidente\n(Estado: {state})'
            df = extract_votes('president')
            df = df.loc[(df['state_po'] == state) & (df['year'] >= years)]
        else:
            tlt = f'Evolucion Votaciones Senado\n(Estado: {state})'
            df = extract_votes('senate')
            df = df.loc[(df['state_po'] == state) & (df['year'] >= years)]

    df['year'] = df['year'].astype(int)
    df = df[['year','party_simplified','share_votes']].groupby(['year','party_simplified']).sum().reset_index()
    for i in ['DEMOCRAT', 'REPUBLICAN', 'LIBERTARIAN', 'OTHER']:
        df[i] = np.where(df['party_simplified'] == i, df['share_votes'], 0)
    df = df.drop(columns=['party_simplified','share_votes'])
    df = df.groupby(['year']).sum().reset_index()
    
    return df, tlt

def race_dict(economics):
    race = dict(economics['Estimate']['Total']['Not Hispanic or Latino'])
    race['Two or more races'].pop('Two races including Some other race')
    race['Two or more races'].pop('Two races excluding Some other race, and three or more races')
    race['Hispanic or Latino'] = {'state': economics['Estimate']['Total']['Hispanic or Latino']['state']}
    if 'district' in economics['Estimate']['Total']['Hispanic or Latino']:
        race['Hispanic or Latino']['district'] = economics['Estimate']['Total']['Hispanic or Latino']['district']
        
    return race

########################################## FUNCIONES PRINCIPALES ##########################################

def menu(chamber, congress = CURRENT_CONGRESS, party = None):
    path = "{congress}/{chamber}/members.json".format(
        congress = congress, chamber = chamber)
    data = requests.get(pp_base + path, headers = headers).json()
    if party:
        return {'{last}, {first} ({party}) - {state}'.format(
        last = i['last_name'], first = i['first_name'], party = i['party'],
        state = i['state']) : i['id'] for i in data['results'][0]['members']
                if i['party'] == party}

    return {'{last}, {first} ({party}) - {state}'.format(
        last = i['last_name'], first = i['first_name'], party = i['party'],
        state = i['state']) : i['id'] for i in data['results'][0]['members']}

def get_info(member_id):
    basic_info = basic(member_id)
    ideology = political_ideology(basic_info)
    topics_of_interest = topics(basic_info)
    bills_info, bills_relevant = bills(member_id)
    funding_info = funding(basic_info['crp_id'])
    economics = econ_demo_info(basic_info["roles"][0])
    if basic_info['rss_url'] is not None:
        feed_url = basic_info['rss_url']
        feed = feedparser.parse(feed_url)
    else:
        feed = None
    individual_lookup(member_id)
    return basic_info, ideology, topics_of_interest, bills_info,\
           bills_relevant, funding_info, economics, feed
           
def topics_lst(subject = 'policies'):
    '''
    Segun la naturaleza de la consulta (policies o legislativesubjects) devuelve
    la lista de temas disponible
    '''
    path = "https://fetch-bill-statuses.appspot.com/listsubjects"
    temp = requests.get(path, {'congress' : str(CURRENT_CONGRESS)}).json()
    topics = [key for key in temp[subject] if temp[subject][key][str(CURRENT_CONGRESS)] > 0]
    topics.sort()
    return topics

def ranking_topic(topic, subject = 'policies', chamber = None, party = None, mem_comm = None):
    if not chamber:
        if party:
            members = dict(mergedicts(menu('senate', party = party),
                                      menu('house', party = party)))
        else:
            members = dict(mergedicts(menu('senate'), menu('house')))
    else:
        if party:
            members = menu(chamber, party = party)
        else:
            members = menu(chamber)
            
    dict_members = {}
    
    for name, id in members.items():
        path = 'https://fetch-bill-statuses.appspot.com/membersearch'
        if (mem_comm and id in mem_comm) or mem_comm is None:
            try:
                temp = requests.get(path, {'congress' : '117', 'member' : id}).json()[subject][topic]
                dict_members[name] = temp
                dict_members[name]['score'] = 0.1 * temp.get('cosponsored', 0)\
                    + temp.get('sponsored', 0) + 0.2 * temp.get('originalcosponsored', 0)
            except:
                continue
        else:
            continue
    
    return dict_members

def menu_commitees(chamber = None):
    '''
    Devuelve la lista de todos los comites del actual congreso. Puede recibir como input un filtro para
    los comites a devolver: 'house', 'senate' o 'joint'.
    '''
    if chamber:
        data = requests.get(f'{pp_base}{CURRENT_CONGRESS}/{chamber}/committees.json', headers = headers).json()
        return {i['name']: i['id'] for i in data['results'][0]['committees']}
    
    commitees = {}
    
    for chamber in ['house', 'senate', 'joint']:
        data = requests.get(f'{pp_base}{CURRENT_CONGRESS}/{chamber}/committees.json', headers = headers).json()
        commitees = dict(mergedicts(commitees,
                    {i['name']: i['id'] for i in data['results'][0]['committees']}))
    
    return commitees

def rep_commitees(list_comm, chamber = None):
    members = set()
    
    if chamber:
        for i in list_comm:
            data = requests.get(f'{pp_base}{CURRENT_CONGRESS}/{chamber}/committees/{i}.json', headers = headers).json()
            members.update([member['id'] for member in data['results'][0]['current_members']])
        return members
    
    for i in list_comm:
        for chamber in ['house', 'senate', 'joint']:
            data = requests.get(f'{pp_base}{CURRENT_CONGRESS}/{chamber}/committees/{i}.json', headers = headers).json()
            if data['status'] == 'OK':
                members.update([member['id'] for member in data['results'][0]['current_members']])
    
    return members

########################################## GRAFICAS ##########################################

def graph_ideology(basic_info, ideology, sup = 'm', legend = False):
    dfa = pd.DataFrame.from_dict(ideology['Democrats'])
    dfa['Category'] = 'Democrats'
    dfb = pd.DataFrame.from_dict(ideology['Republicans'])
    dfb['Category'] = 'Republicans'
    dfc = pd.DataFrame.from_dict(ideology['Objective'])
    dfc['Category'] = basic_info['first_name'] + ' ' + basic_info['last_name']

    g = sns.jointplot(data = pd.concat([dfa, dfb, dfc], ignore_index=True),
                x = 'x_no{}'.format(sup), y = 'y_no{}'.format(sup), hue = 'Category',
                palette = {'Democrats': DEMOCRAT, 'Republicans': REPUBLICAN,
                str(basic_info['first_name'] + ' ' + basic_info['last_name']): EDGECOLOR},
                s = 60, edgecolor = EDGECOLOR, linewidth = 0.5, ratio = 10, space = 0,
                xlim=(-1, 1), ylim=(-1, 1), alpha = .3, legend = legend)

    g.refline(x = ideology['Objective']['x_no{}'.format(sup)],
            y = ideology['Objective']['y_no{}'.format(sup)])
    g.set_axis_labels('Económico / Redistributivo', 'Social / Racial')
    g.fig.suptitle('Espectro Político')
    g.fig.subplots_adjust(top=0.93)
    g.ax_joint.text( 0.61, -0.98, 'conservador')
    g.ax_joint.text(-0.98,  0.95, 'conservador')
    g.ax_joint.text(-0.98, -0.98, 'liberal')
    g.ax_joint.set_xticks([-1, -0.5, 0, 0.5, 1])
    g.ax_joint.set_yticks([-1, -0.5, 0, 0.5, 1])
    
    return g

def graph_top(top, title, rank = 20):
    tot = pd.DataFrame.from_dict(top, orient='index').fillna(0).sort_values('score', ascending = False)
    max_count = max(tot['cosponsored'] + tot['sponsored'] + tot['originalcosponsored'])
    step = int(max_count / 4)
    tot.head(rank).sort_values('score').loc[:, tot.columns != 'score'].plot(kind = "barh",
                                                                      stacked = True,
                                                                      color = {'sponsored': EDGECOLOR,
                                                                               'cosponsored': COLOR,
                                                                               'originalcosponsored': 'gray'},
                                                                      title = title,
                                                                      xlabel = "No. de Bills",
                                                                      xticks = list(range(0, int(max_count) + step, step)),)
    return plt.gcf()

def help_commitee(commitee, topics_dic):
    mem_comm = rep_commitees([commitee])
    list_graphs = []
    for subject in topics_dic:
        for topic in topics_dic[subject]:
            for party in ['R', 'D']:
                try:
                    top = ranking_topic(topic = topic, subject = subject, party = party, mem_comm = mem_comm)
                    g = graph_top(top, topic, rank = 1000)
                    list_graphs.append(g)
                except:
                    continue
    return list_graphs

def graph_top_exp_imp(state = None, district = None, comm_lvl = 'HS6', year = CURRENT_YEAR - 1, rank = 15):
    # Data from Census
    df = update_census(state, comm_lvl, year)
    df.COMMODITY = df.COMMODITY.str.title()
    exp, imp = df.sort_values('EXP_MEX', ascending = False).head(rank), df.sort_values('IMP_MEX', ascending = False).head(rank)
    bot_pos, bot_neg = df.sort_values('MEX_BOT', ascending = False).head(rank), df.sort_values('MEX_BOT', ascending = True).head(rank)
    exp.EXP_MEX = exp.EXP_MEX / 1000000
    imp.IMP_MEX = imp.IMP_MEX / 1000000
    bot_pos.MEX_BOT = bot_pos.MEX_BOT / 1000000
    bot_neg.MEX_BOT = bot_neg.MEX_BOT / 1000000 * -1
    
    # Data from Nafta Dataset
    df_nafta = pd.read_csv('NAFTA.csv')
    aux = f'Estado: {state}' if state else 'Total Nacional'
    
    if district and district == 'At-Large':
        district = None

    fig1, axs1 = plt.subplots(2, 2, figsize = (10,15))

    for i, j in product(range(2), range(2)):
        df, aux1, aux2 = (exp, 'EXP', 'Exportaciones Hacia') if i == 0 else (imp, 'IMP', 'Importaciones Desde')
        aux3, title, dec, per = (f'{aux1}_MEX', f'Top {aux2} México (Fuente: Census)\n({aux})', 2, '')\
            if j == 0 else (f'SHARE_{aux1}', f'% {str.capitalize(aux1)}. Totales del Producto', 1, '%')
        axs1[i, j].barh(df.COMMODITY, df[aux3], align = 'center', color = COLOR, edgecolor = EDGECOLOR)
        axs1[i, j].invert_yaxis()
        axs1[i, j].set(frame_on = False)
        axs1[i, j].set_title(title, fontweight = "bold")
        for k, txt in enumerate(df[aux3]):
            axs1[i, j].annotate(f'{round(txt, dec)}{per}', (df[aux3].iat[k] + .1, df.COMMODITY.iat[k]))
        if j == 0:
            axs1[i, j].set_xlabel('Millones de Dólares')
        else:
            axs1[i, j].get_yaxis().set_visible(False)

    fig2, axs2 = plt.subplots(2, 1, figsize = (4, 13))

    for j in range(2):
        df, aux4 = (bot_pos, 'Exportaciones Netas Hacia') if j == 0 else (bot_neg, 'Importaciones Netas Desde')
        axs2[j].barh(df.COMMODITY, df.MEX_BOT, align = 'center', color = COLOR, edgecolor = EDGECOLOR)
        axs2[j].invert_yaxis()
        axs2[j].set_title(f'Top {aux4} México (Fuente: Census)\n({aux})', fontweight = "bold")
        axs2[j].set_xlabel('Millones de Dólares')
        axs2[j].set(frame_on = False)
        for i, txt in enumerate(df.MEX_BOT):
            axs2[j].annotate(f'{round(txt, 2)}', (df.MEX_BOT.iat[i] + .1, df.COMMODITY.iat[i]))
    
    fig3 = nafta_graph_helper(df_nafta, rank = rank, state = state)
    
    if district:
        fig4 = nafta_graph_helper(df_nafta, rank = rank, state = state, district = district)
        return fig1, fig2, fig3, fig4
    
    return fig1, fig2, fig3, None

def nafta_graph_helper(df, rank, state = None, district = None):
    main = 'Top Exportaciones Hacia México (Fuente: The Trade Partnership)'
    if state is None and district is None:
        df = df.groupby('Description').sum().reset_index()
        df['Mex_Share'] = df['Mexico'] / df['World']
        title = f'{main}\n(Total Nacional)'
    elif state and district is None:
        df = df.loc[(df['state'] == state)].groupby(['Description']).sum().reset_index()
        df['Mex_Share'] = df['Mexico'] / df['World']
        title = f'{main}\n(Estado: {state})'
    else:
        df = df.loc[(df['state'] == state) & (df['district'] == district)]
        title = f'{main}\n(Estado: {state}, Distrito: {district})'

    df = df.sort_values(by='Mexico', ascending = False).head(rank)
    df['Mexico'] = df['Mexico'] / 1000000
    df['World'] = df['World'] / 1000000
    df['Mex_Share'] = df['Mex_Share'] * 100

    fig, axs = plt.subplots(1, 3, figsize = (15, 5))

    axs[0].barh(df.Description, df.Mexico, align = 'center', color = COLOR, edgecolor = EDGECOLOR)
    axs[0].invert_yaxis()
    axs[0].set_title(title, fontweight = "bold")
    axs[0].set_xlabel('Millones de Dólares')
    axs[0].set(frame_on = False)
    for i, txt in enumerate(df.Mexico):
        axs[0].annotate(f'{round(txt, 2)}', (df.Mexico.iat[i] + .1, df.Description.iat[i]))

    axs[1].barh(df.Description, df.Mex_Share, align = 'center', color = COLOR, edgecolor = EDGECOLOR)
    axs[1].get_yaxis().set_visible(False)
    axs[1].invert_yaxis()
    axs[1].set_title('%', fontweight = "bold")
    axs[1].set(frame_on = False)
    for i, txt in enumerate(df.Mex_Share):
        axs[1].annotate(f'{round(txt, 1)}%', (df.Mex_Share.iat[i] + 1, df.Description.iat[i]))

    axs[2].barh(df.Description, df['Direct Jobs'], align = 'center', color = COLOR, edgecolor = EDGECOLOR)
    axs[2].barh(df.Description, df['Indirect Jobs'], align = 'center', left = df['Direct Jobs'],
                color = 'darkgray', edgecolor = EDGECOLOR)
    axs[2].get_yaxis().set_visible(False)
    axs[2].invert_yaxis()
    axs[2].set_title(f'Empleos Relacionados', fontweight = "bold")
    axs[2].set_xlabel('No. de Empleos')
    axs[2].set(frame_on = False)
    axs[2].legend(['Directos','Indirectos'], loc = 'lower right')
    for i, txt in enumerate(df['Total Jobs']):
        axs[2].annotate(f'{round(txt)}', (df['Total Jobs'].iat[i] + 1, df.Description.iat[i]))
        
    return fig

def vot_hist(state, district = None, years = CURRENT_YEAR - 20, president = False):
    pre, pre_tlt = vot_df_hist(state, years = years, president = True)
    sen, sen_tlt = vot_df_hist(state = state, years = years)
    aux_lst = [(pre, pre_tlt), (sen, sen_tlt)]
    if district:
        dis, dis_tlt = vot_df_hist(state = state, years = years, district = district)
        aux_lst.append((dis, dis_tlt))
        
    color_dic = {'DEMOCRAT': DEMOCRAT, 'REPUBLICAN': REPUBLICAN,
                 'LIBERTARIAN': LIBERTARIAN, 'OTHER': COLOR}

    dim = len(aux_lst)
    
    fig, ax = plt.subplots(1, dim, figsize = (5 * dim, 5))
    
    for j, tup in enumerate(aux_lst):
        df, title = tup
        if j == 0:
            df.plot.bar(x = 'year', ax = ax[j], color = color_dic,
                        alpha = 0.5, edgecolor = EDGECOLOR, width= 0.9,
                        legend = False, title = title).legend(loc='upper center',
                                                              bbox_to_anchor=(0.5, -0.2),
                                                              fancybox=True,
                                                              shadow=True, ncol = 4)
        else:
            df.plot.bar(x = 'year', ax = ax[j], color = color_dic,
                        alpha = 0.5, edgecolor = EDGECOLOR, width= 0.9,
                        legend = False, title = title)

    return fig

def graph_funding(funding):
    type_lst = ['Contribuyentes', 'Industrias', 'Sectores']
    df_lst = [funding_helper(funding, i) for i in ['contributors', 'industries', 'sectors']]   
    fig, axs = plt.subplots(3, 1, figsize = (8, 20))
    for j in range(3):
        axs[j].barh(df_lst[j].nombre, df_lst[j].aporte_individual, align = 'center', color = COLOR, edgecolor = EDGECOLOR)
        axs[j].barh(df_lst[j].nombre, df_lst[j].aporte_PAC, align = 'center', left = df_lst[j].aporte_individual,
                    color = 'darkgray', edgecolor = EDGECOLOR)
        axs[j].invert_yaxis()
        axs[j].set_title(f'Top {type_lst[j]}', fontweight = "bold")
        axs[j].set_xlabel('USD miles')
        axs[j].set(frame_on = False)
        for i, txt in enumerate(df_lst[j].aportes):
            axs[j].annotate(f'{round(txt, 1)}', (df_lst[j].aportes.iat[i], df_lst[j].nombre.iat[i]))
    axs[2].legend(['Aporte Individual','Aporte PAC'], loc = 'lower right')
    return fig

def individual_topics_df(topics, rank = 20):
    df = pd.DataFrame.from_dict(topics, orient = 'index').fillna(0).reset_index()
    df['index'] = df['index'].replace([''],'Non-classified')
    df['score'] = 0.1 * df['cosponsored'] + df['sponsored'] + 0.2 * df['originalcosponsored']
    df['tot'] = df['cosponsored'] + df['sponsored'] + df['originalcosponsored']
    df = df.sort_values(by = 'score', ascending = False).head(rank)
    df = df.rename(columns={'index':'cat'})
    return df

def graph_topics_of_interest(topics_of_interest, rank = 20):
    policies = individual_topics_df(topics_of_interest['policies'], rank = rank)
    legsubjects = individual_topics_df(topics_of_interest['legsubjects'], rank = rank)

    fig, axs = plt.subplots(2, 1, figsize = (4, 13))

    for j in range(2):
        df, title = (policies, 'Top temas políticos') if j == 0 else (legsubjects, 'Top temas legislativos')
        axs[j].barh(df.cat, df['sponsored'], align = 'center', color = EDGECOLOR)
        axs[j].barh(df.cat, df['cosponsored'], align = 'center', left = df['sponsored'], color = COLOR)
        axs[j].barh(df.cat, df['originalcosponsored'], align = 'center', left = df['cosponsored'], color = 'gray')
        axs[j].invert_yaxis()
        axs[j].set_title(title, fontweight = "bold")
        axs[j].set_xlabel('Bills')
        axs[j].set(frame_on = False)
        for i, txt in enumerate(df.tot):
            axs[j].annotate(f'{round(txt)}', (df.tot.iat[i], df.cat.iat[i]))
        axs[0].legend(['Sponsored', 'Cosponsored', 'Original Cosponsored'], loc = 'lower right')
        
    return fig

def pie_econ_graph(data_dic, aux_title, list_remove = [], order = [], vertical = False):
    df = dict(data_dic)
    
    if len(list_remove) > 0:
        for i in list_remove:
            df.pop(i)

    [df.pop(i) for i in ['state','district'] if i in df]
    df = pd.DataFrame.from_dict(df, orient = 'index')

    if len(order) > 0:
        df = df.reindex(order)

    for i in ['state', 'district']:
        if i in df:
            df[i] = pd.to_numeric(df[i])
        else:
            continue

    if 'district' not in df:
        fig, ax = plt.subplots()
        ax.pie(df.state, labels = df.index, autopct='%1.1f%%', shadow = True, startangle = 90,
               wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'}, textprops={'size': 'x-large'},
               pctdistance = 0.8, labeldistance = 1.3)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax.set_title(f'{aux_title}\n(a nivel estatal)', fontweight = "bold")
        
        return fig
    
    if vertical:
        fig, axs = plt.subplots(2, 1, figsize = (5, 15))
    else:
        fig, axs = plt.subplots(1, 2, figsize = (15, 5))
    
    for j, dim in enumerate(['state','district']):
        aux = 'a nivel estatal' if dim == 'state' else 'a nivel de distrito'
        axs[j].pie(df[dim], labels = df.index, autopct='%1.1f%%', shadow = True, startangle = 90,
               wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'}, textprops={'size': 'x-large'},
               pctdistance = 0.8)
        axs[j].set_title(f'{aux_title}\n({aux})', fontweight = "bold")
        axs[j].axis('equal')
    
    return fig

def commerce_graph_time(com_map, year = CURRENT_YEAR - 1, tl = None):
    comm_lvl = com_map[0]
    options = com_map[1] 
    df_base, commodities = commerce_graph_helper(options, comm_lvl, year, True)
    base = [df_base]

    for i in range(1, 11):
        if year - i >= 2013:
            aux = commerce_graph_helper(options, comm_lvl, year - i)
            base.append(aux)
        else:
            break

    df = pd.concat(base, axis=1).transpose()
    df.IMP_MEX = df.IMP_MEX / 1000000
    df.EXP_MEX = df.EXP_MEX / 1000000
    df.MEX_BOT = df.MEX_BOT / 1000000
    df = df.rename(columns={'IMP_MEX': 'Importaciones desde México',
                    'EXP_MEX': 'Exportaciones hacia México',
                    'MEX_BOT': 'Balanza Comercial'})
    df = df.drop(['IMP_TOT', 'EXP_TOT', 'SHARE_IMP', 'SHARE_EXP', 'TOT_BOT'], axis=1)
    ax = df.plot(kind = 'line', title = tl if tl else f'{str(commodities[0]).title()}',
                lw = 2.5, colormap = 'jet', marker='.', markersize = 8)
    ax.set_ylabel("Millones de Dólares")
    
    return ax

def commerce_graph_helper(options, comm_lvl, year, base = False):
    df = update_census(state = None, comm_lvl = comm_lvl, year = year)
    df = df[df['COMMODITY_ID'].isin(options)]
    
    if base:
        commodities = df['COMMODITY'].unique()
        df = df.drop(['COMMODITY_ID', 'COMMODITY'], axis=1).sum().to_frame(name = year)
        return df, commodities
    
    df = df.drop(['COMMODITY_ID', 'COMMODITY'], axis=1).sum().to_frame(name = year)
    return df
