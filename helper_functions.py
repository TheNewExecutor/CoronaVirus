import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import plotly.express as px
from fuzzywuzzy import fuzz, process
from collections import defaultdict


def load_raw_global():
    """Loads confirmed, deaths, recovered data frames in a dict."""
    path = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/' \
           'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_'
    world = {i: pd.read_csv(path + i.lower() + '_global.csv').drop(columns=['Lat', 'Long'])
             for i in ['Confirmed', 'Deaths', 'Recovered']}
    return world


def load_local():
    """Loads state, county data in tidy format, packaged in a dict."""
    path = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/'
    data = {'state': pd.read_csv(path + 'us-states.csv'),
            'county': pd.read_csv(path + 'us-counties.csv')}
    data['state'].rename(columns={'date': 'Date', 'state': 'State',
                                  'cases': 'Confirmed', 'deaths': 'Deaths'}, inplace=True)
    data['county'].rename(columns={'date': 'Date', 'county': 'County',
                                   'state': 'State', 'cases': 'Confirmed',
                                   'deaths': 'Deaths'}, inplace=True)
    return data


def search_country(data, countries):
    """Narrow search results for confirmed, deaths"""
    data = data[data['Country/Region'].isin(countries)]
    return confirmed, deaths


def time_series(data):
    """Transpose data with time series index, and swaps index and columns"""
    idx = pd.to_datetime(data.columns, errors='coerce')
    ts = (data
          .T
          .loc[data.columns.drop(labels=['Province/State', 'Country/Region', 'Lat', 'Long'],
                                 errors='ignore')
          ]
          .set_index(idx)
          )
    return ts


def load_merged_daily_global(datadict):
    """Create dataframes of daily confirmed, deaths, recovered in tidy (long) format"""
    d_ = list(datadict.values())
    merged = d_[0]
    for d in d_:
        merged = pd.merge(merged, d)
    for key in ['Confirmed', 'Deaths', 'Recovered']:
        merged['Daily ' + key] = (merged
                                  .groupby('Country/Region')
                                  [key]
                                  .diff()
                                  .fillna(merged[key])
                                  )
    return merged


def load_tidy_global(data):
    """Load confirmed, deaths, recovered in tidy (long) format"""
    datadict = {}
    for key, name in zip(data, ['Confirmed', 'Deaths', 'Recovered']):
        country_aggregated = data[key].groupby('Country/Region').agg(sum).reset_index()
        datadict[key] = pd.melt(country_aggregated, 'Country/Region', var_name='Date', value_name=name)
    return datadict


def load_merged_daily_local(data):
    """Create dict of dataframes of state or county level daily cases, deaths in tidy (long) format in a dictionary."""
    data['Date'] = pd.to_datetime(data.Date)
    iscounty = 'County' in data.columns
    group = 'State'
    if iscounty:
        group = ['County', 'State']
    for i in ['Cases', 'Deaths']:
        data['Daily ' + i] = data.groupby(group)[i].diff().fillna(data[i])
    return data


def load_daily_county(county):
    """Create dict of dataframes of county-level daily cases, deaths
        in tidy (long) format in a dictionary"""
    daily_county = {}
    county['Date'] = pd.to_datetime(county['Date'])
    multi_idx = county.set_index(['County', 'State']).index
    county['County_State'] = multi_idx
    for i in ['Cases', 'Deaths']:
        df = (county
              .pivot(index='Date', columns='County_State', values=i)
              .fillna(0)
              .diff()
              .reset_index()
              )
        daily_county[i] = pd.melt(df, id_vars='Date', var_name=['County_State'],
                                  value_name='Daily ' + i)
        daily_county[i]['County'] = daily_county[i]['County_State'].apply(lambda x: x[0])
        daily_county[i]['State'] = daily_county[i]['County_State'].apply(lambda x: x[1])
    return daily_county


# Functions to filter data
def top_n_countries(world_confirmed, feature='Confirmed', n=10):
    """Return list of top n countries with highest feature (Confirmed, Deaths, Recovered, Daily ...)"""
    top_n = (world_confirmed
             .groupby('Country/Region')
             .agg('max')
             .sort_values(feature, ascending=False)
             .head(n)
             .index
             .values)
    return list(top_n)


def top_n_states(states, feature='Confirmed', n=10):
    top_n = (states
             .groupby('State')
             .agg('max')
             .sort_values(feature, ascending=False)
             .head(n)
             .index
             .values)
    return list(top_n)


def top_n_counties(counties, feature='Confirmed', n=10):
    top_n = (counties
             .groupby('County_State')
             .agg['max']
             .sort_values('Confirmed', ascending=False)
             .head(n)
             .index
             .values)
    return list(top_n)


def filter_countries(data, countries):
    return data[data['Country/Region'].isin(countries)]


def filter_states(data, states):
    return data[data['States'].isin(states)]


def filter_counties(data, counties):
    return data[data['County_States'].isin(counties)]


# Text processing
def convert2num(text):
    """
    Converts unit free text into numbers

    Parameters:
    ----------------
    text: string
        description

    Returns:
    ----------------
    num: float
         Data converted to numerical form

    """
    if text is np.nan:
        return 0
    if type(text) in [np.float64, float, int, np.int64, np.float32, np.float16,
                      np.int32, np.int16]:
        return text

    factor = 1
    dd = defaultdict(lambda: 1, {'m': 1e6, 'b': 1e9, 'tr': 1e12})
    unit = re.search(r'([a-z]*)(illion)', text)
    if unit:
        factor = dd[unit.group(1)]
        text = text.replace(unit.group(), '').rstrip()

    found = re.search(r'-?\d+[\.]?\d*', text.replace('$', '').replace(',', ''))

    if found:
        try:
            num = float(found.group()) * factor
        except:
            num = text
            print('Had trouble converting %s' % text)
            print('Obtained: ', found.group())
            return num
    else:
        num = text
    return num


def remove_stop_words(stop_words, text):
    """
    Description

    Parameters:
    ----------------
    stop_words: list
        List of strings to remove
    text: string
        String to process

    Returns:
    ----------------
    filtered: string
         String with stop words removed
    """
    text = text.lower()
    for word in stop_words:
        text = re.sub(word.lower(), '', text)

    return text


def search_fields(search_term, search_space, scorer='partial', cutoff=80, max_results=15):
    """
    Searches fields in countries dataset by partial string matching

    Parameters:
    ----------------
    search_term: string
        A string that approximates the matches
    search_space: list-like object
        Words to search from
    scorer: {'simple', 'partial', 'token sort', 'tokens set ratio'}
        Strings to specify scorers fuzz.ratio, fuzz.partial_ratio, fuzz.token_set_ratio
        respectively
    cutoff: int
        Minimum score to count as a match
    max_results: int
        Limit on how many results to show

    Returns:
    ----------------
    results: list
        List of words that meet cutoff score

    """
    scorers = {'simple': fuzz.ratio,
               'partial': fuzz.partial_ratio,
               'token sort': fuzz.token_sort_ratio,
               'token set': fuzz.token_set_ratio}

    if scorer not in scorers.keys():
        print(scorer, ' not found among valid scorers. Using default instead.')
        scorer = 'partial'

    results = process.extract(search_term, search_space,
                              scorer=scorers[scorer], limit=max_results)

    filtered_results = [(field, score) for field, score in results if score >= cutoff]

    return filtered_results


def find_aliases(match_space, search_space, stop_words=[],
                 scorers=['partial', 'token sort'], cutoff=70, max_results=5):
    """
    Creates a dictionary between entries of one list and the analogs of another
    through consecutive string matches

    Parameters:
    ----------------
    match_space: list-like
        Index to translate to best matches
    search_space: list-like
        Analogous list to search from
    stop_words: list
        Words to exclude from the match_space
    scorers: list of strings
        List containing first scorer and second scorer for string matching
    cutoff: int
        Threshold for words to consider for matching
    max_results: int
        Expected number of results with top scores

    Returns:
    ----------------
    best_matches: dict
        Dictionary with match_space words as keys and best matches from search_space
        as values
    unmatched: list
        List of words from the match_space with no matches

    """
    match_set = set(match_space)
    search_set = set(search_space)
    intersection = match_set & search_set
    diff = match_set - intersection
    search_space = search_set - intersection
    unmatched = []
    best_matches = {word: word for word in intersection}
    for word in diff:

        # First filter keeping only max scores
        match1 = search_fields(remove_stop_words(stop_words, word), search_space,
                               scorer=scorers[0], max_results=max_results, cutoff=cutoff)
        if len(match1) > 0:
            best_score = match1[0][1]
            first_round_best = [tup[0] for tup in match1 if tup[1] == best_score]

            # Second filter
            match2 = search_fields(remove_stop_words(stop_words, word), first_round_best,
                                   scorer=scorers[1], max_results=1,
                                   cutoff=cutoff)
            if len(match2) > 0:
                best_matches[word] = match2[0][0]
            else:
                best_matches[word] = match1[0][0]
        else:
            unmatched.append(word)

    return best_matches, unmatched


def translate(text, d):
    """Translate text to alias if present in dicitonary"""
    try:
        return d[text]
    except:
        return text


def load_population_data():
    """Load and prepare country populations with same keys as covid19 data."""
    # Load population data
    populations_counties = pd.read_csv('https://www2.census.gov/programs-surveys/popest/datasets/' \
                                       '2010-2019/counties/totals/co-est2019-alldata.csv', encoding='ISO-8859-1')
    populations_states = populations_counties[populations_counties.STNAME == populations_counties.CTYNAME]
    populations_countries = pd.read_csv('countries.csv')

    # Load COVID-19 data
    state, county = load_local().values()
    world = load_raw_global()

    # Sets of keys from each data source
    covid19_keys = {'county': set(county.groupby(['County', 'State']).max().index.values),
                    'state': set(state.State.unique()),
                    'country': set(world['Confirmed'].groupby('Country/Region').max().index.values),
                    }

    pop_keys = {'county': set(populations_counties[~populations_counties
                              .duplicated(['STNAME', 'CTYNAME'])
                              ]
                              .groupby(['CTYNAME', 'STNAME'])
                              .max()
                              .index
                              ),
                'state': set(populations_states.groupby('STNAME').max().index),
                'country': set(populations_countries.country.unique())
                }

    # Preparing initial nation, state and county dataframes
    df_countries = populations_countries.applymap(convert2num)[['country', 'Population']]
    df_countries.rename(columns={'country': 'Country/Region'}, inplace=True)
    df_states = (populations_states[~populations_states[['STNAME', 'CTYNAME']]
                 .duplicated()][['STNAME', 'POPESTIMATE2019']]
                 .rename(columns={'STNAME': 'State', 'POPESTIMATE2019': 'Population'})
                 )

    territories = ['American Samoa', 'Guam', 'Northern Mariana Islands', 'Puerto Rico',
                   'Virgin Islands']
    df_territories = (df_countries[df_countries['Country/Region'].isin(territories)]
                      .rename(columns={'Country/Region': 'State'})
                      )
    df_states_territories = pd.concat([df_states, df_territories], ignore_index=True)

    #  Country Key Dictionary, including cruise ships
    country_search_space = df_countries['Country/Region']
    country_match_space = covid19_keys['country']
    covid19_pop_countries, unmatched_countries = find_aliases(country_match_space, country_search_space)
    pop_covid19_countries = {val: key for key, val in covid19_pop_countries.items()}
    # manual inputs
    pop_covid19_countries['United States'] = 'US'
    pop_covid19_countries['Congo, Republic of the'] = 'Congo (Brazzaville)'
    pop_covid19_countries['Congo, Democratic Republic of the'] = 'Congo (Kinshasa)'
    df_cruise_ships = pd.DataFrame([['The Diamond Princess', 3711], ['MS Zaandam', 1243]],
                                   columns=['Country/Region', 'Population'])
    df_global = df_countries.append(df_cruise_ships, ignore_index=True)
    # State Key Dictionary
    state_search_space = df_states_territories.State
    state_match_space = covid19_keys['state']
    covid19_pop_states, unmatched_states = find_aliases(state_match_space, state_search_space)
    pop_covid19_states = {val: key for key, val in covid19_pop_states.items()}

    # County Key Dictionary, including NYC entry
    county_search_space = county.County.unique()
    county_match_space = populations_counties.CTYNAME.unique()
    covid19_pop_counties, unmatched_counties = find_aliases(county_search_space, county_match_space)
    boroughs = ['Richmond County', 'Queens County', 'Kings County', 'Bronx County', 'New York County']
    inNYstate = populations_counties.STNAME == 'New York'
    same_counties = populations_counties.CTYNAME.isin(boroughs)
    inNYC = same_counties & inNYstate
    NYC_pop = populations_counties[inNYC].loc[:, ('CTYNAME', 'POPESTIMATE2019')]
    NYC_2019est = populations_counties[inNYC].loc[:, 'POPESTIMATE2019'].sum()
    df_NYC = pd.DataFrame({'CTYNAME': 'New York City', 'STNAME': 'New York',
                           'POPESTIMATE2019': NYC_2019est}, index=[0])
    populations_counties.append(df_NYC, ignore_index=True)
    covid19_pop_counties['New York City'] = 'New York City'
    pop_covid19_counties = {val: key for key, val in covid19_pop_counties.items()}

    pop_covid19_US = {**pop_covid19_counties, **pop_covid19_states}
    df_counties = (populations_counties
                   .append(df_NYC, ignore_index=True)
                   .loc[:, ('CTYNAME', 'STNAME', 'POPESTIMATE2019')]
                   .rename(columns={'CTYNAME': 'County', 'STNAME': 'State',
                                    'POPESTIMATE2019': 'Population'})

                   )

    df_counties.loc[:, ('County', 'State')] = (df_counties[['County', 'State']]
                                               .applymap(lambda x: translate(x, pop_covid19_US))
                                               )

    return df_global, df_states_territories, df_counties


def create_per_capita_features(data, population):
    """Merges data set with population data and creates per capita numerical features"""

    num_types = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    df = pd.merge(data, population)
    numerical_fields = [i for i in data.columns
                        if data[i].dtype in num_types]
    for i in numerical_fields:
        df[i + ' per capita'] = df[i] / df['Population']
    return df
