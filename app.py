import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import State, Input, Output
from dash.exceptions import PreventUpdate
import plotly.express as px
import pandas as pd
from helper_functions import create_per_capita_features, load_raw_global, load_local, \
    load_merged_daily_global, load_merged_daily_local
from typing import List, Iterable
import pickle

# Load all data
files = ['pop_countries.pkl', 'pop_states.pkl', 'pop_counties.pkl']
df_countries, df_states, df_countes = [pickle.load('Data/' + file) for file in files]


def load_data():
    """Load all processed COVID-19 data"""
    data = load_raw_global()
    tidy_global = load_tidy_global(data)
    merged_global = load_merged_daily_global(tidy_global)

    # Load local data, and daily data dictionaries
    state, county = load_local().values()
    merged_state = load_merged_daily_local(state)
    merged_county = load_merged_daily_local(county)

    # Create per capita features on merged data sets
    datasets = [merged_global, merged_state.drop(columns='fips'), merged_county.drop(columns='fips')]
    populations = [df_countries, df_states, df_counties]
    world_data, state_data, county_data = [create_per_capita_features(data, pop)
                                   for data, pop in zip(datasets, populations)]
    return world_data, state_data, county_data

def create_plot(type: str, countries :List[str], states: List[str]=[], counties: List[str]=[],
                y_axis: str='linear', per_capita: str='no'):
    """

    Parameters
    ----------
    type : str {'Cases', 'Deaths', 'Recovered'}
        Data to plot
    countries : list of strings
        Countries to plot
    states : list of strings
        States to plot
    counties : list of strings
        Counties to plot
    y_axis : string {'linear', 'log'}
        y axis scale
    per_capita : string {'yes', 'no'}
        Option to normalize data by population

    """
    if per_capita == 'no':
    data = pd.concat([])
    fig = px.line()




