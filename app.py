import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import plotly.express as px
import pandas as pd
from helper_functions import create_per_capita_features, load_raw_global, load_local, \
    load_merged_daily_global, load_merged_daily_local, sieve, load_tidy_global, top_n_locations
from typing import Iterable

external_stylesheets = ['https://maxcdn.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


# Load all data
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
    files = ['pop_countries.csv', 'pop_states.csv', 'pop_counties.csv']
    populations = [pd.read_csv('Data/' + file, index_col=0) for file in files]
    world_data, state_data, county_data = [create_per_capita_features(data, pop)
                                           for data, pop in zip(datasets, populations)]
    world_data['Location'] = world_data['Country/Region']
    state_data['Location'] = state_data['State']
    county_data['Location'] = county_data['County'] + ', ' + county_data['State']
    world_data.loc[world_data.Location == 'Georgia', 'Location'] = 'Georgia (country)'
    return world_data, state_data, county_data


world_data, state_data, county_data = load_data()
top10_countries = top_n_locations(world_data, feature='Confirmed')


def create_plot(content: str = 'Confirmed', countries: Iterable = top10_countries,
                states: Iterable = [], counties: Iterable = [],
                log_y: str = 'linear', per_capita: str = 'total'):
    """

    Parameters
    ----------
    content : str {'Confirmed', 'Deaths', 'Recovered'}
        Data to plot
    countries : list of strings
        Countries to plot
    states : list of strings
        States to plot
    counties : list of strings
        Counties to plot
    log_y : string {'linear', 'log'}
        y axis scale
    per_capita : string {'total', 'per capita'}
        Option to normalize data by population

    Returns
    -----------
    fig : object
        The plot output
    """
    data = {'countries': sieve(world_data, 'Location', countries),
            'states': sieve(state_data, 'Location', states),
            'counties': sieve(county_data, 'Location', counties)}
    filtered_data = pd.concat(data.values(), join='inner')

    if per_capita !='total':
        content = ' '.join([content, 'per capita'])
    y_axis = log_y != 'linear'
    fig = px.line(filtered_data, x='Date', y=content, log_y=y_axis,
                  color='Location', hover_name='Location', title=content)

    return fig


def create_y_radio():
    return dcc.RadioItems(
        options=[
            {'label': 'log', 'value': 'log'},
            {'label': 'linear', 'value': 'linear'}
        ],
        value='linear',
        id='y-axis'
    )


def create_content_radio():
    return dcc.RadioItems(
        options=[{'label': val, 'value': val}
                 for val in ['Confirmed', 'Deaths', 'Recovered']],
        value='Confirmed',
        labelStyle={'display': 'inline-block'},
        id='content-radio'
    )


def create_per_capita_element():
    return dcc.RadioItems(
        options=[{'label': 'Total', 'value': 'total'},
                 {'label': 'per capita', 'value': 'per capita'}
                 ],
        id='per-capita-radio',
        value='total'
    )


def create_location_select(data, id):
    return dcc.Dropdown(
        options=[{'label': location, 'value': location}
                 for location in data['Location'].unique()],
        multi=True,
        id=id
    )


app.layout = html.Div(
    #className='container',
    children=[
        html.Div(
            html.Div([create_content_radio()]),
            id='row1',
            className='row-sm-12 text-center',
        ),
        html.Div(
            id='row2',
            className='four columns card',
            children=[
                html.Div(children=[html.H6('Choose y-axis Scaling'),
                                   create_y_radio()],
                         className='two columns card'
                         ),
                html.Div(
                    dcc.Graph(
                        figure=create_plot(),
                        id='plot1',
                    ),
                    className='col-sm-8'
                ),
                html.Div(
                    create_per_capita_element(),
                    id='per-capita-div',
                    #className='col-sm-2'
                )
            ],
        ),
        html.Div(id='row3',
                 className='row-sm-12',
                 children=[
                     html.Div(id='location-container',
                              children=[
                                  html.Div(
                                      id='loc-sel-col1',
                                      className='col-sm-4 padding-top-bot',
                                      children=[html.H6('Choose Countries:'),
                                                create_location_select(world_data, 'countries-multiselect'),
                                                html.H6('Choose States:'),
                                                create_location_select(state_data, 'states-multiselect')
                                      ]
                                  ),

                                  html.Div(
                                      id='loc-sel-col3',
                                      className='col-sm-4',
                                      children=[html.H6('Choose Counties:'),
                                                create_location_select(county_data, 'counties-multiselect')]
                                  )
                              ])
                 ]
        ),

    ]
)


@app.callback(
    [Output('plot1', 'figure')],
    [Input('content-radio', 'value'),
     Input('countries-multiselect', 'value'),
     Input('states-multiselect', 'value'),
     Input('counties-multiselect', 'value'),
     Input('y-axis', 'value'),
     Input('per-capita-radio', 'value')
     ]
)
def update_plot(content: str, countries: Iterable, states: Iterable, counties: Iterable,
                log_y: bool, per_capita: bool):
    """
    Update plot based on figure

    Parameters
    ----------
    content : str {'Confirmed', 'Deaths', 'Recovered'}
        Data to plot
    countries : list of strings
        Countries to plot
    states : list of strings
        States to plot
    counties : list of strings
        Counties to plot
    log_y : string {'linear', 'log'}
        y axis scale
    per_capita : string {'yes', 'no'}
        Option to normalize data by population

    Returns
    -------
    fig: object
        The plot with updated
    """
    try:
        return create_plot(content, countries, states, counties, log_y, per_capita)
    except:
        raise PreventUpdate


if __name__ == '__main__':
    app.run_server(debug=True, port=3004)
