import os
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_excel(os.path.join(BASE_DIR, 'data', 'Crime_Data.xlsx'))

modelo_edad = joblib.load(os.path.join(BASE_DIR, 'modelos', 'modelo_edad_rf.pkl'))
modelo_arresto = joblib.load(os.path.join(BASE_DIR, 'modelos', 'modelo_arresto_rf.bin'))
encoders = joblib.load(os.path.join(BASE_DIR, 'modelos', 'encoders_arresto.pkl'))      

le_sex = encoders['sex']
le_descent = encoders['descent']

areas = df[['AREA', 'AREA NAME']].drop_duplicates().sort_values('AREA NAME')
crimenes = df[['Crm Cd', 'Crm Cd Desc']].drop_duplicates().sort_values('Crm Cd Desc')
premisas = df[['Premis Cd', 'Premis Desc']].dropna().drop_duplicates().sort_values('Premis Desc')
armas = df[['Weapon Used Cd', 'Weapon Desc']].dropna().drop_duplicates().sort_values('Weapon Desc')

dias_semana = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves',
               4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}


fig_top_crimenes = px.bar(
    df['Crm Cd Desc'].value_counts().head(10).reset_index(),
    x='count', y='Crm Cd Desc', orientation='h',
    title='Top 10 tipos de crimen (2020-2021)',
    labels={'count': 'Casos', 'Crm Cd Desc': ''}
).update_layout(yaxis={'categoryorder': 'total ascending'})

fig_hora = px.histogram(
    df, x='Hora_OCC', nbins=24,
    title='Distribución de casos por hora del día',
    labels={'Hora_OCC': 'Hora'}
)

fig_area = px.bar(
    df.groupby('AREA NAME')['Arresto'].mean().sort_values(ascending=False).reset_index(),
    x='AREA NAME', y='Arresto',
    title='Tasa de arresto por área',
    labels={'Arresto': 'Tasa de arresto', 'AREA NAME': 'Área'}
)

fig_anio = px.bar(
    df.groupby(['Anio_OCC', 'Mes_OCC']).size().reset_index(name='casos'),
    x='Mes_OCC', y='casos', color='Anio_OCC', barmode='group',
    title='Casos por mes y año'
)


app = dash.Dash(__name__)
server = app.server 

app.layout = html.Div([
    html.H1('Dashboard de Criminalidad — LAPD 2020-2021'),

    html.H2('Análisis exploratorio'),
    html.Div([
        dcc.Graph(figure=fig_top_crimenes, style={'width': '48%'}),
        dcc.Graph(figure=fig_hora, style={'width': '48%'}),
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'}),
    html.Div([
        dcc.Graph(figure=fig_area, style={'width': '48%'}),
        dcc.Graph(figure=fig_anio, style={'width': '48%'}),
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'}),

    html.Hr(),
    html.H2('Controles de predicción'),

    html.Div([
        html.Label('Área'),
        dcc.Dropdown(id='in-area',
                     options=[{'label': r['AREA NAME'], 'value': r['AREA']} for _, r in areas.iterrows()],
                     value=areas['AREA'].iloc[0]),

        html.Label('Tipo de crimen'),
        dcc.Dropdown(id='in-crmcd',
                     options=[{'label': r['Crm Cd Desc'], 'value': r['Crm Cd']} for _, r in crimenes.iterrows()],
                     value=crimenes['Crm Cd'].iloc[0]),

        html.Label('Lugar (premisa)'),
        dcc.Dropdown(id='in-premisa',
                     options=[{'label': r['Premis Desc'], 'value': r['Premis Cd']} for _, r in premisas.iterrows()],
                     value=premisas['Premis Cd'].iloc[0]),

        html.Label('Arma usada'),
        dcc.Dropdown(id='in-arma',
                     options=[{'label': r['Weapon Desc'], 'value': r['Weapon Used Cd']} for _, r in armas.iterrows()],
                     value=armas['Weapon Used Cd'].iloc[0]),

        html.Label('Sexo de la víctima'),
        dcc.Dropdown(id='in-sexo',
                     options=[{'label': s, 'value': s} for s in le_sex.classes_],
                     value=le_sex.classes_[0]),

        html.Label('Descendencia de la víctima (código LAPD)'),
        dcc.Dropdown(id='in-descent',
                     options=[{'label': d, 'value': d} for d in le_descent.classes_],
                     value=le_descent.classes_[0]),

        html.Label('Edad de la víctima (para el modelo de arresto)'),
        dcc.Slider(id='in-edad', min=0, max=99, step=1, value=30,
                   marks={i: str(i) for i in range(0, 100, 10)}),

        html.Label('Mes'),
        dcc.Slider(id='in-mes', min=1, max=12, step=1, value=6,
                   marks={i: str(i) for i in range(1, 13)}),

        html.Label('Día de la semana'),
        dcc.Dropdown(id='in-dia',
                     options=[{'label': v, 'value': k} for k, v in dias_semana.items()],
                     value=0),

        html.Label('Hora del día'),
        dcc.Slider(id='in-hora', min=0, max=23, step=1, value=12,
                   marks={i: str(i) for i in range(0, 24, 4)}),
    ], style={'width': '420px', 'display': 'flex', 'flexDirection': 'column', 'gap': '6px'}),

    html.Hr(),
    html.Div(id='out-edad', style={'fontSize': 22, 'fontWeight': 'bold', 'marginTop': 10}),
    html.Div(id='out-arresto', style={'fontSize': 22, 'fontWeight': 'bold', 'marginTop': 10, 'color': '#b23'}),
])


@app.callback(
    Output('out-edad', 'children'),
    Output('out-arresto', 'children'),
    Input('in-area', 'value'),
    Input('in-crmcd', 'value'),
    Input('in-premisa', 'value'),
    Input('in-arma', 'value'),
    Input('in-sexo', 'value'),
    Input('in-descent', 'value'),
    Input('in-edad', 'value'),
    Input('in-mes', 'value'),
    Input('in-dia', 'value'),
    Input('in-hora', 'value'),
)
def predecir(area, crmcd, premisa, arma, sexo, descent, edad, mes, dia, hora):
    anio = 2021  

    fila_edad = pd.DataFrame([{
        'AREA': area, 'Part 1-2': 1, 'Crm Cd': crmcd,
        'Weapon Used Cd': arma, 'Premis Cd': premisa,
        'Anio_OCC': anio, 'Mes_OCC': mes, 'DiaSemana_OCC': dia, 'Hora_OCC': hora
    }])[modelo_edad.feature_names_in_]
    pred_edad = modelo_edad.predict(fila_edad)[0]


    sexo_cod = le_sex.transform([sexo])[0]
    descent_cod = le_descent.transform([descent])[0]

    fila_arresto = pd.DataFrame([{
        'AREA': area, 'Part 1-2': 1, 'Crm Cd': crmcd,
        'Vict Age': edad, 'Vict Sex': sexo_cod, 'Vict Descent': descent_cod,
        'Premis Cd': premisa, 'Weapon Used Cd': arma,
        'Anio_OCC': anio, 'Mes_OCC': mes, 'DiaSemana_OCC': dia, 'Hora_OCC': hora
    }])[modelo_arresto.feature_names_in_]

    prob_arresto = modelo_arresto.predict_proba(fila_arresto)[0][1]
    pred_arresto = modelo_arresto.predict(fila_arresto)[0]
    etiqueta = 'SÍ' if pred_arresto == 1 else 'NO'

    return (
        f'Edad estimada de la víctima: {pred_edad:.1f} años',
        f'Predicción de arresto: {etiqueta}  (probabilidad: {prob_arresto*100:.1f}%)'
    )


if __name__ == '__main__':
    app.run(debug=True)
