# Importing the libraries
import pandas as pd
import dash
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input,Output,State
import dash_core_components as dcc
import webbrowser
import plotly.express as px
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


df=pd.read_csv('scrappedReviewsPrediction.csv')

graph=px.pie(data_frame=df,values=[df['prediction'].value_counts()[1],df['prediction'].value_counts()[0]],
       names=['Positive Reviews','Negative Reviews'],
       color=['Positive Reviews','Negative Reviews'],
       color_discrete_sequence=['lightgreen','red'],width=500,height=400,hole=0.5)


# Declaring Global variables
app=dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
project_name=None
pickle_model=None
vocab=None
scrappedReviews=None




# Defining My Functions

def load_model():
    global scrappedReviews
    global pickle_model
    global vocab
    
    
    scrappedReviews=pd.read_csv('etsyScrappedReview.csv')['reviews']
    scrappedReviews.head()
    
    file1=open('pickle_model.pkl','rb')
    pickle_model=pickle.load(file1)
    
    file2=open('feature.pkl','rb')
    vocab=pickle.load(file2)
    
def check_review(reviewText):
    transformer=TfidfTransformer()
    loaded_vect=CountVectorizer(decode_error='replace',vocabulary=vocab)
    vectorised_review=transformer.fit_transform(loaded_vect.fit_transform([reviewText]))
            
    return pickle_model.predict(vectorised_review)
    
    

def open_browser():
    webbrowser.open_new('http://127.0.0.1:8050/')

def create_app_ui():
    main_layout=dbc.Jumbotron([
        
            html.H1(children="Sentiment Analysis",
                    style={
                            'backgroundColor':'#e7696f',
                            'textAlign':'center',
                            'font-size':'35px',
                            'height':'50px'
                        }),
            html.Div([
                    html.H1(children="Word Cloud",style={'font-size':'30px','margin-top':'30px'}),
                    dbc.Button(id='allbtn',children="All Words", outline=True, color="primary", className="mr-1",n_clicks_timestamp=0,style={'margin-top':'15px'}),
                    dbc.Button(id='posbtn',children="Positive Words", outline=True, color="success", className="mr-1",n_clicks_timestamp=0,style={'margin-top':'15px','padding':'10px'}),
                    dbc.Button(id='negbtn',children="Negative Words", outline=True, color="danger", className="mr-1",n_clicks_timestamp=0,style={'margin-top':'15px'}),
                    
                ],style={'textAlign':'center'}),
            html.Div(id='img',style={'textAlign':'center','margin-top':'30px'}),
            html.Div([html.H1(children="Distribution of Positive and Negative Reviews",style={'backgroundColor':'#e7696f','textAlign':'center','font-size':'35px','height':'50px','margin-top':'35px'})]),
            html.Br(),
            dcc.Graph(figure=graph,id='pie',className='d-flex justify-conten-center'),
            html.Br(),html.Hr(),html.Br(),
            
            html.Div(
                    [
                        html.H1(children="Please enter your reviews for the product",style={'backgroundColor':'#e7696f','textAlign':'center','font-size':'35px','height':'50px','margin-top':'10px'}),
                        html.Br(),html.Br(),
                        dcc.Textarea(id='textarea_review',placeholder="Enter your review here!!",required="required",style={'width':'100%','height':100}),
                        html.Br(),html.Br(),
                    ]),
            html.Div(
                [
                    dbc.Button(children="Submit",id='submit_review',color='dark',n_clicks=0,className="mt-2 mb-3"),
                    html.Div(id='h1_review',children=None)
                ]
                ,className='text-center'
                ),html.Br(),html.Br(),
            html.Div(
                [
                html.H1(children="Select an existing review",style={'backgroundColor':'#e7696f','textAlign':'center','font-size':'35px','height':'50px'}),
                html.Br(),html.Br(),
                dcc.Dropdown(id='dropdown',
                    options=[{'label': i[:150] , 'value': i} for i in scrappedReviews],
                    
                    searchable=False
                    ),html.Br(),html.Br(),
            html.Div(id='review_result',children=None)
                ],className='text-center'),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),
            
            
            
            ],style={'margin':'35px'})
    return main_layout


@app.callback(
    
    Output('img','children'),
    [
     Input('allbtn','n_clicks_timestamp'),
     Input('posbtn','n_clicks_timestamp'),
     Input('negbtn','n_clicks_timestamp')
     ])
def wordcloudbutton(allbtn,posbtn,negbtn):
    
    if int(allbtn)>int(posbtn) and int(allbtn)>int(negbtn):
        print(allbtn)
        return html.Div([
            html.Img(src=app.get_asset_url('All.png'))])
    
    elif int(posbtn)>int(allbtn) and int(posbtn)>int(negbtn):
        print(posbtn)
        return html.Div([
            html.Img(src=app.get_asset_url('Poscloud.png'))])
    
    elif int(negbtn)>int(allbtn) and int(negbtn)>int(posbtn):
        print(negbtn)
        return html.Div([
            html.Img(src=app.get_asset_url('Negcloud.png'))])
    else:
        pass
    
@app.callback(
    
    Output('review_result','children'),
    [
     
     Input('dropdown','value')
     ]

    )    
def handle_dropdown_review(dropdown_review_text):
    print("Data Type = ", str(type(dropdown_review_text)))
    print("Value = ", str(dropdown_review_text))
    
    if dropdown_review_text !=None:
        responce=check_review(dropdown_review_text)
        
        
        
        if responce[0]==0:
            return dbc.Alert(children='Negative',color='danger',style={'margin-top':'25px'})
            
        elif responce[0]==1:
            
            return dbc.Alert(children='Positive',color='success',style={'margin-top':'25px'})
       
        else:
            return dbc.Alert(children='Unknown',color='dark',style={'margin-top':'25px'})
        
    else:
        pass
    
@app.callback(
    Output('h1_review','children'),
    [
     Input('submit_review','n_clicks')
     ],
    State('textarea_review','value')
    
    )
def handle_review(n_clicks,textarea_value):
    print("Data Type = ", str(type(n_clicks)))
    print("Value = ", str(n_clicks))
    
    print("Data Type = ", str(type(textarea_value)))
    print("Value = ", str(textarea_value))
   
    
    if n_clicks>0:
        if textarea_value.isnumeric() !=True:
            responce=check_review(textarea_value)
            if responce[0]==0:
                return dbc.Alert(children='Negative',color='danger',style={'margin-top':'25px'})
            
            elif responce[0]==1:
                return dbc.Alert(children='Positive',color='success',style={'margin-top':'25px'})
        else:
            return dbc.Alert(children='Unknown',color='dark',style={'margin-top':'25px'})
    else:
        pass


# Main Function to control the Flow of your Project
def main():
    project_name="Sentiment Analysis with Insights"
    load_model()
    open_browser()
    app.title=project_name
    app.layout=create_app_ui()
    app.run_server()




# Calling the main function 

if __name__=='__main__':
    main()


