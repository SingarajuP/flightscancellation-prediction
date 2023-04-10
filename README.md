# flights cancellation prediction
<br />The deployed app can be accessed in the link below.
<br />[Flight cancellation prediction](https://singarajup-flightscancellation-prediction-app-06n6fx.streamlit.app/)


This is a mid-term project at Lighthouse Labs.
The data is obtained from Lighthouse Labs.

## Aim:

Regression Problem: The goal is to predict delay of flights.
Multiclass Classification: If the plane is delayed, predict what type of delay it is (will be).
Binary Classification: The goal is to predict if the flight will be cancelled.
Data: The data is for the air travel industry and is obtained from Lighthouse Labs in a postgres database. There are four separate tables:

flights: The departure and arrival information about flights in US in years 2018 and 2019 and January 2020. January 2020 will be used for evaluation.
fuel_comsumption: The fuel comsumption of different airlines from years 2015-2019 aggregated per month.
international_passengers: The passenger totals on different international routes from years 2015-2019 aggregated per month.
domestic_passengers: The passenger totals on different domestic routes from years 2015-2019 aggregated per month.

The web app has been developed for the cancellation of flights with a probability of cancellation. 

## Milestones
<br>To improve the model for the classification of the flights
<br>To make a web app for the delay of the flights
<br>To get weather data to predict the delay and cancellation of the flights
## Usage
Clone repo 
```bash
 git clone https://github.com/SingarajuP/flights_delay_cancellation_prediction.git
```
<br />Setup a virtual environment
```bash
conda create -n yourenvname python=3.10.9
```
<br />Activate the virtual environment

```bash
conda activate yourenvname
```
<br />Install all requirements using pip:
```bash
pip install -r requirements.txt
```
<br />To run web application stay in the main directory and run the command:
```bash
streamlit run app.py
It will open a web page in the browser 

```
