# airbnb-analysis
Analysis of data from Seattle and Boston AirBNB homes.

## Motivation
Anwsering on 3 key questions:
* How much Airbnb homes costs in certain time frames?
* How much Airbnb homes costs in certain areas?
* What other factors influence prices?
For this perpuse I had to get data, clean it, handle NaN values, create charts and decision tree price-model. 

## Installation neccecery libraries

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install all libraries.

```bash
pip install pandas
pip install numpy
pip install seaborn
pip install matplotlib
pip install scikit-learn
```

## Project structure
* data folder contains all the data used
* airbnb-data-analysis.ipynb - main file with all charts and data transformation
* utils.py - help file with custom made functions

## Conclusions from the analysis
* How much Airbnb homes costs in certain time frames?
  Homes are more expensive during Summer and chepper during Winter.
  If you are short on Budget it is good to awoid weekends when pices are 4-6% higher.
* How much Airbnb homes costs in certain areas?
  In Boston you wil pay around $200 per night and in Seattle around $130.
* What other factors influence prices?
  Key factors are: privacy of accommodation, amount of bedroom, amount of bethrooms, location, Internet access.

Full analsis is on Medium.
https://medium.com/@chrapkus/what-factors-affect-airbnb-prices-in-boston-and-seattle-11c815a39622
