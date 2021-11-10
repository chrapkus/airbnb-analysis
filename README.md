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
  ![yearly_sesonality](https://user-images.githubusercontent.com/38725851/141102477-b5c62da0-a17e-431f-9b35-e3c4ee45191f.png)
  If you are short on Budget it is good to awoid weekends when pices are 4-6% higher.
  ![weakly_sesonality](https://user-images.githubusercontent.com/38725851/141102503-c0ce1569-6f13-463c-9a71-a404817edcc2.png)
* How much Airbnb homes costs in certain areas?
  In Boston you wil pay around $200 per night and in Seattle around $130.
  ![Boston](https://user-images.githubusercontent.com/38725851/141102545-f8283359-ef72-4c9e-ac87-ec238ee30af4.png)
  ![Seattle](https://user-images.githubusercontent.com/38725851/141102557-ec322ed6-dd42-4f59-abab-6f1e7b6bdcb4.png)
* What other factors influence prices?
  ![feature_model](https://user-images.githubusercontent.com/38725851/141102579-195edd7b-7ffd-497e-a3a8-6eff12f89e60.PNG)
  Key factors are: privacy of accommodation, amount of bedroom, amount of bethrooms, location, Internet access.

Full analsis is on Medium.
https://medium.com/@chrapkus/what-factors-affect-airbnb-prices-in-boston-and-seattle-11c815a39622
