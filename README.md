# TVshows

#### Tools: BeautifulSoup, Seaborn, Pandas, Statsmodels

This personal project aims at forecasting the audience of three French TV shows, namely "Quotidien", "TPMP", and "C à vous" 
using advanced Machine Learning techniques. 

### Data
The daily audience data of the three shows is scraped from the web, from a [non-official website](http://www.leblogtvnews.com/2017/09/les-audiences-chaque-jour-de-tpmp-quotidien-et-c-a-vous.html). 
Thus, the data is unformatted and requires extensive pre-processing. The data spans from December 2017 to November 2019. 
Those TV shows only air on weekdays between 6pm and 9pm.
Note that the history of data available on the website is typically only 1 year old, i.e. historical data needs to be saved 
into an additional data file.

### EDA
Exploratory data analysis is performed and revealed a clear seasonality per month (less viewers during the Fall, and more 
viewers during the Winter or at the end of the Spring), but also per day of week (clear drop in audience on Friday).

### Forecasting
Given the seasonality of the audience data, a SARIMAX model is used to forecast the performance of the shows in the next few 
days. The parameters of the SARIMA model are found using grid search.
