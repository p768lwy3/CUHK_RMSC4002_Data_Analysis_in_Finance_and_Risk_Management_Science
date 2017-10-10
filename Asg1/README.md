# RMSC4002  2017/18  1st term  Assignment 1

Q1. 
The file “hkse50.csv” contains 50 names and codes of stocks list in the main board of Hong Kong Stock Exchange. 
Use the last 5 digits of your student ID as random seed and select 5 stocks randomly from the list.
For example, if the last 5 digits of your student ID is 12345,
    
        stock<-read.csv("hkse50.csv")		  # read in data
        set.seed(12345)     # set random seed
        r<-sample(1:50,size=5)     # select 5 random integers
        stock[r,]     # list the 5 selected stocks
            
           code            name
        37 1044    Hengan Int'l
        43 1880     Belle Int'l
        50 3988   Bank of China
        42 1398            ICBC
        21  291 China Resources


In chapter one of my notes, I have generated one single path of future 90 days of stock prices for HSBC, CLP and CK.
Now you are going to perform simulation on these 5 stocks. 
Imagine we have a portfolio of 5,000 shares of each of these 5 stocks. 
Now we want to generate 1000 random paths of future 10 days of these 5 stocks and hence compute the value of this portfolio based on these simulated prices. 
Modify my R codes according to the following:
    
    1.	In the internet, search and download the adjusted daily closing prices of the 5 selected stocks from 1/1/2015 to 31/12/2016. 
        (In the tseries library, there is a function get.hist.quote() can download stock price easily, 
        see help(get.hist.quote) for more details).
    2.	Compute the value of your portfolio based on the closing price of the last day, 31/12/2016, say, v0 .
    3.	Using the last 5 digits of your student id as initial seed, 
        set up a loop to generate 1000 random paths of the prices for these 5 stocks for future 10 days. 
        Use the last 60 days in your dataset to estimate the mean vector and covariance matrix in your simulation.
    4.	Save the last simulated stock prices and compute the portfolio value based on these simulated stock prices. 
        Compute the profit/loss by simulated stock prices – v0.
    5.	Find the min, max, mean, median, sd, lowest 1 and 5 percentile from this profit/loss distribution.
