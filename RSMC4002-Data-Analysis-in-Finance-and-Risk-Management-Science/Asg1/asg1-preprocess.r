"
  This is the R script for Assignment 1 to download the data of RMSC4002 in CUHK.
Ver: 0.0.1
Date: 7/9/2017
Dataset: stock(5).csv
Libary:
1. tseries
"
## Import Lib:
library('tseries')

## Prepare for the dataset:
## Read Stock Number list:
path = 'C:/Users/LIWAIYIN/Desktop/hkse50(1).csv'
stock<-read.csv(path)		# read in data
set.seed(63766)					# set random seed

## Since the dataset is not avalible now.
## So, choose the stock manually.
# r<-sample(1:50,size=5)			# select 5 random integers
r<-c(1, 7, 22, 35, 41)
selected<-stock[r,]						# list the 5 selected stocks

## Download the dataset and Export as "asg1.csv'
ts0<-ts()
for(st in selected[,1]){
  head<-paste(rep("0", 4-nchar(st)), collapse = "")
  stockno<-paste(head, st, '.hk', sep="")
  print(stockno)
  ts1<-get.hist.quote(instrument=stockno, quote=c("Adjusted"), provider="yahoo", start="2015-01-01", end="2016-12-31")
  names(ts1)[1] <- stockno
  print(head(ts1, 5))
  ts0<-merge.zoo(ts0,ts1)
  ## Remove first column
  ts0<-ts0[-1,]
  ts0$ts0<-NULL
}
print(head(ts0,5))
write.csv(ts0, file="asg1.csv", sep=',', row.names=FALSE)
