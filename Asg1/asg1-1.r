"
  This is the R script for Assignment 1 of RMSC4002 in CUHK.
  Date: 10/10/2017
  Dataset: stock(5).csv, asg1.csv(which will build in this file)
  Libary:
  1. tseries
  Method:
  1. Using Cholesky decompostion to form a multivariable normal dist.
"
## Import Lib:
library('tseries')

## Prepare for the dataset:
## Read Stock Number list:
path = 'hkse50(1).csv'
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



## Assg Q.1:
## Read dataset:
ds_path = 'asg1.csv'
ds<-read.csv(ds_path)

## Build TimeSeries datatype objects:
t1<-as.ts(ds$X0001.hk)
t2<-as.ts(ds$X0011.hk)
t3<-as.ts(ds$X0293.hk)
t4<-as.ts(ds$X0941.hk)
t5<-as.ts(ds$X1299.hk)

#	Compute the value V0 of portfolio
V0<-sum(tail(ds, 1))*5000

## Calculate daily pecentage Return:
u1<-(lag(t1)-t1)/t1
u2<-(lag(t2)-t2)/t2
u3<-(lag(t3)-t3)/t3
u4<-(lag(t4)-t4)/t4
u5<-(lag(t5)-t5)/t5
u<-cbind(u1, u2, u3, u4, u5) # combine into u

## Using Cholesky decompostion to generate Future Price:
set.seed(63766)
u60<-tail(u, 60)      # get the latest 60 days price
mu<-apply(u60,2,mean) # compute daily return rate
sigma<-var(u60)       # compute daily variance rate
C<-chol(sigma)        # compute Cholesky Decompostion
for(i in 1:10){       # simulate the price with future 10 days
  z<-rnorm(5)         # generate normal random vector
  v<-mu+t(C)%*%z      # transform to multivariate normal
  s1<-s0*(1+v)        # new stock price
  v1<-sum(s1)*5000    # simulated price of portfolio
  ds<-rbind(ds,s1)    # append s1 to ds
  v0<-rbind(v0,v1)    # append v0 to v1
  s0<-s1              # update s0 by s1
}

## Find min, max, mean, median, sd, lowest 1 and 5 percentile from this profit/loss distribution:
min(v0)
max(v0)
mean(v0)
median(v0)
sd(v0)
quantile(v0, c(.01, .05))

## Plot to see the result:
matplot(ds, type='l')
plot(as.ts(ds))
