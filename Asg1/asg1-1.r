"
  This is the R script for Assignment 1 of RMSC4002 in CUHK.
  Ver: 0.0.1
  Date: 7/9/2017
  Dataset: asg1.csv (from asg1-preprocess.r)
  Libary:
  1. tseries
  Method:
  1. Using Cholesky decompostion to form a multivariable normal dist.
"

## Assg Q.1:
## Read dataset:
ds_path = 'C:/Users/LIWAIYIN/Desktop/asg1.csv'
ds<-read.csv(ds_path)

## Build TimeSeries datatype objects:
t1<-as.ts(ds$X0001.hk)
t2<-as.ts(ds$X0011.hk)
t3<-as.ts(ds$X0293.hk)
t4<-as.ts(ds$X0941.hk)
t5<-as.ts(ds$X1299.hk)

#	Compute the value V0 of portfolio
v0<-sum(tail(ds, 1))*5000

## Calculate daily pecentage Return:
u1<-(lag(t1)-t1)/t1
u2<-(lag(t2)-t2)/t2
u3<-(lag(t3)-t3)/t3
u4<-(lag(t4)-t4)/t4
u5<-(lag(t5)-t5)/t5
u<-cbind(u1, u2, u3, u4, u5) # combine into u

## Using Cholesky decompostion to generate Future Price:
set.seed(63766)
u60<-tail(u, 60)         # get the latest 60 days price
mu<-apply(u60,2,mean)    # compute daily return rate
sigma<-var(u60)          # compute daily variance rate
C<-chol(sigma)           # compute Cholesky Decompostion
s0<-ds[494,]             # set s0 to the most recent price 
for(i in 1:10){          # simulate the price with future 10 days
  z<-rnorm(5)            # generate normal random vector
  v<-mu+t(C)%*%z         # transform to multivariate normal
  s1<-s0*(1+v)           # new stock price
  v1<-sum(s1)*5000       # simulated price of portfolio
  ds<-rbind(ds,s1)       # append s1 to ds
  v0<-rbind(v0,v1)       # append v0 to v1
  s0<-s1                 # update s0 by s1
}

## Find min, max, mean, median, sd, lowest 1 and 5 percentile from this profit/loss distribution:
min(v0)
max(v0)
mean(v0)
median(v0)
sd(v0)
quantile(v0, c(.01, .05))

## Plot to see the result:
matplot(ds, type='l')    # plot to see the stocks in one map   
plot(as.ts(ds))          # plot to see the stocks seperately    
plot(as.ts(v0))          # plot to see the portfolio result
