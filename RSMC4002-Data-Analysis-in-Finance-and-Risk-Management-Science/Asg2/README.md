# RMSC4002  2017/18  1st term  Assignment 2
## Q2 </br>
The file “credit.csv” contains 7 columns and 690 records of credit application in a bank:  </br>

| No. | Column | Attribute Information: | name | value |
| ------------- | ------------- | ------------- | ------------- | ------------- |
|1|Age:|`Age`|continuous.|
|2|Mean time at address:|`Address`|continuous.|
|3|Mean time with employers:|`Employ`|continuous.|
|4|Time with bank:|`Bank`|continuous.|
|5|Monthly housing expense:|`House`|continuous.|
|6|Savings account balance:|`Save`|continuous.|
|7|Result:|`Result`|binary|

The first 6 columns are continuous while the last column is the result of the credit application (1=accept, 0=reject).</br>

  (a) Read in the dataset “credit.csv” and save it in d.</br>
  (b) Using the last 5 digits of your student ID as random seed, </br>
      partition d into training dataset d1 and testing dataset d2 as follow: </br>
```
> set.seed(xxxxx)			# use the last 5 digits of your student id
> id<-sample(1:690,size=600)		# generate random row index for d1
> d1<-d[id,]				# training dataset
> d2<-d[-id,]				# testing dataset
``` 
  </br>
  (c) Using d1 and the glm() function in R, fit a logistic regression of Result on other variables.</br>
      Exclude the insignificant variables step by step to arrive at a final model and save the output in lreg. </br>
      Produce the classification table for this logistic regression on the training dataset d1.</br>
  (d) Using the predict() function in R to produce the classification table on the testing dataset d2.</br>
  (e) Compute and compare the misclassification rate for training and testing dataset.</br>
  (f) Produce the lift chart for training dataset d1 as on p.11 of Chapter 5, </br>
      using cumulative percentage of success vs. the proportion (1:n)/n. (i.e. the second graph on p.11).</br>
  (g) Plot a similar lift chart for the testing dataset d2 on the same graph in (f). </br>
      [Hint: use lines() instead of plot() will add the line on the same graph in (f).]</br>
  (h) Compare and comment on these two lines in (f) and (g).</br>
