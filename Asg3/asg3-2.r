"
The script for RMSC4002 Asg3-2.
Dataset: credit.csv
"
# (0). Define functions for self-use.
# (i). ANN
library(nnet)
ann <- function(X, y, size, maxit=100, linout=F, try=5){     # define ann fn
  ann1 <- nnet(y~., data=X, size=size, maxit=maxit, linout=linout)
  v1   <- ann1$value                                         # save the result of first trial
  for (i in 2:try){                                          # try 'try'-times
    ann <- nnet(y~., data=X, size=size, maxit=maxit, linout=linout)
    if (ann$value < v1){                                     # check if the current values is better
      v1  <- ann$value                                       # save the best values
      ann1 <- ann                                            # save the result
    }
  }
  return(ann1)                                               # return the result
}
# (ii). scoring: to print error rate of classification
scoring <- function(table){
  tp <- table[2,2]
  fp <- table[1,2]
  fn <- table[2,1]
  precision <- tp/(tp+fp)
  recall    <- tp/(tp+fn)
  f1_score  <- 2*precision*recall/(precision+recall)
  errorrate <- (table[1,2]+table[2,1])/(sum(table))
  cat(sprintf("Error Rate = %f\n", errorrate))
  cat(sprintf("Precision = %f\n", precision))
  cat(sprintf("Recall = %f\n", recall))
  cat(sprintf("F1-Score = %f\n", f1_score))
  return(c(errorrate, precision, recall, f1_score))
}

# (a). Using the same dataset d1 and d2 in question 1 (a) as the training dataset and testing dataset. 
d <- read.csv('credit.csv')    #Read in credit.csv and save it in d.
set.seed(960430)               #Use the last six digits of your birth date as the random seed
id <- sample(1:690, size=580)  #randomly sample 580 records from d as the training dataset
d1 <- d[id,]                   #and save it in d1
d2 <- d[-id,]                  #Save the other records in d2 as the testing dataset

# Fit an improved version ann() function with size=6, linout=T, maxit=500 and try=25 and save the output to ann6.
ann6 <- ann(d1[, -which(names(d1) %in% c('Result'))], d1$Result,
            size=6, linout=T, maxit=500, try=25)

# (b). Repeat part (a) using size=7, 8 and 9. Save the result to ann7, ann8 and ann9 respectively.
ann7 <- ann(d1[, -which(names(d1) %in% c('Result'))], d1$Result,
            size=7, linout=T, maxit=500, try=25)
ann8 <- ann(d1[, -which(names(d1) %in% c('Result'))], d1$Result,
            size=8, linout=T, maxit=500, try=25)
ann9 <- ann(d1[, -which(names(d1) %in% c('Result'))], d1$Result,
            size=9, linout=T, maxit=500, try=25)

# (c). Compare the final value in ann6, ann7, ann8 and ann9.
cat(sprintf("ann6 value = %f\n", ann6$value))
cat(sprintf("ann7 value = %f\n", ann7$value))
cat(sprintf("ann8 value = %f\n", ann8$value))
cat(sprintf("ann9 value = %f\n", ann9$value))
# Choose the best (smallest) one and produce the classification table for the training dataset d1. 
pr <- round(ann9$fitted.values)
ttrain <- table(pr, d1$Result)
print(ttrain)

# Compute the training error rate.
scoring(ttrain)

# (d). Use the best ANN model in part (c), produce the classification table for the testing dataset d2 
# and hence compute the testing error rate.
prtest <- round(predict(ann9, d2))
ttest <- table(prtest, d2$Result)
print(ttest)
scoring(ttest)
