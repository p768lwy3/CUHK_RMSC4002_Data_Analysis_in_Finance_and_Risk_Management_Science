"
The script for RMSC4002 Asg3-1.
Dataset: credit.csv
"
# (0). Define functions for self-use.
# (i). scoring: to print error rate of classification
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

# (a). Read in credit.csv and save it in d.
d <- read.csv('credit.csv')    #Read in credit.csv and save it in d.
set.seed(960430)               #Use the last six digits of your birth date as the random seed
id <- sample(1:690, size=580)  #randomly sample 580 records from d as the training dataset
d1 <- d[id,]                   #and save it in d1
d2 <- d[-id,]                  #Save the other records in d2 as the testing dataset

# (b). Using the training dataset d1, build a classification tree of Result with other variables.
library(rpart)
ctree <- rpart(Result~Age+Address+Employ+Bank+House+Save+Result, 
               data=d1, method='class', maxdepth=3)


# (d). Produce the classification table and compute the training error rate.
pr <- predict(ctree)                # get prediction probabilty from ctree
cl <- max.col(pr) - 1               # turn it back into 0 or 1
ttrain <- table(cl, d1$Result)      # classification table
print(ttrain)                       # print table
scoring(ttrain)                     # print error rate

# (e) Apply the classification rules in (d) using the testing dataset d2 
#     and produce the corresponding classification table. 
#     Compute the testing error rate.
prtest <- predict(ctree, d2)        # predict with test set
cltest <- max.col(prtest) - 1       # turn it back into 0 or 1
ttest  <- table(cltest, d2$Result)  # classification table
print(ttest)                        # print table
scoring(ttest)                      # print error rate
