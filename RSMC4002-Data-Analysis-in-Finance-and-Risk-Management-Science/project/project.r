"
  This is the R script for group project of RMSC4002 in CUHK.
The course is about data analysis in Fincance and Risk Management
Science. Topics included GARCH model, PCA, logit, c-tree and ANN.
The project is to use the algorithms teached on the lessons to do
data analysis.

This project is the classification with Wine.

Ver: 0.0.1
Date: 17/11/2017
Dataset: train.csv
The dataset is from:https://www.kaggle.com/c/sf-crime/data
Method: 
1. Logistic Regression
2. Classification Tree
3. Artificial Neural Network
Libary:

"
"
i). Setup:
0. Load the library to use
1. Read the dataset to variable train
2. Print out some info of dataset
3. Define some function for self-use
"

# 0. Load the library to use.
library(ggplot)
library(nnet)
library(rpart)

# 1. Read the dataset to variable train
wine_r <- read.csv('winequality-red.csv', sep=';')    # read csv of red wine
wine_w <- read.csv('winequality-white.csv', sep=';')  # read csv of white wine
# 1.1. generate dummy variable 1 if red, else 0
wine_r['red'] <- 1
wine_w['red'] <- 0
# 1.2. merge them into train
train <- rbind(wine_r, wine_w)

# 2. Print out some info of dataset
#str(train)                               # print out info of dataset
#head(train, 5)                           # print the first 5 rows
#ncol(train)                              # print numbers of columns: 13
#nrow(train)                              # print numbers of rows: 6497
#colnames(train)                          # print the columns names
"
Rmk.
The training set include 13 columns,
1 - fixed acidity 
2 - volatile acidity 
3 - citric acid 
4 - residual sugar 
5 - chlorides 
6 - free sulfur dioxide 
7 - total sulfur dioxide 
8 - density 
9 - pH 
10 - sulphates 
11 - alcohol 
Output variable (based on sensory data): 
12 - quality (score between 0 and 10)
13 - red (if red, 0, else 1)
"

# 3. Define some function for self-use
# 3.1 Explore is a function to make bar plot with single variable to see the distribution.
explore<-function(x){                                         
  barplot(table(x),horiz=T,cex.names=0.7,las=2)
}
# 3.2 train_test_split is a function to split data into training and testing set.
train_test_split <- function(arrays, test_size=0.25, seed=NULL){
  if(!is.null(seed)) set.seed(seed)
  index <- 1:nrow(arrays)
  testindex <- sample(index, trunc(length(index)*test_size))
  testset  <- arrays[testindex, ]
  trainset <- arrays[-testindex, ]
  return(list(trainset, testset))
}
# 3.3 mdist is a function to calculate geodesic distances
mdist<-function(x){
  t<-as.matrix(x)
  m<-apply(t,2,mean)
  s<-var(t)
  mahalanobis(t,m,s)
}
# 3.4 Z-transformation
ztran<-function(var){
  return((var-mean(var))/sd(var))
}
# 3.5 Scoring: Precision, Recall, F1_score
scoring <- function(table){
     tp <- table[2,2]
     fp <- table[1,2]
     fn <- table[2,1]
     precision <- tp/(tp+fp)
     recall    <- tp/(tp+fn)
     f1_score  <- 2*precision*recall/(precision+recall)
     cat(sprintf("Precision = %f\n", precision))
     cat(sprintf("Recall = %f\n", recall))
     cat(sprintf("F1-Score = %f\n", f1_score))
     return(c(precision, recall, f1_score))
}

# ii). Data Visualization
# 1. Bar plot
# explore(train$fixed.acidity)
# explore(train$volatile.acidity)
# explore(train$citric.acid)
# explore(train$residual.sugar)
# explore(train$chlorides)
# explore(train$free.sulfur.dioxide)
# explore(train$total.sulfur.dioxide)
# explore(train$density)
# explore(train$pH)
# explore(train$sulphates)
# explore(train$alcohol)
# explore(train$quality)
# 2. Corr plot
# cor(train)                          # correlation matrix
# pairs(red~fixed.acidity+volatile.acidity+citric.acid+residual.sugar+chlorides+
#       free.sulfur.dioxide+total.sulfur.dioxide+density+pH+sulphates+alcohol+
#       quality, data=train, main="Correlation Plot")

# Rmk. Any things else?

# iii). Data Cleansing / Pre-Processing
# 1. Normalize large scale variables
train$fixed.acidity <- ztran(train$fixed.acidity)
train$free.sulfur.dioxide <- ztran(train$free.sulfur.dioxide)
train$total.sulfur.dioxide <- ztran(train$total.sulfur.dioxide)
train$pH <- ztran(train$pH)
train$alcohol <- ztran(train$alcohol)

# 2. Split data into training and testing set
train_split <- train_test_split(train, test_size=0.3, seed=6789)
train_X <- as.data.frame(train_split[1])
test_X <- as.data.frame(train_split[2])
#nrow(train_X)
#nrow(test_X)

# iv). Logit
# 1. fit a simple single logistics regression to classify wines.
lreg_red <- glm(red~fixed.acidity+volatile.acidity+citric.acid+residual.sugar+
                  chlorides+free.sulfur.dioxide+total.sulfur.dioxide+density+pH+
                  sulphates+alcohol+quality, data=train_X, family=binomial)
#summary(lreg_red)
#step(lreg_red)
#lreg_red_v2 <- glm(formula = red ~ volatile.acidity + citric.acid + residual.sugar + 
#                        chlorides + free.sulfur.dioxide + total.sulfur.dioxide + 
#                        density + sulphates + alcohol, family = binomial, data = train_X)

# 1.1. Classification table with training set
pr <- (lreg_red$fitted.values>0.5)
#pr_v2 <- (lreg_red_v2$fitted.values>0.5)
t1 <- table(pr, train$red)
#t2 <- table(pr_v2, train$red)
print(t1)
#print(t2)
scoring(t1)
#scoring(t2)

# 1.2. Classification table with testing set
c <- predict(lreg_red, test_X)
pi <- exp(c)/(1+exp(c))
pr <- (pi>0.5)
t3 <- table(pr, test_X$red)
print(t3)
scoring(t3)
# c <- predict(lreg_red_v2, test_X)
# pi <- exp(c)/(1+exp(c))
# pr <- (pi>0.5)
# t4 <- table(pr, test_X$red)
# print(t4)
# scoring(t4)

# 1.2. Life Chart
# 1.2.1. Cum sum
ysort <- train_X$red[order(lreg_red$fit,decreasing=T)]   # sort y according to lreg$fit
ylen  <- length(ysort)                                   # get length of y sort
percl <- cumsum(ysort)/(1:ylen)                          # compute cumulative percentage
plot(percl, type='l', col='blue')                        # plot perc with line type
abline(h=sum(train_X$red)/ylen)                          # add the baseline
yideal <- c(rep(1,sum(train_X$red)), 
            rep(0,length(train_X$red)-sum(train_X$red))) #the ideal case
percideal <-cumsum(yideal)/(1:ylen)                      # compute cumulative percentage of ideal case
lines(percideal, type='l', col='red')                    # plot the ideal case in red line

# 1.2.2. Cum precentage
perc2 <- cumsum(ysort)/sum(ysort)                        # cumulative perc. of success
pop   <- (1:ylen)/ylen                                   # x-coordinate
plot(pop, perc2, type='l', col='blue')                   # plot
lines(pop, pop)                                          # add the reference line
percideal2 <- cumsum(yideal)/sum(yideal)                 # cumulative perc. of success for ideal
lines(pop, percideal2, type='l', col='red')              # plot the ideal case in red line

# 2. fit a Multinomial Logit with quality classification
mnl <- multinom(quality~red+fixed.acidity+volatile.acidity+citric.acid+
                residual.sugar+chlorides+free.sulfur.dioxide+
                total.sulfur.dioxide+density+pH+sulphates+alcohol,
                data=train_X, maxit=300)                 # perform MNL
#summary(mnl)                                            # display MNL
prmnl<- predict(mnl)                                     # predicition
table(prmnl, train_X$quality)                            # tabulate pred vs quality

# 2.1. predict with testing set
prmnl_test <- predict(mnl, test_X)
table(prmnl_test, test_X$quality)

# 2.2. Outlier detection
x <- train_X[, -which(names(train_X) %in% c('quality'))] # select columns without quality
md <- mdist(train)                                       # compute mdist
#plot(md)                                                # plot md
(c<-qchisq(0.99, df=6))                                  # p=6, and type-I error = 0.01
train_X_v2 <- train_X[md<c,]                             # select cases from train_X with md<c
#dim(train_v2)                                           # we have throw 4548-3908=640 cases

mnl_v2 <- multinom(quality~red+fixed.acidity+volatile.acidity+citric.acid+
                   residual.sugar+chlorides+free.sulfur.dioxide+
                   total.sulfur.dioxide+density+pH+sulphates+alcohol,
                   data=train_X_v2, maxit=1500)          # using train_X_v2 to train a new MNL
#summary(mnl_v2)                                         # display MNL
prmnl2<- predict(mnl_v2)                                 # predicition
table(prmnl2, train_X$quality)                           # tabulate pred vs quality
# 2.2.1. predict with testing set
prmnl2_test <- predict(mnl_v2, test_X)
table(prmnl2_test, test_X$quality)

# 2.3. Dummy variable
g <- (train_X_v2 > 0) + 1                                # create dummy variable g=2 if ?>??, else g=1
mnl_v3 <- multinom(quality~g+fixed.acidity+volatile.acidity+citric.acid+
                   residual.sugar+chlorides+free.sulfur.dioxide+
                   total.sulfur.dioxide+density+pH+sulphates+alcohol+
                   g*fixed.acidity+g*volatile.acidity+g*citric.acid+
                   g*residual.sugar+g*chlorides+g*free.sulfur.dioxide+
                   g*total.sulfur.dioxide+g*density+g*pH+g*sulphates+g*alcohol,
                   data=train_X_v2, maxit=1500)          # using dummy variable to train a new MNL
prmnl3 <- predict(mnl_v3)                                # Predicted Values
table(prmnl3, train_X$quality)                           # Classification table
# Rmk. as using length g used in model is 3908, which is not same as length of test X,
#      So, it cannot predict test X directly.
"
g <- as.numeric(g[,1])
train_X_v2_d <- cbind(train_X_v2, g)
mnl_v3 <- multinom(quality~g+volatile.acidity+citric.acid+residual.sugar+
                   chlorides+free.sulfur.dioxide+total.sulfur.dioxide+
                   density+pH+sulphates+alcohol+g*volatile.acidity+g*citric.acid+
                   g*residual.sugar+g*chlorides+g*free.sulfur.dioxide+
                   g*total.sulfur.dioxide+g*density+g*pH+g*sulphates,
                   data=train_X_v2_d, maxit=2000)  
"

# iv). C-tree
# 1. Binary Classification with red and white wine
ctree <- rpart(red~fixed.acidity+volatile.acidity+citric.acid+residual.sugar+
                    chlorides+free.sulfur.dioxide+total.sulfur.dioxide+density+pH+
                    sulphates+alcohol+quality, data=train_X, method='class')
plot(ctree)                         # plot classification tree
text(ctree, use.n=T, cex=1)         # add text on ctree
print(ctree)                        # print the values of ctree
# 1.1. Classifiaction table checking
pr <- predict(ctree)                # get prediction probabilty from ctree
cl <- 0*(pr[,1]>0.5)+1*(pr[,2]>0.5) # turn it back into 0 or 1
t1 <- table(cl, train_X$red)        # classification table
print(t1)
scoring(t2)
pr <- predict(ctree, test_X)        # predict with test set
cl <- max.col(pr)                   # turn it back into 0 or 1
t2 <- table(cl, test_X$red)         # classification table
print(t2)
scoring(t2)


# 2. Multi class classification with quality by Ctree
ctree <- rpart(quality~fixed.acidity+volatile.acidity+citric.acid+residual.sugar+
                 chlorides+free.sulfur.dioxide+total.sulfur.dioxide+density+pH+
                 sulphates+alcohol+red, data=train_X, method='class')
plot(ctree)                         # plot classification tree
text(ctree, use.n=T, cex=0.6)       # add text on ctree

# 2.1. Classifiaction table checking
pr <- predict(ctree)                # get prediction probabilty from ctree
cl <- max.col(pr)                   # turn it back into 0 or 1
table(cl, train_X$quality)          # classification table
pr <- predict(ctree, test_X)        # predict with test set
cl <- max.col(pr)                   # turn it back into 0 or 1
table(cl, test_X$quality)           # classification table

# v). ANN
ann <- function(X, y, size, maxit=100, linout=F, softmax=FALSE, entropy=TRUE, try=5){     # define ann fn
  ann1 <- nnet(y~., data=X, size=size, maxit=maxit, linout=linout, softmax=softmax, entropy=entropy)
  v1   <- ann1$value                                                                      # save the result of first trial
  for (i in 2:try){                                                                       # try 'try'-times
    ann <- nnet(y~., data=X, size=size, maxit=maxit, linout=linout, softmax=softmax, entropy=entropy)
    if (ann$value < v1){                                                                  # check if the current values is better
        v1  <- ann$value                                                                  # save the best values
        ann1 <- ann                                                                       # save the result
    }
  }
  return(ann1)                                                                            # return the result
}
# .. category classification by ANN
train_y <- class.ind(train_X$quality)                                                     # split y for trainning
train_x <- train_X[, -which(names(train_X) %in% c('quality'))]                            # split x for trainning
nn_v0 <- nnet(train_x, train_y, size=9, maxit=1000, linout=F, softmax=TRUE, entropy=TRUE) # build ANN
#summary(nn_v0)                                                                           # summary
predicted_values_nn <- nn_v0$fitted.values                                                # get predicted probability
prnn <- colnames(predicted_values_nn)[apply(predicted_values_nn,1,which.max)]             # get columns of max prob as predicition
train_y_result <- colnames(train_y)[apply(train_y,1,which.max)]                           # get True values
table(prnn, train_y_result)                                                               # classificaion table

nn_v1 <- ann(train_x, train_y, size=16, maxit=1000, linout=F, try=10)                     # try 10 times
#summary(nn_v1)                                                                           # summary
predicted_values_nn1 <- nn_v1$fitted.values                                               # get predicted probability
prnn1 <- colnames(predicted_values_nn1)[apply(predicted_values_nn1,1,which.max)]          # get columns of max prob as predicition
table(prnn1, train_y_result)                                                              # classificaion table

"
library(devtools)                                                                         # library to get online source for R
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')
# source of plot.nnet
plot.nnet(nn_v0)
plot.nnet(nn_v1)
"
