"
The script for RMSC4002 Asg2.
Dataset: credit.csv
"
# (a). Read-in the dataset "credit.csv" and save it in d
d <- read.csv('credit.csv')    #read dataset "credit.csv" as d

# (b). Using the last 5 digits of your student ID as random seed, partition d into training
#      dataset d1 and testing dataset d2 as follow:
set.seed(63766)                #use the last 5 digits of student id as seed
id <- sample(1:690, size=600)  #generate random row index for d1
d1 <- d[id,]                   #training dataset
d2 <- d[-id,]                  #testing dataset

# (c). using d1 and glm() to fit a logistic regression of Result on the other variables.
# lreg model,
lreg <- glm(Result~Age+Address+Employ+Bank+House+Save, data=d1, binomial(link="logit"))
summary(lreg)             #since ... <- how to decision to drop out? 
anova(lreg, test="Chisq") #to check sig. by chisq test
# the most important varibles: Employ, Bank, Save

# New lreg model,
lreg <- glm(Result~Employ+Bank+Save, data=d1, binomial(link="logit"))
names(lreg) #display the items in lreg

# Produce the classifaction table for this logistic regression on d1.
pr1  <- (lreg$fitted.values>0.5) #set pr1=True if fitted > 0.5 or otherwise
t1   <- table(pr1, d1$Result)    #classification table of d1
p_d1 <- t1[1,1]/sum(t1[1,])      #precision = TP/(TP+FP)
r_d1 <- t1[1,1]/sum(t1[,1])      #recall = TP/(TP+FN)
f1_d1 <- 2*p_d1*r_d1/(p_d1+r_d1) #F1 score
m_d1 <- (t1[1,2]+t1[2,1])/sum(t1)#misclassification rate of d1

# (d). using predict() to produce the classification table on the testing dataset d2
pv_d2 <- predict.glm(lreg, newdata=d2) #save predicted values of d2 with lreg. 
pr2 <- (pv_d2>0.5)                     #set pr2=True if predicted > 0.5 or otherwise
t2  <- table(pr2, d2$Result)           #classification table of d2
p_d2 <- t2[1,1]/sum(t2[1,])            #precision = TP/(TP+FP)
r_d2 <- t2[1,1]/sum(t2[,1])            #recall = TP/(TP+FN)
f1_d2 <- 2*p_d2*r_d2/(p_d2+r_d2)       #F1 score
m_d2 <- (t2[1,2]+t2[2,1])/sum(t2)      #misclassification rate of d2

# (f). compute life chart for training dataset d1, using cumulative percentage
#   of success vs. the proportion (1:n)/n
ysort_d1 <- d1$Result[order(lreg$fit, decreasing=T)]   #sort y according to lreg$fit
n_d1 <- length(ysort_d1)                                  #get length of ysort
percl_d1 <- cumsum(ysort_d1)/(1:n_d1)                  #compute cumlative percentage
plot(percl_d1, type="l", col="blue")                   #plot perc with line type
abline(h=sum(d1$Result)/n_d1)                          #add the baseline
yideal_d1 <- c(rep(1, sum(d1$Result)), rep(0, length(d1$Result)-sum(d1$Result))) #the ideal case
perc_ideal_d1 <- cumsum(yideal_d1)/(1:n_d1)            #compute cumulative percentage of ideal case
lines(perc_ideal_d1, type="l", col="red")              #plot the ideal case in red line

# (g). plot a similar lift chart for the testing dataset d2 on the same graph in (f).
ysort_d2 <- d2$Result[order(pv_d2, decreasing=T)]   #sort y according to predict values of lreg as pr2
n_d2 <- length(ysort_d2)                                  #get length of ysort
percl_d2 <- cumsum(ysort_d2)/(1:n_d2)                  #compute cumlative percentage
lines(percl_d2, type="l", col="green")                   #plot perc with line type
abline(h=sum(d2$Result)/n_d2)                          #add the baseline
yideal_d2 <- c(rep(1, sum(d2$Result)), rep(0, length(d2$Result)-sum(d2$Result))) #the ideal case
perc_ideal_d2 <- cumsum(yideal_d2)/(1:n_d2)            #compute cumulative percentage of ideal case
lines(perc_ideal_d2, type="l", col="brown")              #plot the ideal case in red line

