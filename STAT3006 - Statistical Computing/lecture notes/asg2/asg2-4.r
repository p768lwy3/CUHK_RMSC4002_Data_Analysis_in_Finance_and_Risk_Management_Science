

set.seed(12345)

# read data
x <- read.csv('salary_data.txt', sep=' ', 
              col.names=c('Salary', 'Age.Indicator'))

# visualization
boxplot(Salary~Age.Indicator, data=x, ylab='Number', xlab='Age Group')
hist(x[x$Age.Indicator==1, ]$Salary, breaks=20, freq=F)
hist(x[x$Age.Indicator==2, ]$Salary, breaks=20, freq=F)
hist(x[x$Age.Indicator==3, ]$Salary, breaks=20, freq=F)
hist(subset(x, Age.Indicator == 2| Age.Indicator == 3)$Salary,
    breaks=20, freq=F)

# stratified sampling
require(dplyr)
foo <- split(x, x$Age.Indicator)
ssize <- sapply(foo, nrow)
psize <- nrow(x)
weight <- ssize / psize
smpsize <- 100
smplist <- round(smpsize * weight)
foo.smp <- list()
for(i in seq(along=foo)){
  foo.smp[i] <- sample_n(foo[[i]], smplist[i])
}
smpmns <- sapply(foo.smp, mean)
smpvar <- sapply(foo.smp, var)
for(i in 1:3){
  cat(
    sprintf(
      'Group #%s: Mean: %.4f, Std:%.4f \n', 
      i, smpmns[i], sqrt(smpvar[i])
    )
  )
}

smpsize <- 1000
smplist <- round(smpsize * weight)
cat(
  sprintf(
    'No of samples in each group: (%d, %d, %d) \n',
    smplist[1], smplist[2], smplist[3]
  )
)
foo.smp <- list()
for(i in seq(along=foo)){
  foo.smp[i] <- sample_n(foo[[i]], smplist[i])
}
smpmns <- sapply(foo.smp, mean)
mean.est <- (1 / psize) * sum(smpmns * ssize)
cat(sprintf("The estimated mean is: %.4f", mean.est))

