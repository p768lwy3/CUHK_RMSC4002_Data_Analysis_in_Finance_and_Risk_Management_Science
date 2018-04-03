"
Estimate the integration integral(3, +inf)(sin(x)*exp(-x^(2)/2)) dx
a. using 5000 samples from Q2.
b. using importance sampling (drawing 5000 samples from g(x)
= 3exp(-3*(x-3))*I(x>=3) to estimate the integration.)
"

# (a) using 5000 samples from Q2;
X <- read.csv("part2.csv")$x # if there are any problem of reading files, plz change the dir.
integration <- NULL
for(i in seq(1, length(X))){
  mc.integral <- sqrt(pi*2)*(1-pnorm(3))*mean(sin(X[1:i]))
  integration <- c(integration, mc.integral)
}
cat(sprintf('Mean: %.6f, Var: %.6f.\n', 
            mean(integration), var(integration)))
plot(integration, type='l', main="Monte Carlo Integration", 
     xlab="sample number", ylab="estimate")
abline(h=mean(integration))

# (b) using importance sampling (drawing 5000 samples from
#     g(x) = 3exp(-3(x-3))I(x>=3) to estimate the integration).
g <- function(n) (rexp(n, 3) + 3)
X <- g(5000)
integration <- NULL
for(i in seq(1, length(X))){
  x <- X[1:i]
  mc.integral <- exp(-3^2)*sum(sin(
    x)*exp(-x^2/2+3*x))/(3*i)
  integration <- c(integration, mc.integral)
}
cat(sprintf('Mean: %.6f, Var: %.6f.\n', 
            mean(integration), var(integration)))
plot(integration, type='l', main="Monte Carlo Integration", 
     xlab="sample number", ylab="estimate")
abline(h=mean(integration))

