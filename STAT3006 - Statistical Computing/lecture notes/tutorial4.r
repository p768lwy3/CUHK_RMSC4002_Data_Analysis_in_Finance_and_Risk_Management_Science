"
try to code the tutorial 4...
"

set.seed(12345678)

accept.reject <- function(f, g, n){
  n.accepts <- 0
  result.sample <- rep(NULL, n)
  while(n.accepts < n){
    u <- runif(1, 0, 1)
    y <- g(1)
    if(u <= f(y)){
      n.accepts <- n.accepts + 1
      result.sample[n.accepts] = y
    }
  }
  return(result.sample)
}

f <- function(x) (sqrt(2*pi)*exp(-x^2/2+2.5*(x-2.5)+2.5^2/2))
g <- function(n) (rexp(n, 2.5) + 2.5)
output <- accept.reject(f, g, 5000)
hist(output, breaks=20, freq=FALSE)

integration <- NULL
for(i in seq(1, length(output))){
  mc.integral <- sqrt(pi*2)*(1-pnorm(2.5))*mean(sin(output[1:i]))
  integration <- c(integration, mc.integral)
}
plot(integration, type='l', main="Monte Carlo Integration", 
     xlab="sample number", ylab="estimate", ylim=c(0,0.010))
abline(h=0.005)

# importance sampling,
g <- function(n) (rexp(n, 2.5) + 2.5)
samples <- g(5000)
integration <- NULL
for(i in seq(1, length(samples))){
  s <- samples[1:i]
  mc.integral <- exp(-2.5^2)*sum(sin(
    s)*exp(-s^2/2+2.5*s))/(2.5*i)
  #mc.integral <- sum(m(samples[1:i]))/i
  integration <- c(integration, mc.integral)
}
plot(integration, type='l', main="Monte Carlo Integration", 
     xlab="sample number", ylab="estimate")
abline(h=0.005)
