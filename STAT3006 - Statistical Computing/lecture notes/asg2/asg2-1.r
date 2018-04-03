'
Use the inverse method to genearate 5000 samples from
Poisson(lambda=10) and draw a histogram for the 5000
samples. The probability mass function of Poisson(lambda) is
P(X = m) = exp(-lambda)*lambda^m/frac(m), with m =0, 1, 2, 3...
# http://epoli.pbworks.com/f/Math276-Week3annot.pdf
'

simPoisson <- function(n=1, lambda=1){
  u <- runif(n, min=0, max=1) #simulate standard uniforms
  v <- rep(0, n)
  for(j in 1:n){
    i <- 0
    p <- exp(-lambda)
    f <- p
    while(u[j] >= f){
      p <- lambda*p/(i+1)
      f <- f + p
      i <- i + 1
    }
    v[j] <- i
  }
  return(v)
}

n = 5000
lambda = 10
output <- simPoisson(n=n, lambda=lambda)
mean(output)
xtabs(~output)/5000
barplot(xtabs(~output))
# barplot(xtabs(~output)/5000)

