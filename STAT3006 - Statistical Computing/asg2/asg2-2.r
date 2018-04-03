'
Use the accept-reject method to genearate 5000 samples from f(x) = 
(1/sqrt(2*pi)*exp(-x^(2)/2)*I(x>=3))/(1-faiz(3)), in which g(x) = 
3*exp(-3*(x-3))*I(x>=3) and M = e^(-4.5)/(3*sqrt(2*pi)*(1-faiz(3))),
and prove that f(x) <= M*g(x)
# http://www.di.fc.ul.pt/~jpn/r/ECS/index.html
# 
# https://wiki.math.uwaterloo.ca/statwiki/index.php?title=stat341f11
# https://wiki.math.uwaterloo.ca/statwiki/index.php?title=stat340s13
# http://psiexp.ss.uci.edu/research/teachingP205C/205C.pdf
# https://www.youtube.com/watch?v=KoNGH5PkXDQ
# https://stats.stackexchange.com/questions/19736/what-does-truncated-distribution-mean

'
set.seed(54321)

accept.reject <- function(f, g, m, n){
  n.accepts <- 0
  n.rejects <- 0
  result.sample <- rep(NULL, n)
  while(n.accepts < n){
    u <- runif(1, min=0, max=1)
    y <- g(1)
    if(u <= f(y)){
      n.accepts <- n.accepts + 1
      result.sample[n.accepts] = y
    }else{
      n.rejects <- n.rejects + 1
    }
  }
  cat(sprintf("Accept Ratio by m: %f. \n", 1/m))
  cat(sprintf("Accept Ratio by real: %f. \n", 
              n.accepts/(n.accepts+n.rejects)))
  return(result.sample)
}
'indicator <- function(x, threshold=3){
  if(x >= threshold){
    return(x)
  }else{
    return(0)
  }
}'

f <- function(x) (sqrt(2*pi)*exp(-x^2/2+3*(x-3)+3^2/2))
g <- function(n) (rexp(n, 3) + 3)
m <- exp(-3^2/2) / (3 * sqrt(2 * pi) * (1 - pnorm(3)))
output <- accept.reject(f, g, m, 5000)

hist(output, breaks=20, freq=FALSE)

# save as txt for part 3,
f <- "part2.csv"
write.csv(output, file=f)

