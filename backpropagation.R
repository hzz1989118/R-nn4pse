library(nnet)
PI <- 3.1415926
x <- matrix(seq(0,2*PI,0.1),nrow=1) #simulate one period of sine function
t <- sin(x)
x.2 <- t(x) #for nnet package
t.1 <- (t-min(t))/diff(range(t)) #targets. They are the range of linearly transformed sine function. The range is now [0,1]
t.2 <- t(t.1) #for nnet package

# nn funciton takes three parameters: x - stimulus, t - targets, hidden.size - number of nerons in the hidden layer. The function returns weights that minimizes training MSE. Logistic transfer functions are applied twice.
nn <- function (x,t,hidden.size){
	init <- seq(0.2,0.9,0.1) #domain of initial weights
	n.nodes <- hidden.size
    ##########initialize weights and intercpets###############
	z <- t(sample(init,n.nodes,replace=TRUE))
	w <- t(t(sample(init,n.nodes,replace=TRUE)))
	b.1 <- t(sample(init,n.nodes,replace=TRUE))
	b.2 <- t(sample(init,1,replace=TRUE))
	temp <- cbind(z,t(w),b.1,b.2)
	temp <- as.vector(temp)
    ######################end#################################
    #f.loss calculates loss (or training error).
	f.loss <- function (theta,x,t){
		n.nodes <- (length(theta)-1)/3
		z <- matrix(theta[1:n.nodes],nrow=1)
		w <- matrix(theta[(n.nodes+1):(2*n.nodes)],ncol=1)
		b.1 <- theta[(2*n.nodes+1):(3*n.nodes)]
		b.2 <- theta[3*n.nodes+1]
		h.k <- plogis(t(z)%*%x+b.1) #5.9
	    t.hat <- plogis(t(w)%*%h.k+b.2)
	    e.k <- t - t.hat
	    loss <- 0.5*sum(e.k^2)
	    loss
	}
	fit <- optim(temp,f.loss,x=x,t=t,method="BFGS",control=c(maxit=100000))#minizie loss over weight space. BFGS is used.
	convergence <- fit$convergence
	weights <- fit$par
	mse <- fit$value
	z <- matrix(weights[1:n.nodes],nrow=1)
	w <- matrix(weights[(n.nodes+1):(2*n.nodes)],ncol=1)
	b.1 <- matrix(weights[(2*n.nodes+1):(3*n.nodes)],nrow=1)
	b.2 <- matrix(weights[3*n.nodes+1],nrow=1)
	out <- list(z,w,b.1,b.2,convergence,mse)
} # z,w are weights; b.1 and b.2 are intercepts. convergence is a binary var indicating convergence of optimization whtin 1e5 iterations. mse is final training MSE.

# predict.nn takes object nn and input vector.
predict.nn <- function(my.net, x){
    z <- t(unlist(my.net[1]))
    w <- t(t(unlist(my.net[2])))
    b.1 <- (unlist(my.net[3]))
    b.2 <- (unlist(my.net[4]))
    n.nodes=8
    h.k <- plogis(t(z)%*%x+b.1) #5.9
    t.hat <- plogis(t(w)%*%h.k+b.2)
    t.hat <- as.vector(t.hat)
    t.hat
}

my.net.1 <- nnet(x=x.2,y=t.2,size=8,maxit=10000) #fit nnet
my.net <- nn(x=x,t=t.1,hidden.size=8) #fit nn
x <- as.vector(x)
plot(predict.nn(my.net,x)~x,col="red",xlab="x",ylab="y")
points(t.2~x,col="blue")
points(predict(my.net.1)~x.2,col="green")
legend(4,1, c("true targerts","nn","nnet"), lty="99",pch=1,col=c("blue","red","green"))
