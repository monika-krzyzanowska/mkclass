# ----------------------------------------------------------------------
# DISCRIMINANT FUNCTION
# ----------------------------------------------------------------------
#'  
#' Predict categorical dependent variable (class) using indepedent variables (features).
#'
#' @param data A data frame or a matrix where rows are observations and 
#' columns are features, true classes and target variables.
#' @param features A data frame (can be a subset of "data") that containts
#' the features used to predict the dependent variable.
#' @param target A target variables with binary values
#' @param trueClass A vector with labels for each row in \code{data} 
#' @return A list with following elements: predictedLabels, 
#' performance and performanceProp.
#' @export
#' @import assertthat 
#' @examples
#' 
#' 
#' 
#' # create artificial dataset
#' 
#' # install.packages('mvtnorm', dependencies = TRUE, repos="http://cran.rstudio.com/")
#' 
#' if (!require("mvtnorm")) install.packages("mvtnorm", dependencies = TRUE, repos="http://cran.rstudio.com/")
#' library(mvtnorm)
#' 
#' # covariance matrix function
#' sigmaXY <- function(rho, sdX, sdY) {
#'   covTerm <- rho * sdX * sdY
#'   VCmatrix <- matrix(c(sdX^2, covTerm, covTerm, sdY^2), 
#'                      2, 2, byrow = TRUE)
#'   return(VCmatrix)
#' }
#' 
#' # creating random sample function
#' genBVN <- function(n = 1, seed = NA, muXY=c(0,1), sigmaXY=diag(2)) {
#'   if(!is.na(seed)) set.seed(seed)
#'   rdraws <- rmvnorm(n, mean = muXY, sigma = sigmaXY)
#'   return(rdraws)
#' }
#' 
#' # correlation of cats
#' sigmaCats <- sigmaXY(rho=-0.1, sdX=1, sdY=20)
#' # correlation of dogs
#' sigmaDogs <- sigmaXY(rho=0.8, sdX=2, sdY=30)
#' # correlation of horses
#' sigmaHorses <- sigmaXY(rho=0.2, sdX=1.5, sdY=23)
#' 
#' # create samples
#' noCats <- 50
#' noDogs <- 50
#' noHorses <- 50
#' muCats <- c(4, 130)
#' muDogs <- c(10, 120)
#' muHorses <- c(15, 180) 
#' cats <- genBVN(noCats, muCats, sigmaCats, seed = 7851)
#' dogs <- genBVN(noDogs, muDogs, sigmaDogs, seed = 7852)
#' horses <- genBVN(noHorses, muHorses, sigmaHorses, seed = 7853)
#' 
#' # store the data created in a data frame
#' animalsDf <- as.data.frame(rbind(cats,dogs, horses))
#' Animal <- c(rep("Cats", noCats), rep("Dogs", noDogs), rep("Horses", noHorses))
#' animalsDf <- cbind(animalsDf, Animal)
#' colnames(animalsDf) <- c("weight", "height", "Animal")
#' 
#' catsAndDogs <- function(noCats, noDogs, noHorses, muCats, muDogs, muHorses, sdCats, 
#' sdDogs, sdHorses, rhoCats, rhoDogs, rhoHorses, seed=1111) {
#'   sigmaCats <- sigmaXY(rho=rhoCats, sdX=sdCats[1], sdY=sdCats[2])
#'   sigmaDogs <- sigmaXY(rho=rhoDogs, sdX=sdDogs[1], sdY=sdDogs[2])
#'   sigmaHorses <- sigmaXY(rho=rhoHorses, sdX=sdHorses[1], sdY=sdHorses[2])
#'   cats <- genBVN(noCats, muCats, sigmaCats, seed = seed)
#'   dogs <- genBVN(noDogs, muDogs, sigmaDogs, seed = seed+1)
#'   horses <- genBVN(noHorses, muHorses, sigmaHorses, seed = seed+2)
#'   animalsDf <- as.data.frame(rbind(cats,dogs, horses))
#'   Animal <- c(rep("Cats", noCats), rep("Dogs", noDogs), rep("Horses", noHorses))
#'   animalsDf <- cbind(animalsDf, Animal)
#'   colnames(animalsDf) <- c("weight", "height", "Animal")
#'   return(animalsDf)
#' }
#' 
#' noCats <- 50
#' noDogs <- 50
#' noHorses <- 50
#' 
#' animalsDf <- catsAndDogs(noCats, noDogs, noHorses, muCats, muDogs, muHorses, 
#'                          c(1,20), c(2,30), c(1.5,23), -0.1, 0.8, 0.2)
#'                          
#' # add binary target variables
#' 
#' animalsDf1 <- cbind(animalsDf, target = c(rep(0, noCats), rep(1, noDogs), rep(0, noHorses)) )
#' animalsDf2 <- cbind(animalsDf, target = c(rep(0, noCats), rep(0, noDogs), rep(1, noHorses)) )
#' animalsDf3 <- cbind(animalsDf, target = c(rep(1, noCats), rep(0, noDogs), rep(0, noHorses)) )
#' 
#' # create a new dataframe
#' 
#' animalsDf <- cbind(animalsDf,
#'                    target1 = animalsDf1$target ,
#'                    target2 = animalsDf2$target ,
#'                    target3 = animalsDf3$target)
#'                    
#'                    
#' # executing the function
#' 
#' discriminant_classifier(data = animalsDf, features = animalsDf[,1:2], target = animalsDf[, 4:6], trueClass = animalsDf[, 3])



discriminant_classifier <- function(data, features, target, trueClass){

# analytical solution
X <- as.matrix(cbind(ind=rep(1, nrow(data)),
                     features))
Y <- as.matrix(target)

weightsOptim <- solve(t(X)%*%X) %*% t(X) %*% Y

# compute predictions
predictions <- X %*% weightsOptim

# classify according to the argmax criterion
predictedLabels <- rep("Class 1", nrow(data))
predictedLabels[(predictions==apply(predictions, 1, max))[,1]] <- "Class 2"
predictedLabels[(predictions==apply(predictions, 1, max))[,2]] <- "Class 3"

# classification algorithm performance
performance <- table(trueClass, predictedLabels)

performanceProp <- prop.table(performance, 1)

# grabbing the coefficients
weights <-  t(matrix(c(((weightsOptim[,1]-weightsOptim[,2])[2:3]),
                       ((weightsOptim[,2]-weightsOptim[,3])[2:3]),
                       ((weightsOptim[,1]-weightsOptim[,3])[2:3])), 2, 3))

bias <- c((weightsOptim[,1]-weightsOptim[,2])[1],
          (weightsOptim[,2]-weightsOptim[,3])[1],
          (weightsOptim[,1]-weightsOptim[,3])[1])

# the boundaries
x <- seq(min(data[,1]), max(data[,1]), length.out = nrow(data))
y <- -(weights[1,1]/weights[1,2])*x + (-bias[1])/weights[1,2]
boundaryDf1 <- data.frame(feature1=x, feature2=y,
                          class=rep("Boundary1", length(x)))

x <- seq(min(data[,1]), max(data[,1]), length.out = nrow(data))
y <- -(weights[2,1]/weights[2,2])*x + (-bias[2])/weights[2,2]
boundaryDf2 <- data.frame(feature1=x, feature2=y,
                          class=rep("Boundary2", length(x)))

x <- seq(min(data[,1]), max(data[,1]), length.out = nrow(data))
y <- -(weights[3,1]/weights[3,2])*x + (-bias[3])/weights[3,2]
boundaryDf3 <- data.frame(feature1=x, feature2=y,
                          class=rep("Boundary3", length(x)))

# return the results
return(list(predictedLabels=predictedLabels,
            performance=performance,
            performanceProp=performanceProp
            ))
}