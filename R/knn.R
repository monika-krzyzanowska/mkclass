# ----------------------------------------------------------------------
# K-NEAREST NEIGHBORS CLASSIFIER
# ----------------------------------------------------------------------
#'  
#' Classify a new object using k nearest neighbors.
#'
#' @param data A data frame or a matrix where rows are observations and 
#' columns are features. If \code{type} is "train" this is training 
#' dataset, and if it is "predict" it is test dataset.
#' @param trueClasses A vector with labels for each row in \code{data} 
#' if \code{type} is "train", and with labels for each row in 
#' \code{memory} if \code{type} is "predict".
#' @param memory A data frame or a matrix where rows are observations 
#' and columns are features. If \code{type} is "train" this argument 
#' is not needed, and if it is "predict" it is a training dataset.
#' @param k Number of neighbors that the classifier should use.
#' @param p Distance metric the classifier should use, the value can be 
#' either 1 (Manhattan), 2 (Euclidean) or Inf (Chebyshev). 
#' @param type Whether the goal is to train the classifier or predict 
#' classes of new observations based on past ones. The value can be 
#' either "train" or "predict".
#' @param columns Columns that containts the features (usefull in case
#' when one of the columns contains the true classes)
#' @return A list with following elements: predictedClasses, 
#' accuracy and errorCount.
#' @export
#' @import assertthat 
#' @examples
#' # create artificial dataset
#' 
#' data  <- matrix(rnorm(180), ncol=3)
#' classes <- c(rep(0, 20), rep(1, 20), rep(2, 20))
#' data <- cbind(data, classes)
#' 
#' # executing the function
#' trueClasses = data[,4]
#' columns = 1:3
#' 
#' kNN_classifier(data, trueClasses, memory=NULL,
#'                k=1, p=2, type="train", columns)




kNN_classifier <- function(data, trueClasses, memory=NULL,
                           k=1, p=2, type="train", columns) {
  # test the inputs
  library(assertthat)
  not_empty(data); not_empty(trueClasses);
  if (type=="train") {
    assert_that(nrow(data)==length(trueClasses))
  }
  is.string(type); assert_that(type %in% c("train", "predict"))
  is.count(k);
  assert_that(p %in% c(1, 2, Inf))
  if (type=="predict") {
    assert_that(not_empty(memory) &
                  ncol(memory)==ncol(data) &
                  nrow(memory)==length(trueClasses))
  }
  
  # Compute the distance between each point and all others
  noObs <- nrow(data)
  
  # if we are making predictions on the test set based on the memory,
  # we compute distances between each test observation and observations
  # in our memory
  if (type=="train") {
    predictionId <- 1
    distMatrix <- matrix(NA, noObs, noObs)
    for (obs in 1:noObs) {
      # getting the probe for the current observation
      probe <- as.numeric(data[obs,columns])
      probeExpanded <- matrix(rep(probe, each=noObs), nrow=noObs)
      # computing distances between the probe and exemplars in the
      # training data
      if (p %in% c(1,2)) {
        distMatrix[obs, ] <- (rowSums((abs(data[, columns] -
                                             probeExpanded))^p) )^(1/p)
      } else if (p==Inf) {
        
        distMatrix[obs, ] <- apply(abs(data[, columns] - probeExpanded), 1, max)
      }
    }
  } else if (type == "predict") {
    predictionId <- 0
    noMemory <- nrow(memory)
    
    distMatrix <- matrix(NA, noObs, noMemory)
    for (obs in 1:noObs) {
      # getting the probe for the current observation
      probe <- as.numeric(data[obs,columns])
      probeExpanded <- matrix(rep(probe, each=noMemory), nrow=noMemory)
      # computing distances between the probe and exemplars in the memory
      if (p %in% c(1,2)) {
        distMatrix[obs, ] <- (rowSums((abs(memory[, columns] -
                                             probeExpanded))^p) )^(1/p)
      } else if (p==Inf) {
        distMatrix[obs, ] <- apply(abs(memory[, columns] - probeExpanded), 1, max)
      }
    }
  }
  
  # Sort the distances in increasing numerical order and pick the first
  # k elements
  neighbors <- apply(distMatrix, 2, order)
  # Compute and return the most frequent class in the k nearest neighbors
  predictedClasses <-  rep(NA, noObs)
  for (obs in 1:noObs) {
    
    predictedClasses[obs] <- names(sort(table(trueClasses[neighbors[2:(k+1), obs]]),
                                        decreasing=T)[1])
    
  }
  
  
  # predictedClasses <- rep(NA, noObs)
  # for (obs in 1:noObs) {
  # prob[obs] <- mean(trueClasses[neighbors[(1+predictionId):
  #                                         (k+predictionId), obs]])
  # if(prob[obs] > 0.5) {
  #  predictedClasses[obs] <- 1
  #  } else {
  #   predictedClasses[obs] <- 0
  #  }
  # }
  
  # examine the performance, available only if training
  if (type=="train") {
    errorCount <- table(predictedClasses, trueClasses)
    accuracy <- mean(predictedClasses==trueClasses)
  } else if (type == "predict") {
    errorCount <- NA
    accuracy <- NA
  }
  
  # return the results
  return(list(predictedClasses=predictedClasses,
              #prob=prob,
              accuracy=accuracy,
              errorCount=errorCount))
}
