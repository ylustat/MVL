MIDAS <- function(X, training_epochs = 10, seed = 1) {
  X_complete <- tryCatch({
    X_conv <- rMIDAS::convert(data.frame(X))
    X_train <- rMIDAS::train(X_conv,
                             training_epochs = training_epochs,
                             seed = 1)
    rMIDAS::complete(X_train, m = 1)[[1]] %>% data.matrix()
  },silent = TRUE, error = function(x) return(NA))
  if(length(X_complete) == 1) {
    X_complete <- X
    X_complete[is.na(X)] <- mean(X,na.rm = T)
  }
  X_complete <- as.matrix(X_complete)
  return(X_complete)
  return(X)
}

