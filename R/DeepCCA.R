DeepCCA <- function(X,Y,method="DVCCA", LATENT_DIMS = 2, EPOCHS = 100, lr = 0.001, dropout = 0.05, nw=2, python_path=NULL) {
  # if(is.null(python_path)){
  #   python_path <- reticulate::py_discover_config()$python
  # }
  # reticulate::use_python(python_path)
  reticulate::source_python(paste0(system.file("python", package = "MVL", mustWork = TRUE),"/DeepCCA_base.py"))
  return(Deep_Models(X,Y,method="DVCCA", LATENT_DIMS = LATENT_DIMS, EPOCHS = EPOCHS, lr = lr, dropout = dropout, nw=nw))
}
