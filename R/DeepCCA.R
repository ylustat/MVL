DeepCCA <- function(X,Y,method="DVCCA", LATENT_DIMS = 2, EPOCHS = 100, lr = 0.001, dropout = 0.05, nw=0, python_path=NULL) {
  # if(is.null(python_path)){
  #   python_path <- reticulate::py_discover_config()$python
  # }
  # reticulate::use_python(python_path)
  # reticulate::source_python(paste0(system.file("python", package = "MVL", mustWork = TRUE),"/DeepCCA_base.py"))
  res <- Deep_Models(X,Y,method=method, LATENT_DIMS = LATENT_DIMS, EPOCHS = EPOCHS, lr = lr, dropout = dropout, nw=nw)
  return(res)
}
