DeepCCA <- function(X,Y,method="DVCCA",python_path=NULL) {
  if(is.null(python_path)){
    python_path <- reticulate::py_discover_config()$python
  }
  reticulate::use_python(python_path)
  reticulate::source_python(paste0(system.file("python", package = "MVL", mustWork = TRUE),"/DeepCCA_base.py"))
  return(Deep_Models(X,Y))
}
