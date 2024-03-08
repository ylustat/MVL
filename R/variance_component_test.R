#' @export
variance_component_test <- function(res){
  Q <- res$QRes %>% lapply(function(x) Reduce("+",x)/length(x))
  W <- res$WRes %>% lapply(function(x) Reduce("+",x)/length(x))

  p <- tryCatch({mapply(function(q,w) {
    SKAT::Get_Davies_PVal(q, w)$p.value
  }, Q, W, SIMPLIFY = F) %>%
      unlist()},silent = TRUE, error = function(x) return(NA))
  names(p) <- names(res$gamma)
  return(p)
}
