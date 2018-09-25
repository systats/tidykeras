#' k_run
#'
#' @param container list constisting of train and test set
#' @return list of performance data
#'
#' @export

k_run <- function(container){
  
  final <- container$params %>% 
    split(1:nrow(.)) %>% 
    map(~{
      
      message(paste0("Model: ", .x$model))
      
      keras::k_clear_session()
      
      #corpus <- prepare_text(params = .x, data = container$data)
      corpus <- do.call(prepare_text, c(data = list(container$data), as.list(.x)))
      model <- do.call(.x$model, as.list(.x))
      
      ptm <- proc.time()
      
      if(.x$out_dim == 1){
        model %>% 
          keras::fit(
            x = corpus$train, 
            y = matrix(container$data$train$target, ncol = 1),
            batch_size = 30,
            epochs = 1,
            validation_split = .1
          ) 

      } else {
        model %>% 
          keras::fit(
            x = corpus$train, 
            y = k_onehot(container$data$train$target),
            batch_size = 30,
            epochs = 1,
            validation_split = .1
          ) 
        
      }
      
      exec_time <- proc.time() - ptm

      if(.x$out_dim == 1){
        
        pred <- predict(model, x = corpus$test)
        pred <-  ifelse(as.vector(pred) > .5, 1, 0)
        
      } else {
        
        pred <- predict(model, corpus$test) %>% 
          as_tibble %>% 
          split(1:nrow(.)) %>% 
          map_int(~{
            which(.x == max(.x))
          })
        
      }
      
      u <- union(pred, container$data$test$target) %>% as.numeric %>% sort
      tab <- table(factor(pred, u), factor(container$data$test$target, u))
      mat <- caret::confusionMatrix(tab) 
      
      out <- .x %>% 
        bind_cols(
          tibble(
            caret = list(mat), 
            exec_time = list(as.list(exec_time))
          )  
        )
      
      keras::k_clear_session()
      
      return(out)
    })
  return(final)
}