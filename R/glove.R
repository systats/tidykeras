#' k_glove
#'
#' get Keras GloVe model
#'
#' @param in_dim Number of total vocabluary/words used
#' @param in_length Length of the input sequences
#' @param embed_dim Number of word vectors
#' @param out_dim Number of neurons of the output layer
#' @param out_fun Output activation function
#' @param ... Exit arguments
#' @return model
#'
#' @export

k_glove <- function(
  in_dim = 10000, 
  in_length = 100,
  embed_dim = 128,
  #layer_drop = .3, 
  out_dim = 1,
  out_fun = "sigmoid",
  ...
){
  
  model <- keras_model_sequential() %>%
    layer_embedding(
      input_dim = in_dim, 
      output_dim = embed_dim, 
      input_length = in_length
    ) %>%
    layer_global_average_pooling_1d() %>%
    layer_dense(units = out_dim, activation = out_fun) %>% 
    compile(
      loss = "binary_crossentropy",
      optimizer = "adam",
      metrics = "accuracy"
    )
  
  return(model)
}