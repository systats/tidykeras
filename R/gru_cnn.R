#' k_gru_cnn
#'
#' get Keras stacked CNN GRU model
#'
#' @param in_dim Number of total vocabluary/words used
#' @param in_length Length of the input sequences
#' @param embed_dim Number of word vectors
#' @param sp_drop Spatial Dropout after Embedding
#' @param gru_dim Number of GRU neurons
#' @param out_dim Number of neurons of the output layer
#' @param out_fun Output activation function
#' @param ... Exit arguments
#' @return model
#'
#' @examples 
#' Taken from * [Code](https://www.kaggle.com/mosnoiion/two-rnn-cnn-columns-networks-with-keras)
#' [Paper](http://people.idsia.ch/~ciresan/data/cvpr2012.pdf)
#' 
#' @export

k_gru_cnn <- function(
  in_dim = 10000, 
  in_length = 100,
  embed_dim = 128,
  sp_drop = .2,
  gru_dim = 64,
  gru_drop = .2,
  out_dim = 1,
  out_fun = "sigmoid",
  ...
){
  
  inp <- layer_input(shape = list(in_length), dtype = "int32", name = "input")
  emm <- inp %>%
    layer_embedding(
      input_dim = in_dim, 
      output_dim = embed_dim, 
      input_length = in_length
    ) %>%
    layer_spatial_dropout_1d(rate = sp_drop)
  
  model_1 <- emm %>%
    bidirectional(
      layer_gru(units = gru_dim, return_sequences = T, recurrent_dropout = gru_drop) 
    ) %>% 
    layer_conv_1d(
      60, 
      3, 
      padding = "valid",
      activation = "relu",
      strides = 1
    ) 
  
  model_2 <- emm %>%
    bidirectional(
      layer_gru(units = gru_dim, return_sequences = T, recurrent_dropout = gru_drop) 
    ) %>% 
    layer_conv_1d(
      120, 
      2, 
      padding = "valid",
      activation = "relu",
      strides = 1
    ) 
  
  max_pool1 <- model_1 %>% layer_global_max_pooling_1d()
  ave_pool1 <- model_1 %>% layer_global_average_pooling_1d()
  max_pool2 <- model_2 %>% layer_global_max_pooling_1d()
  ave_pool2 <- model_2 %>% layer_global_average_pooling_1d()
  
  outp <- layer_concatenate(list(ave_pool1, max_pool1, ave_pool2, max_pool2)) %>%
    layer_dense(units = out_dim, activation = out_fun)
  
  model <- keras_model(inp, outp) %>% 
    compile(
      optimizer = "adam",
      loss = "binary_crossentropy",
      metrics = c("acc")
    )
  
  return(model)
}