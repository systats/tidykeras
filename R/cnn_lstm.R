#' k_cnn_lstm
#'
#' get Keras stacked CNN LSTM model
#'
#' @param in_dim Number of total vocabluary/words used
#' @param in_length Length of the input sequences
#' @param embed_dim Number of word vectors
#' @param sp_drop Spatial Dropout after Embedding
#' @param filter_sizes Filter sizes (windows)
#' @param num_filters Number of filters per layer
#' @param kernel_size Size of feature map
#' @param pool_size Size of feature map
#' @param lstm_dim Number of neurons in the LSTM layer
#' @param lstm_drop Dropout ratio in the LSTM layer
#' @param layer_drop Dropout ratio between layers
#' @param out_dim Number of neurons of the output layer
#' @param out_fun Output activation function
#' @param ... Exit arguments
#' @return model
#' 
#' @examples 
#' 
#' @export

k_cnn_lstm <- function(
  in_dim = 10000, 
  in_length = 100,
  embed_dim = 128,
  sp_drop = .2,
  filter_sizes = c(1, 2, 3, 5),
  num_filters = 100,
  kernel_size = 5, 
  pool_size = 4,
  lstm_dim = 64,
  lstm_drop = .3, 
  layer_drop = .3, 
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
    #layer_dropout(0.25) %>%
    layer_conv_1d(
      num_filters, 
      kernel_size, 
      padding = "valid",
      activation = "relu",
      strides = 1
    ) %>%
    layer_max_pooling_1d(pool_size) %>%
    layer_lstm(units = lstm_dim, dropout = layer_drop, recurrent_dropout = lstm_drop) %>%
    layer_dense(units = out_dim, activation = out_fun) %>%
    compile(
      loss = "binary_crossentropy",
      optimizer = "adam",
      metrics = "accuracy"
    )
  
  return(model)
}
