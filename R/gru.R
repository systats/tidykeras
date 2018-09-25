#' k_gru
#'
#' get Keras GRU model
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
#' Taken from https://www.kaggle.com/yekenot/pooled-gru-fasttext
#' 
#' def get_model():
#'   inp = Input(shape=(maxlen, ))
#' x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
#' x = SpatialDropout1D(0.2)(x)
#' x = Bidirectional(GRU(80, return_sequences=True))(x)
#' avg_pool = GlobalAveragePooling1D()(x)
#' max_pool = GlobalMaxPooling1D()(x)
#' conc = concatenate([avg_pool, max_pool])
#' outp = Dense(6, activation="sigmoid")(conc)
#' 
#' model = Model(inputs=inp, outputs=outp)
#' model.compile(loss='binary_crossentropy',
#'               optimizer='adam',
#'               metrics=['accuracy'])
#'               
#' @export

k_gru <- function(
  in_dim = 10000, 
  in_length = 100,
  embed_dim = 128,
  sp_drop = .2,
  gru_dim = 64,
  out_dim = 1,
  out_fun = "sigmoid",
  ...
){
  
  inp <- keras::layer_input(shape = list(in_length))
  
  main <- inp %>%
    layer_embedding(
      input_dim = in_dim, 
      output_dim = embed_dim, 
      input_length = in_length
    ) %>% 
    layer_spatial_dropout_1d(sp_drop) %>% 
    keras::bidirectional(keras::layer_gru(units = gru_dim, return_sequences = T))
  
  avg_pool <- main %>% layer_global_average_pooling_1d()
  max_pool <- main %>% layer_global_average_pooling_1d()
  
  outp <- layer_concatenate(c(avg_pool, max_pool)) %>% 
    layer_dense(units = out_dim, activation = out_fun)
  
  model <- keras::keras_model(inp, outp) %>%
    compile(
      loss = "binary_crossentropy",
      optimizer = "adam",
      metrics = "accuracy"
    )  
  
  return(model)
}