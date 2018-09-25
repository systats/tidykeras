#' k_mcnn
#'
#' get Keras Multi Channel CNN model
#'
#' @param in_dim Number of total vocabluary/words used
#' @param in_length Length of the input sequences
#' @param embed_dim Number of word vectors
#' @param sp_drop Spatial Dropout after Embedding
#' @param filter_sizes Filter sizes (windows)
#' @param num_filters Number of filters per layer
#' @param out_dim Number of neurons of the output layer
#' @param out_fun Output activation function
#' @param ... Exit arguments
#' @return model
#' 
#' @examples 
#' 
#' Architecture is taken from [Code](https://www.kaggle.com/yekenot/textcnn-2d-convolution)
#' 
#' max_features = 100000
#' maxlen = 200
#' embed_size = 300
#' filter_sizes = [1,2,3,5]
#' num_filters = 32
#' 
#' inp = Input(shape=(maxlen, ))
#' x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
#' x = SpatialDropout1D(0.4)(x)
#' x = Reshape((maxlen, embed_size, 1))(x)
#' 
#' conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], embed_size), kernel_initializer='normal',
#'                 activation='elu')(x)
#' conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], embed_size), kernel_initializer='normal',
#'                 activation='elu')(x)
#' conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], embed_size), kernel_initializer='normal',
#'                 activation='elu')(x)
#' conv_3 = Conv2D(num_filters, kernel_size=(filter_sizes[3], embed_size), kernel_initializer='normal',
#'                 activation='elu')(x)
#' 
#' maxpool_0 = MaxPool2D(pool_size=(maxlen - filter_sizes[0] + 1, 1))(conv_0)
#' maxpool_1 = MaxPool2D(pool_size=(maxlen - filter_sizes[1] + 1, 1))(conv_1)
#' maxpool_2 = MaxPool2D(pool_size=(maxlen - filter_sizes[2] + 1, 1))(conv_2)
#' maxpool_3 = MaxPool2D(pool_size=(maxlen - filter_sizes[3] + 1, 1))(conv_3)
#' 
#' z = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])   
#' z = Flatten()(z)
#' z = Dropout(0.1)(z)
#' 
#' outp = Dense(6, activation="sigmoid")(z)
#' 
#' model = Model(inputs=inp, outputs=outp)
#' model.compile(loss='binary_crossentropy',
#'               optimizer='adam',
#'               metrics=['accuracy'])
#' 
#'               
#' @export

k_mcnn <- function(
  in_dim = 10000, 
  in_length = 100,
  embed_dim = 128,
  sp_drop = .2,
  filter_sizes = c(1, 2, 3, 5),
  num_filters = 32,
  out_dim = 1,
  out_fun = "sigmoid",
  ...
){
  
  inp <- keras::layer_input(shape = list(in_length))
  
  x <- inp %>%
    layer_embedding(
      input_dim = in_dim, 
      output_dim = embed_dim, 
      input_length = in_length
    ) %>% 
    layer_spatial_dropout_1d(sp_drop) %>% 
    layer_reshape(list(in_length, embed_dim, 1))
  
  conv_1 <- x %>% 
    layer_conv_2d(
      num_filters, 
      kernel_size = list(filter_sizes[1], embed_dim), 
      kernel_initializer = 'normal',
      activation='elu'
    )
  
  conv_2 <- x %>% 
    layer_conv_2d(
      num_filters, 
      kernel_size = list(filter_sizes[2], embed_dim), 
      kernel_initializer = 'normal',
      activation='elu'
    )
  
  conv_3 <- x %>% 
    layer_conv_2d(
      num_filters, 
      kernel_size = list(filter_sizes[3], embed_dim), 
      kernel_initializer = 'normal',
      activation='elu'
    )
  
  conv_4 <- x %>% 
    layer_conv_2d(
      num_filters, 
      kernel_size = list(filter_sizes[4], embed_dim), 
      kernel_initializer = 'normal',
      activation='elu'
    )
  
  
  max_pool1 <- conv_1 %>% 
    layer_max_pooling_2d(pool_size=list(in_length - filter_sizes[1] + 1, 1))
  
  max_pool2 <- conv_2 %>% 
    layer_max_pooling_2d(pool_size=list(in_length - filter_sizes[2] + 1, 1))
  
  max_pool3 <- conv_3 %>% 
    layer_max_pooling_2d(pool_size=list(in_length - filter_sizes[3] + 1, 1))
  
  max_pool4 <- conv_4 %>% 
    layer_max_pooling_2d(pool_size=list(in_length - filter_sizes[4] + 1, 1))
  
  z <- layer_concatenate(list(max_pool1, max_pool2, max_pool3, max_pool4), axis = 1) %>% 
    layer_flatten()
  
  outp <- z %>% 
    layer_dense(units = out_dim, activation = out_fun)
  
  model <- keras::keras_model(inp, outp) %>%
    compile(
      loss = "binary_crossentropy",
      optimizer = "adam",
      metrics = "accuracy"
    ) 
  
  return(model)
}
