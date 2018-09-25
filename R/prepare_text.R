#' prepare_text
#'
#'
#' @param data list constisting of train and test set
#' @param in_dim Max top words (max_features)
#' @param in_length Max sequence length
#' @param char_level Word or character level?
#' @param ... Exit arguments
#' @return model
#'
#' @export

prepare_text <- function(data, in_dim = 10000, in_length = 100, char_level = F, return_tokenizer = F, ...){
  
  tokenizer <- text_tokenizer(
    num_words = in_dim, 
    char_level = char_level
  )
  
  tokenizer %>% fit_text_tokenizer(x = data$train$text)
  
  train <- tokenizer %>% 
    texts_to_sequences(texts = data$train$text) %>% 
    pad_sequences(maxlen = in_length, value = 0)
  
  test <- tokenizer %>% 
    texts_to_sequences(texts = data$test$text) %>% 
    pad_sequences(maxlen = in_length, value = 0)
  
  if(return_tokenizer){
    out <- list(train = train, test = test, tokenizer = tokenizer)
  } else {
    out <- list(train = train, test = test)
  }
  
  return(out)
}
