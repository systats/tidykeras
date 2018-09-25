#' k_onehot
#'
#' @param x A string vector
#' @return transforms a k-level factor vector into a n*k matrix (one-hote encoding)\code{x}
#'
#' @export
k_onehot <- function(x){
  return(dummies::dummy(x) %>% as.matrix())
}

#' params_helper
#'
#' @param input list
#' @param expend_params Wheter to grid expand the model params
#' @return input modified
#'
#' @export

params_helper <- function(input, expend_params = F){
  if(expend_params){
    input$params <- cross_df(input$params)
  } else {
    input$params <- as_tibble(input$params)
  }
  return(input)
}


# Nice 
`%||%` <- function(lhs, rhs) {
  if (!is.null(lhs)) {
    lhs
  } else {
    rhs
  }
}


#' vis_table
#'
#'
#' @param data data
#' @param text text variable
#' @return out ...
#'
#' @export

vis_table <- function(pred, real){
  tibble(preds = pred, real = real) %>% 
    dplyr::count(preds, real) %>% 
    dplyr::group_by(real) %>% 
    dplyr::mutate(n_real = sum(n)) %>% 
    ungroup() %>% 
    dplyr::mutate(perc_real = round(n/n_real * 100, 1)) %>%
    dplyr::mutate(label = paste0(n, "\n", perc_real, "%")) %>% 
    mutate(preds = factor(preds, levels = sort(unique(preds), decreasing = F))) %>% 
    mutate(real = factor(real, levels = sort(unique(real), decreasing = T))) %>% 
    #mutate(real = factor(real)) %>% 
    ggplot(aes(real, preds, fill = n)) + 
    ggplot2::geom_tile(alpha = 0.8) + 
    #viridis::scale_fill_viridis(direction = -1) + 
    scale_fill_gradient(low = "white", high = "black")+
    scale_x_discrete(position = "top") + 
    ggthemes::theme_few() + 
    theme(legend.position = "none") + 
    coord_equal() + 
    labs(x = "Real value y", y = "Predicted value y hat") +
    ggplot2::geom_text(aes(label = label))
}


#' corpus_description
#'
#' corpus_description
#'
#' @param data data
#' @param text text variable
#' @return out ...
#'
#' @export
corpus_description <- function(data, text){
  dat <- data %>%
    dplyr::rename_("in_text" = text) %>%
    dplyr::mutate(nchar = in_text %>% nchar())  %>%
    dplyr::mutate(ntok = tidyTX::tx_n_tokens(in_text))
  
  tc <- dat %>%
    dplyr::select(in_text) %>%
    tidytext::unnest_tokens(word, in_text, token = "words") %>% 
    dplyr::count(word) %>% 
    dplyr::arrange(desc(n)) 
  
  out <- list(
    char = list(
      mean = mean(dat$nchar, na.rm = T) %>% floor(),
      med = median(dat$nchar, na.rm = T) 
    ),
    token = list(
      mean = mean(dat$ntok, na.rm = T) %>% floor(),
      med = median(dat$ntok, na.rm = T),
      quant = quantile(dat$ntok),
      denc = quantile(dat$ntok, probs = seq(.1:1, by = .1)),
      n_5 = tc %>%
        filter(n > 5) %>%
        nrow(),
      n_3 = tc %>%
        filter(n > 3) %>% 
        nrow(),
      n_all = tc %>%
        nrow(),
      tokens = tc
    ),
    plot = dat %>%
      select(ntok, nchar, target) %>%
      gather(var, value, -target) %>%
      ggplot(aes(value)) +
      geom_histogram() +
      facet_wrap(~var, scales = "free"),
    dat = dat
  )
  return(out)
}