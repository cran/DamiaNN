
#' @description main function that runs the interactive script
#' @details
#' takes your numerai training data and trains a neural network to your architectural specifications.
#' provides you with the out of sample error
#' offers to retrain with a new architecture or predict on a numerai tournament dataset.
#' Can then write the predictions to a CSV
#' @name Start
#' @export
#' @importFrom caret createDataPartition
#' @title start script
Start <- function(){

  continue_with_same_data <- TRUE
  continue_with_same_shape <- TRUE
  predict_again <- TRUE

  #library(caret)

  while(continue_with_same_data){

    absolute_path <- readline("Give me the absolute path to your numerai training data, forward slashes only :  ")
    absolute_path <- as.character(absolute_path)

    training_data <- read.csv(file = absolute_path, header = TRUE)

    while(continue_with_same_shape){

      shape <- readline("Do you want to specify the shape of your neural network? Type yes or no :  ")
      if(shape == "no"){
        first_layer <- max(floor(ncol(training_data) / 2),10)
        second_layer <- max(floor(ncol(training_data) / 4), 5)
        third_layer <- max(floor(ncol(training_data) / 8), 5)
        layer_sizes <- c(first_layer, second_layer, third_layer)
      }
      if(shape == "yes"){
        print(paste("Note : this Neural Network is optimized using Vanilla Gradient Descent. We recommend using a shallow architecture to avoid vanishing gradients.","\n"))
        hidden_layers <- readline("How many hidden layers do you want? Type a positive integer : ")
        hidden_layers <- as.integer(hidden_layers)
        layer_sizes <- NULL
        for(layer in 1:hidden_layers){
          layer_size <- readline(paste("how many nodes do you want layer",layer,"to have? Type a positive integer :  "))
          layer_size <- as.integer(layer_size)
          layer_sizes <- c(layer_sizes,layer_size)
        }
      }

      target <- 22
      average_train_loss <- 0
      average_oos_loss <- 0

      for(run in 1:5){
        inTrain <- createDataPartition(training_data$target, p = 4/5)[[1]]
        train_data <- training_data[inTrain,]
        train_target <- train_data[,target]
        validate_data <- training_data[-inTrain,]
        validate_target <- validate_data[,target]
        NN <- new("Neural_Network",(ncol(train_data)-1),layer_sizes)
        NN <- Train(NN, train_data,.00001,.05,.000000001)
        train_preds <- Predict(NN,train_data)
        train_loss <- Get_LogLoss(train_preds,train_target)
        validate_preds <- Predict(NN,validate_data)
        validate_loss <- Get_LogLoss(validate_preds, validate_target)
        average_oos_loss <- average_oos_loss + validate_loss
        average_train_loss <- average_train_loss + train_loss
      }

      average_train_loss <- average_train_loss / 5
      average_oos_loss <- average_oos_loss / 5

      print(paste("We did 5-fold cross validation . Here's your logarithmic loss \n Train : ",average_train_loss,"\n","Test : ",average_oos_loss))

      NN <- new("Neural_Network", (ncol(training_data)-1),layer_sizes)
      NN <- Train(NN, training_data, .00001, .05, .000000001)

      predict_or_not <- readline("would you like to use this model to predict? Type yes or no :  ")
      if(predict_or_not == "yes"){
        prediction_data_path <- readline("please type the full path to your numerai tournament data :  ")
        prediction_data <- read.csv(prediction_data_path, header = TRUE)
        while(predict_again){

          id <- prediction_data[,1]
          prediction_data <- prediction_data[,-1]
          prediction_data <- cbind(prediction_data,id)
          preds <- Predict(NN,prediction_data)

          write_path <- readline("do you want to write your predictions to a csv? Type no or the full file path :  ")
          if(write_path == "no"){}
          else {write.csv(x = preds, file = write_path)}


          again <- readline("do you want to predict on a different data set? Type yes or no :  ")
          if(again == "no"){break}
          prediction_data_path <- readline("please type the full path to your prediction data :  ")
          header_or_no <- readline("does it have a header? :  ")
          if(header_or_no == "yes"){
            prediction_data <- read.csv(prediction_data_path, header = TRUE)}
          else{prediction_data <- read.csv(prediction_data,header=FALSE)}
        }
      }
      exit <- readline("do you want to exit? Type yes or no :  ")
      if(exit == "yes"){
        stop()
      }
      new_shape <- readline("would you like to retrain on the same data, but with a differently shaped network? Type yes or no :  ")
      if(new_shape == "no"){break}

    }
    new_data <- readline("would you like to train a network on different training data? Type yes or no :   ")
    if(new_data == "no"){break}
  }

}

###############################################################################################################
#' Neural Network implementation

setClass("Neural_Network",
         slots = list(
           weight_matrices = "list", # of matrices ... connection strengths
           pre_sigmoidal_outputs = "list", # of numerics/matrices ... neurons
           post_sigmoidal_outputs = "list", # of numerics/matrices ... neurons
           sigmoidal_derivatives = "list", # of numerics/matrices ... elasticity
           delta_terms = "list", # of numerics/matrices ... feedback strength
           weight_updates = "list" # of matrices ... connection updates
           ),
         sealed = FALSE)

#' @description initalizes a neural network capable of studying datasets with ncol = to the ncol(sample_dataset) and making predictions on such datasets
#' @param .Object ... a Neural_Network object
#' @param number_predictors ... a numeric telling how many preditors there are
#' @param hidden_layer_lengths ... a numeric telling the number of layers and the number of neurons in each layer
#' @details NN is parametrized by its connection_strength matrices
#' @return Neural_Network
#' @title init
setMethod(
  f = "initialize",

  signature = "Neural_Network",

  definition = function(.Object, number_predictors, hidden_layer_lengths){
    input_length <- number_predictors
    number_of_weight_matrices <- length(hidden_layer_lengths) + 1
    all_layer_sizes <- c(input_length,hidden_layer_lengths,1)
    for(weight_matrix_index in 1:number_of_weight_matrices){
      input_layer_size <- all_layer_sizes[weight_matrix_index] + 1
      output_layer_size <- all_layer_sizes[weight_matrix_index + 1]
      weight_matrix_size <- input_layer_size * output_layer_size
      .Object@weight_updates[[weight_matrix_index]] <- matrix(rep(0,weight_matrix_size),output_layer_size,input_layer_size)
      initial_weight_max <- min((1 / (weight_matrix_size)^(1/5)),.25)
      weight_matrix <- 2*runif(weight_matrix_size)*initial_weight_max - initial_weight_max
      dim(weight_matrix) <- c(output_layer_size,input_layer_size)
      .Object@weight_matrices[[weight_matrix_index]] <- weight_matrix}
    return(.Object)})

setGeneric(name = "forward_propogation",
           valueClass = "Neural_Network",
           def = function(object, dataset){standardGeneric("forward_propogation")})

#' @description ... part of the training program
#' @param object is a Neural_Network
#' @param dataset is a matrix not containing the target vector
#' @return Neural_Network
#' @title f_prop

setMethod(f = "forward_propogation",
          signature = c("Neural_Network","matrix"),
          valueClass = "Neural_Network",
          definition = function(object,dataset){
            number_of_weight_matrices <- length(object@weight_matrices)
            input <- dataset
            object@pre_sigmoidal_outputs[[1]] <- NULL
            object@post_sigmoidal_outputs[[1]] <- input
            object@sigmoidal_derivatives[[1]] <- NULL
            for(weight_matrix_index in 1:number_of_weight_matrices){
              weight_matrix <- object@weight_matrices[[weight_matrix_index]]
              input <- object@post_sigmoidal_outputs[[weight_matrix_index]]
              input <- cbind(rep(1,nrow(input)),input)
              z <- input %*% t(weight_matrix)
              a <- 1 / (1 + exp(-z))
              g <- a * (1-a)
              object@pre_sigmoidal_outputs[[weight_matrix_index + 1]] <- z
              object@post_sigmoidal_outputs[[weight_matrix_index + 1]] <- a
              object@sigmoidal_derivatives[[weight_matrix_index + 1]] <- g}
            return(object)})


setGeneric(name = "back_propogation",
           valueClass = "Neural_Network",
           def = function(object,target,regularization_parameter,learning_rate){
             standardGeneric("back_propogation")})

#' @description updates connection strengths using results of last forward prop
#' @param object is a Neural_Network
#' @param target is a numeric vector
#' @param regularization_parameter is non-negative number punishes strong connections
#' @param learning_rate is a positive number that controls the rate at which connections are adjusted
#' @return Neural_Network
#' @title back prop

setMethod(f = "back_propogation",
          signature = c("Neural_Network","numeric","numeric","numeric"),
          valueClass = "Neural_Network",
          definition = function(object,target,regularization_parameter,learning_rate){
            object@delta_terms[[1]] <- NULL
            number_of_network_layers <- length(object@weight_matrices) + 1
            for(network_layer_index in number_of_network_layers:2){
              if(network_layer_index == number_of_network_layers){
                delta <- object@post_sigmoidal_outputs[[network_layer_index]] - target
                object@delta_terms[[network_layer_index]] <- delta
                next_post_sigmoidal_output <- object@post_sigmoidal_outputs[[network_layer_index-1]]
                intercept_terms <- rep(1,nrow(next_post_sigmoidal_output))
                next_post_sigmoidal_output <- cbind(intercept_terms,next_post_sigmoidal_output)
                update_matrix <- t(delta) %*% next_post_sigmoidal_output
                object@weight_updates[[network_layer_index-1]] <- update_matrix}
              else{
                previous_delta <- object@delta_terms[[network_layer_index + 1]]
                previous_weight_matrix <- object@weight_matrices[[network_layer_index]][,-1]
                current_sigmoidal_derivative <- object@sigmoidal_derivatives[[network_layer_index]]
                delta <- previous_delta %*% previous_weight_matrix * current_sigmoidal_derivative
                object@delta_terms[[network_layer_index]] <- delta
                next_post_sigmoidal_output <- object@post_sigmoidal_outputs[[network_layer_index-1]]
                intercept_terms <- rep(1,nrow(next_post_sigmoidal_output))
                next_post_sigmoidal_output <- cbind(intercept_terms,next_post_sigmoidal_output)
                update_matrix <- t(delta) %*% next_post_sigmoidal_output
                object@weight_updates[[network_layer_index-1]] <- update_matrix}
              weights <- object@weight_matrices[[network_layer_index - 1]]
              updates <- object@weight_updates[[network_layer_index - 1]]
              number_observations <- Get_Number_Observations(object)
              weights <- weights - learning_rate*( (updates/number_observations) + (regularization_parameter*weights) ) # is this the update formula??
              object@weight_matrices[[network_layer_index - 1]] <- weights}
            return(object)})

setGeneric(name = "Predict",
           valueClass = "numeric",
           def = function(object,dataset){standardGeneric("Predict")})

#' @title predict stuff
#' @description returns predictions
#' @param object : a neural network
#' @param dataset : a dataframe of features and observations
#' @return Numeric

setMethod(f = "Predict",
          signature = c("Neural_Network","data.frame"),
          valueClass = "numeric",
          definition = function(object,dataset){
            target <- ncol(dataset)
            Dataset <- dataset[,-target]
            Dataset <- as.matrix(Dataset)
            object <- forward_propogation(object,Dataset)
            prediction_layer_index <- length(object@post_sigmoidal_outputs)
            predictions <- drop(object@post_sigmoidal_outputs[[prediction_layer_index]])
            return(predictions)})

#' @description get log loss
#' @param predictions is a numeric vector
#' @param target is a numeric vector
#' @return Numeric
#' @title log loss

Get_LogLoss <- function(predictions,target){
  predictions <- drop(predictions)
  target <- drop(target)
  number_observations <- length(target)
  log_loss <- sum((-target)*log(predictions) - (1-target)*log(1-predictions))
  log_loss <- log_loss / number_observations
  return(log_loss)}

setGeneric(name = "Get_Cost",
           valueClass = "numeric",
           def = function(object,target){standardGeneric("Get_Cost")})

#' @description get the logarithmic loss for a set of predictions
#' @param object ... a Neural_Network that has run forward_prop at least once
#' @param target ... a numeric vector ... the target ...
#' @return Numeric
#' @title cost

setMethod(f = "Get_Cost",
          signature = c("Neural_Network","numeric"),
          valueClass = "numeric",
          definition = function(object,target){
            number_predictions <- Get_Number_Observations(object)
            predictions <- object@post_sigmoidal_outputs[[length(object@post_sigmoidal_outputs)]]
            log_loss <- sum((-target)*log(predictions) - (1-target)*log(1-predictions))
            log_loss <- log_loss / number_predictions
            return(log_loss)
          })

setGeneric(name = "Get_Number_Observations",
           valueClass = "numeric",
           def = function(object){standardGeneric("Get_Number_Observations")})

#' @description returns the number of observations that the network has processed
#' @param object ... a Neural Network that has called fprop. ie. that has called train/predict
#' @return Numeric
#' @title num observs

setMethod(f = "Get_Number_Observations",
          signature = c("Neural_Network"),
          valueClass = "numeric",
          definition = function(object){
          prediction_layer_index <- length(object@post_sigmoidal_outputs)
          predictions <- drop(object@post_sigmoidal_outputs[[prediction_layer_index]])
          number_observations <- length(predictions)
          return(number_observations)
          })

setGeneric(name = "Train",
           valueClass = "Neural_Network",
           def = function(object,dataset,regularization_constant,learning_rate,tolerable_error){
             standardGeneric("Train")})
#' @description gets NN parameters that minimize cost on dataset using optimization_method
#' @param object is a Neural Network
#' @param dataset is a data.frame, the original data frame that includes the target
#' @param learning_rate is a numeric
#' @param regularization_constant is a numeric
#' @param tolerable_error is a numeric, units : log loss
#' @return Neural_Network
#' @title train the NN

setMethod(f = "Train",
          signature = c("Neural_Network","data.frame","numeric","numeric","numeric"),
          valueClass = "Neural_Network",
          definition = function(object,dataset,regularization_constant,learning_rate,tolerable_error){
            previous_cost <- 0
            cost <- 1
            data_matrix <- as.matrix(dataset)
            last_column_index <- ncol(data_matrix)
            target <- data_matrix[,last_column_index]
            data <- data_matrix[,-last_column_index]
            iter <- 1
            while(abs(previous_cost - cost) > tolerable_error){
              print(iter)
              iter <- iter + 1
              previous_cost <- cost
              object <- forward_propogation(object,data)
              cost <- Get_Cost(object,target)
              print(cost)
              object <- back_propogation(object,target,regularization_constant,learning_rate)}
            return(object)})

