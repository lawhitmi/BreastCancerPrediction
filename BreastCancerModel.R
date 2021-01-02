# Breast Cancer Prediction

# Load Data
data = read.csv('Data/wdbc.data', header=FALSE)

summary(data)
dim(data) #569 x 32
# Add in column names from the documentation
colnames(data) = c('ID', 'Diag', 'radius', 'texture','perimeter', 'area', 'smoothness', 'compactness', 'concavity', 
                   'concave_points', 'symmetry', 'fractal_dim', 'se_radius', 'se_texture','se_perim', 'se_area', 'se_smooth', 
                   'se_compact', 'se_concavity', 'se_concpts', 'se_symm', 'se_fractdim', 'max_rad', 'max_text', 'max_perim', 
                   'max_area', 'max_smooth', 'max_compact', 'max_concavity', 'max_concpts', 'max_symm', 'max_fracdim')
# Remove ID column
data = subset(data, select=-ID)

# EDA
pairs(data[,1:11])
pairs(data[,c(1,12:21)])
pairs(data[,c(1,22:31)])

# Test/Train Split
set.seed(2)
numObs = dim(data)[1]
train.idx = sample(1:numObs,0.8*numObs)

train = data[train.idx,]
test = data[-train.idx,]

##############################################
# SVM Classifier
##############################################

library(e1071) #For SVM

# Train
svmfit = svm(Diag~.,data=train, kernel='linear', cost=20, scale=TRUE)
# The svm below uses only the variables which were identified as significant using the maximum AIC method in the LogReg section below (stepAIC())
# This svm achieves 98.6% training accuracy and 98.2% test accuracy with just 16 variables(/30)
svmfit1 = svm(Diag~max_concpts+max_perim+max_smooth+fractal_dim+smoothness+radius+symmetry+max_area
              +concave_points+max_concavity+compactness+se_concpts+max_symm+se_fractdim+se_radius+max_text, data=train, kernel='linear',
              cost=10, scale=TRUE)
summary(svmfit)


# Trying some plot of the svmfit
plot(svmfit, train, radius ~ max_area)

ypred = predict(svmfit, train)
table(ypred, truth=train$Diag)

sum(diag(prop.table(table(ypred, truth=train$Diag)))) #99.6% Accuracy

# Test
pred.test = predict(svmfit, test)
table(pred.test, truth=test$Diag)

sum(diag(prop.table(table(pred.test, truth=test$Diag)))) #98.2% Accuracy

#####################################################################
# Logistic Regression
#####################################################################
library(MASS) # for the stepAIC method

trainlm = train
trainlm$Diag = as.numeric(trainlm$Diag)-1
testlm = test
testlm$Diag = as.numeric(testlm$Diag)-1

# Train
lm = glm(Diag~.,data=trainlm, family='binomial')
summary(lm)
lm.probs = predict(lm,type="response")
lm.pred = rep(0,length(lm.probs))
lm.pred[lm.probs>.5]=1
table(lm.pred, trainlm$Diag)

sum(diag(prop.table(table(lm.pred, truth=trainlm$Diag)))) # 100% Accuracy
mod.select = stepAIC(lm) #this iteratively determines the best model using the AIC value
# Test
probs.test = predict(lm, testlm, type="response")
pred.test = rep(0,length(probs.test))
pred.test[probs.test>0.5] = 1
table(pred.test, testlm$Diag)
sum(diag(prop.table(table(pred.test, truth=testlm$Diag)))) # 94.7% Accuracy

####################################################################
# Neural Network
####################################################################
library(tensorflow)
install_tensorflow(method='conda', envname='r-reticulate')
library(keras)
library(tensorflow)
library(dplyr) 
library(tfdatasets) # needed for 'feature_spec' function
library(ggplot2)

x_train = train[,-1]
y_train = train[,1]
x_test = test[,-1]
y_test = test[,1]
x_train = as.matrix(x_train)
dimnames(x_train) <- NULL
x_test = as.matrix(x_test)
dimnames(x_test) <- NULL

# Normalize Independent variables
x_train = normalize(x_train, axis=2)
x_test = normalize(x_test, axis=2)

# One-hot Encode Target Variable
y_test_vec = as.numeric(y_test)-1
y_train = to_categorical(as.numeric(y_train)-1)
y_test = to_categorical(as.numeric(y_test)-1)
y_train = as.matrix(y_train)
y_test = as.matrix(y_test)

print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 80 == 0) cat("\n")
    cat(".")
  }
) 

train_model <- function(layers){
  # Initialize Model
  model <- keras_model_sequential()
  
  # Add layers from function input
  start = TRUE
  for(i in layers) {
    if (start==TRUE) {
      model %>% layer_dense(units = i, activation = 'relu', input_shape = c(30))
      prev_layer = i
      start = FALSE
    } else {
      model %>% layer_dense(units = i, activation = 'relu', input_shape = c(prev_layer))
      prev_layer = i
    }
  }
  #Add last layer
  model %>% layer_dense(units = 2, activation = 'softmax')
  
  
  model %>% compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam',
    metrics = 'accuracy'
  )

  # Fit the model and
  # Store the fitting history in `history` 
  history <- model %>% fit(
    x_train, 
    y_train, 
    epochs = 200,
    batch_size = 5, 
    validation_split = 0.2,
    verbose=0,
    callbacks = list(print_dot_callback)
  )
  
  plot(history)
  
  train_score <- model %>% evaluate(x_train, y_train, batch_size=5)
  
  return(c(model,train_score))
}

layers = c(64,64,32)
res = train_model(layers)
#64/32/16 : 94.3,
#64/32/32 : 93.2,
#64/32 : 92.8,
#64/64/32: 88.6,93.9


classes <- model %>% predict_classes(x_train, batch_size=5)
table(y_test_vec, classes)


# Evaluate on Test Data
score <- model %>% evaluate(x_test, y_test, batch_size = 5)

classes <- model %>% predict_classes(x_train, batch_size=5)
table(y_test_vec, classes)


