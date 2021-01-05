# Breast Cancer Prediction


library(ggplot2)

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

# We can chech the relation between Diag and concave_points and several other variables
p1 <- ggplot(data=data, mapping = aes(x = concave_points, fill = Diag))
p1 <- p1 + geom_histogram(aes(y = ..density..), alpha = 0.6) + ggtitle("Distribution Concave Points by Diagnosis")
p1

pairs(data[,1:11])
pairs(data[,c(1,12:21)])
pairs(data[,c(1,22:31)])

summary(data)

table(data$Diag) #B: 357 M: 212

sum(is.na(data)) #0 -> no missing data

sum(duplicated(data)) #0 -> no duplicate data

# Test/Train Split
set.seed(2)
numObs = dim(data)[1]
train.idx = sample(1:numObs,0.8*numObs) # Reserve 20% of data for testing (CV used throughout, no need for validation set)

train = data[train.idx,]
test = data[-train.idx,]

##############################################
# SVM Classifier
##############################################

library(e1071) #For SVM

# Function for Evaluating SVM Performance
eval_perf <- function(clf, data) {
  ypred = predict(clf, data)
  
  #Print out confusion matrix
  print(table(ypred, truth=data$Diag))
  
  #Return Accuracy of Model
  return(list(sum(diag(prop.table(table(ypred, truth=data$Diag)))),ypred))
}

# Function for Pretty Plot of Confusion Matrix
plot_conf <- function(pred, truth) {
  conf_mat = as.data.frame(table(predicted=pred,truth=truth))
  ggplot(data=conf_mat, mapping=aes(x=truth,y=predicted))+
    geom_tile(aes(fill=Freq))+
    geom_text(aes(label=sprintf("%1.0f", Freq)), vjust=0.5)+
    scale_fill_gradient(low="white",
                        high="green",
                        trans="log",
                        guide=FALSE)+
    xlim(rev(levels(conf_mat$truth)))
}

# TRAIN

# For the document example plot, we can train the model as following:
svm.hpo = tune(svm, Diag~max_concpts+max_perim, data=train, 
               ranges=list(cost=c(0.1,0.5,1,2,5,10,20,50), 
                           kernel=c('radial','linear','polynomial')), 
               scale=TRUE)
# The best model for plotting
svm.hpo$best.model # radial and cost 2
# Trying some plot of the svmfit
plot(svm.hpo$best.model, train, max_concpts~max_perim)

# 10 fold CV with HPO on 'Cost' and 'Kernel'
svm.hpo = tune(svm, Diag~., data=train, ranges=list(cost=c(0.1,0.5,1,2,5,10,20,50), kernel=c('radial','linear','polynomial')), scale=TRUE)
summary(svm.hpo) # C=2, radial basis function

perf.hpo = eval_perf(svm.hpo$best.model, train) 
perf.hpo[1] #98.9%

plot_conf(perf.hpo[[2]], train$Diag)

# Check if scale=FALSE improves performance (as this change can cause 'max iterations reached' error)
svm.noscale = tune(svm, Diag~., data=train, kernel='radial', scale=FALSE)
summary(svm.noscale) # error: 0.36  -> Stick with scale=TRUE

# The svm below uses only the variables which were identified as significant using the maximum AIC method in the LogReg section below (stepAIC())
# This svm achieves 98.6% training accuracy with just 16 variables(/30)
svm.out = tune(svm, Diag~max_concpts+max_perim+max_smooth+fractal_dim+smoothness+radius+symmetry+max_area
              +concave_points+max_concavity+compactness+se_concpts+max_symm+se_fractdim+se_radius+max_text, data=train, kernel='radial',
              cost=2, scale=TRUE)
summary(svm.out)
summary(svm.out$best.model)
perf.out = eval_perf(svm.out$best.model,train)
perf.out[1]



# Trying some plot of the svmfit
plot(svm.hpo$best.model, train, max_perim ~ max_area)

# Test
test.hpo=eval_perf(svm.hpo$best.model, test) 
test.hpo[1] #Acc: 98.2%

plot_conf(test.hpo[[2]], test$Diag)
####ADD ROC CURVE????#####


#####################################################################
# Logistic Regression
#####################################################################
library(MASS) # for the stepAIC method

trainlm = train
trainlm$Diag = as.numeric(trainlm$Diag)-1
testlm = test
testlm$Diag = as.numeric(testlm$Diag)-1

# Train
lm = tune(glm,Diag~.,data=trainlm, family='binomial')
summary(lm)
lm.probs = predict(lm$best.model,type="response")
lm.pred = rep(0,length(lm.probs))
lm.pred[lm.probs>.5]=1
table(lm.pred, trainlm$Diag)

sum(diag(prop.table(table(lm.pred, truth=trainlm$Diag)))) # 100% Accuracy (Overfitted)
mod.select = stepAIC(lm) #this iteratively determines the best model using the AIC value

####################################################################
# Neural Network
####################################################################
library(tensorflow)
install_tensorflow(method='conda', envname='r-reticulate')
library(keras)
library(tensorflow)
library(dplyr) 
library(tfdatasets) # needed for 'feature_spec' function

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
y_train_vec = as.numeric(y_train)-1
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
    epochs =500,
    batch_size = 5, 
    validation_split = 0.2,
    verbose=0,
    callbacks = list(print_dot_callback)
  )
  
  train_score <- model %>% evaluate(x_train, y_train, batch_size=5)
  
  return(list(model,train_score,history))
}

layers = c(64,32,16)
res = train_model(layers)
plot(res[[3]])
#relu:
#64/32/16 : 94.3
#64/32/32 : 93.2
#64/32 : 92.8
#64/64/32: 88.6,93.9
#Epochs 300:
#64/32/16 : 94.9
#Epochs 400: 
#64/32/16 : 96.0 #Choose this, validation results start to diverge after this point (see plot)
#Epochs 500:
#64/32/16 : 95.6
#sigmoid:
#64/32/16 : 92.3
#64/32/32 : 90.3


classes <- res[[1]] %>% predict_classes(x_train, batch_size=5)
table(y_train_vec, classes)


# Evaluate on Test Data
score <- res[[1]] %>% evaluate(x_test, y_test, batch_size = 5) # 97.4%

classes <- res[[1]] %>% predict_classes(x_test, batch_size=5)
table(y_test_vec, classes)


