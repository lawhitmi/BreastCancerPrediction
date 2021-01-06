# Breast Cancer Prediction


library(ggplot2)
library(pROC)

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

table(data$Diag) #B: 357 M: 212

sum(is.na(data)) #0 -> no missing data

sum(duplicated(data)) #0 -> no duplicate data

# Test/Train Split
set.seed(2)
numObs = dim(data)[1]
train.idx = sample(1:numObs,0.8*numObs) # Reserve 20% of data for testing (CV used throughout, no need for validation set)

train = data[train.idx,]
test = data[-train.idx,]

# EDA

# We can check the relation between Diag and variables
for (i in colnames(train)[-1]) {
  p1 <- ggplot(data=train, mapping = aes(x = !!rlang::sym(i), fill = Diag))
  p1 <- p1 + geom_histogram(aes(y = ..density..), alpha = 0.6, bins = 30) + ggtitle(paste0('Distribution of ', i, ' by Diagnosis'))
  print(p1)
}

# Look for correlation between variables
pairs(train[,1:11])
pairs(train[,c(1,12:21)])
pairs(train[,c(1,22:31)])

summary(train)

# PCA
pr.out<-prcomp(train[,-1],scale.=TRUE)
plot(pr.out$x[,1:2], col=as.numeric(train[,1]))

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
plot_conf <- function(pred, truth, title) {
  conf_mat = as.data.frame(table(predicted=pred,truth=truth))
  ggplot(data=conf_mat, mapping=aes(x=truth,y=predicted))+
    geom_tile(aes(fill=Freq))+
    geom_text(aes(label=sprintf("%1.0f", Freq)), vjust=0.5)+
    scale_fill_gradient(low="white",
                        high="green",
                        trans="log",
                        guide=FALSE)+
    xlim(rev(levels(conf_mat$truth)))+
    ggtitle(title)
}

# TRAIN

######## Model trained for obtaining plot to demonstrate SVM Method ####################
# For the document example plot, we can train the model as following:
svm.hpo = tune(svm, Diag~max_concpts+max_perim, data=train, 
               ranges=list(cost=c(0.1,0.5,1,2,5,10,20,50), 
                           kernel=c('radial','linear','polynomial')), 
               scale=TRUE)
# The best model for plotting
svm.hpo$best.model # radial and cost 2
# Trying some plots of the svmfit
plot(svm.hpo$best.model, train, max_concpts~max_perim)
########################################################################################

# 10 fold CV with HPO on 'Cost' and 'Kernel'
svm.hpo = tune(svm, Diag~., data=train, ranges=list(cost=c(0.1,0.5,1,2,5,10,20,50), kernel=c('radial','linear','polynomial')), scale=TRUE)
summary(svm.hpo) # C=2, radial basis function

perf.hpo = eval_perf(svm.hpo$best.model, train) 
perf.hpo[1] #98.9%

plot_conf(perf.hpo[[2]], train$Diag, 'SVM Training Confusion Matrix')

# Check if scale=FALSE improves performance (as this change can cause 'max iterations reached' error)
svm.noscale = tune(svm, Diag~., data=train, kernel='radial', scale=FALSE)
summary(svm.noscale) # error: 0.36 vs. ~0.03 for scaled  -> Stick with scale=TRUE

# 10 fold CV with HPO on 'Cost' and 'Kernel' and PCA preprocessing
train_pca = cbind(as.data.frame(train$Diag),as.data.frame(pr.out$x[,1:15]))
colnames(train_pca)[1] = 'Diag'
svm.pca = tune(svm, Diag~., data=train_pca,
               ranges=list(cost=c(0.1,0.5,1,2,5,10,20,50), kernel=c('radial','linear','polynomial')), scale=FALSE)
summary(svm.pca) # C=0.1, linear

perf.pca = eval_perf(svm.pca$best.model, train_pca) 
perf.pca[1] # 98.6%


# Feature Selection Using Logistic Regression
#####################################################################
library(MASS) # for the stepAIC method

trainlm = train
trainlm$Diag = as.numeric(trainlm$Diag)-1
testlm = test
testlm$Diag = as.numeric(testlm$Diag)-1

# LM TRAIN
lm = glm(Diag~.,data=trainlm, family='binomial') 
# glm.fit: algorithm did not converge
# glm.fit: fitted probabilities numerically 0 or 1 occurred
# Training data could be separable, or probabilities of some points are indistinguishable from 0.
# didn't further investigate this as the StepAIC method below still gives some results.

summary(lm)
lm.probs = predict(lm,type="response")
lm.pred = rep(0,length(lm.probs))
lm.pred[lm.probs>.5]=1
table(lm.pred, trainlm$Diag)

sum(diag(prop.table(table(lm.pred, truth=trainlm$Diag)))) # 100% Accuracy 
mod.select = stepAIC(lm) #this iteratively determines the best model using the AIC value

# The svm below uses only the variables which were identified as significant using the stepAIC method in the LogReg section 
# This svm achieves 98.6% training accuracy with just 16 variables(/30)
svm.out = tune(svm, Diag~max_concpts+max_perim+max_smooth+fractal_dim+smoothness+radius+symmetry+max_area
              +concave_points+max_concavity+compactness+se_concpts+max_symm+se_fractdim+se_radius+max_text, data=train,
              ranges=list(cost=c(0.1,0.5,1,2,5,10,20,50), kernel=c('radial','linear','polynomial')), scale=TRUE)
summary(svm.out)
summary(svm.out$best.model) #C=1, linear
perf.out = eval_perf(svm.out$best.model,train)
perf.out[1] #98.6


# Model below uses just those features which appear to have a visual separation between the distributions for malignant and benign classes
svm.visselect = tune(svm, Diag~max_concpts+max_perim+max_rad+concave_points+concavity+max_concavity+max_area+area+perimeter+radius,
                     data=train,ranges=list(cost=c(0.1,0.5,1,2,5,10,20,50), kernel=c('radial','linear','polynomial')), scale=TRUE)
summary(svm.visselect) # C=20, radial
perf.visselect = eval_perf(svm.visselect$best.model, train)
perf.visselect[1] # 97.4


# TEST
# Select model using all features
best.svm = svm(Diag~., data=train, prob=TRUE, cost=2, scale=TRUE, kernel='radial')

test.hpo=eval_perf(best.svm, test) 
test.hpo[1] #Acc: 98.2%

plot_conf(test.hpo[[2]], test$Diag, 'SVM Test Confusion Matrix')

#Identify which points were misclassified
test.hpo[[2]][test.hpo[[2]]!=test$Diag]

#Attempt to visualize the misclassified points
pr.out.test<-prcomp(test[,-1],scale.=TRUE)
plot(pr.out.test$x[,1:2], col=as.numeric(test[,1]),main="Misclassified Points - SVM")
text(pr.out.test$x["69",1],pr.out.test$x["69",2], col=as.numeric(test["69",1]), label=69)
text(pr.out.test$x["74",1],pr.out.test$x["74",2], col=as.numeric(test["74",1]), label=74)
legend(-12, 9, legend=c("Benign","Malignant"), col=c("black","red"), pch=1)

# ROC curve
test.pred = predict(best.svm, test, prob=TRUE)
test.predprob = attr(test.pred, 'probabilities')[,1]
roc.obj1 = roc(test$Diag, test.predprob)
ggroc(roc.obj1)
auc(roc.obj1)


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
    #optimizer = optimizer_adam(lr=0.01),
    metrics = 'accuracy'
  )

  # Fit the model and
  # Store the fitting history in `history` 
  history <- model %>% fit(
    x_train, 
    y_train, 
    epochs = 400,
    batch_size = 5, 
    validation_split = 0.2,
    verbose=0,
    callbacks = list(print_dot_callback)
  )
  
  train_score <- model %>% evaluate(x_train, y_train, batch_size=5)
  
  return(list(model,train_score,history))
}

# TRAIN
 
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
#Epochs 400: (batch=5, lr=default from Adam, (64/32/16), binary_crossentropy) 
#32/32/16 : 95.6
#64/32/16 : 96.0 #Choose this setup
#Epochs 500:
#64/32/16 : 95.6
#sigmoid:
#64/32/16 : 92.3
#64/32/32 : 90.3


classes <- res[[1]] %>% predict_classes(x_train, batch_size=5)
table(y_train_vec, classes)


# TEST
score <- res[[1]] %>% evaluate(x_test, y_test, batch_size = 5) # 98.2%

classes <- res[[1]] %>% predict_classes(x_test, batch_size=5)
table(y_test_vec, classes)

plot_conf(classes, y_test_vec, "Neural Network Test Confusion Matrix")

# Identify Misclassified Points

test[classes!=y_test_vec,] #15, 482

plot(pr.out.test$x[,1:2], col=as.numeric(test[,1]), main="Misclassified Points - NN")
text(pr.out.test$x["15",1],pr.out.test$x["15",2], col=as.numeric(test["15",1]), label=15)
text(pr.out.test$x["482",1],pr.out.test$x["482",2], col=as.numeric(test["482",1]), label=482)
legend(-12, 9, legend=c("Benign","Malignant"), col=c("black","red"), pch=1)
