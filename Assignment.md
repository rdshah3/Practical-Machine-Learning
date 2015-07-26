---
title: "Practical Machine Learning - Prediction Assignment Writeup"
author: "Roshan Shah"
date: "Thursday, July 23, 2015"
output: html_document
---

Background
--------------------------------------------
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

The goal of the project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

Data
--------------------------------------------

First, we need to read in the training and test data sets.

```{r, load data} 
training <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",na.strings=c("", "NA", "NULL"))
testing <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",na.strings=c("", "NA", "NULL"))
```

```{r}
dim(training)
dim(testing)
```

Let's get some summary statistics on the training dataset.
```{r, results='hide'}
summary(training)
```

- Remove variables that don't have predictive value
```{r}
remove <- c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2","cvtd_timestamp","new_window", "num_window")
training <- training[,!(names(training) %in% remove)]
dim(training)
```

- The summary stats revealed many variables with NAs. We could try to impute these variables using method "knnImpute" from the preProcess function, but since we have so many covariates, let's just filter them out.
  
```{r}
training <- training[ , colSums(is.na(training)) == 0]
dim(training)
```

Before we start building a model, let's remove the zero covariates


```{r, message=FALSE, echo=FALSE}
library(caret)
```
```{r}
any(nearZeroVar(training, saveMetrics = TRUE)$nzv == TRUE)
```

None of the predictors have near zero variance, so we will not eliminate any based on this criteria. However, let's continue pre-processing by evaluating the correlation between variables and remove predictors with a correlation of greater than 0.90. 

```{r, message=FALSE, echo=FALSE}
library(corrplot)
```
```{r}
M <- abs(cor(training[,-53]))
```
```{r, eval=FALSE}
corrplot(M, type="lower", tl.cex = 0.5, main="Correlation Matrix")
```
```{r, echo=FALSE}
corrplot(M, type="lower", tl.cex = 0.5, main="\nCorrelation Matrix")
```
```{r}
training <- training[,-findCorrelation(M, cutoff = 0.90)]
dim(training)
```

After pre-processing, we now have 46 covariates (45 not including the classe variable).

Building the Model
------------------
First, we need to create a training and validation set for cross validation.

```{r}
inTrain <- createDataPartition(y=training$classe, p=0.7, list=FALSE)
training_post <- training[inTrain,]
validation_post <- training[-inTrain,]
```

```{r}
dim(training_post)
dim(validation_post)
```

After identifying a lack of correlation between the classe variable and the predictors, a random forest model was chosen.

```{r}
cor <- abs(sapply(colnames(training_post[, -ncol(training)]), function(x) cor(as.numeric(training_post[, x]), as.numeric(training_post$classe), method = "spearman")))
```

Build a random forest model using the training data set and 4-fold cross validation. 

```{r, Train model}
set.seed(3456)
library(randomForest)
modelFit <- train(classe ~ ., method="rf", data=training_post, trControl=trainControl(method="cv", number=4), importance=TRUE, allowParallel=TRUE)
modelFit$finalModel
# closeAllConnections()
```

Evaluate the accuracy of the model by applying it to the validation dataset.

```{r}
validation_predict <- predict(modelFit, newdata=validation_post)
cf <- confusionMatrix(validation_predict, validation_post$classe)
cf
```

Evaluate the importance of the predictors.

```{r}
imp <- varImp(modelFit)$importance
varImpPlot(modelFit$finalModel, sort = TRUE, main = "Importance of the Predictors")

```

After predicting results from the validation dataset, the random forest model, using 4-fold cross validation, has an accuracy of `r cf$overall["Accuracy"]`, giving us an out-of-sample error of `r (1-cf$overall["Accuracy"])*100`%

Prediction
--------------------------------
Finally, let's use the validated model to predict values from the testing dataset.

```{r}
testing_pred <- predict(modelFit, newdata=testing)
write_files <- function(x) {
        n <- length(x)
        for (i in 1:n) {
                filename <- paste0("problem_id", i, ".txt")
                write.table(x[i], file=filename, quote=FALSE, row.names=FALSE,col.names=FALSE)
        }
}
write_files(testing_pred)
```

