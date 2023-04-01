# set working directory
setwd("/Users/ayan/Desktop/BU/Spring 2023/CS699_project")

##file_path <- "~/Desktop/weather_data.csv"
##weather_data <- read.csv(file_path)

# import data file
weather_data <- read.csv("weather_data.csv")

# These columns can be used for classification:
# AWND, PRCP, PRCP_ATTRIBUTES, SNOW_ATTRIBUTES, TAVG, TMAX, TMAX_ATTRIBUTES == TMIN_ATTRIBUTES (can only use 1),
# TMIN, WDF2, WDF5, WSF2, WSF5, WT01, WT02, WT03, WT04, WT05, WT06, WT08, WT09, DATE ???

# these are the features mentioned above
features <- c(
  'AWND', 
  'PRCP', 
  'PRCP_ATTRIBUTES', 
  'SNOW_ATTRIBUTES', 
  'TAVG', 
  'TMAX', 
  'TMAX_ATTRIBUTES', 
  'TMIN', 
  'WDF2', 
  'WDF5', 
  'WSF2', 
  'WSF5', 
  'WT01', 
  'WT02', 
  'WT03', 
  'WT04', 
  'WT05', 
  'WT06', 
  'WT08', 
  'WT09',
  'weather_condition'
)

# we take only the columns/features we need
weather_data_subset <- weather_data[features]

# make result column a factor
weather_data_subset$weather_condition <- as.factor(weather_data_subset$weather_condition)

# change categorical variables to numeric
weather_data_subset$PRCP_ATTRIBUTES <- factor(
  weather_data_subset$PRCP_ATTRIBUTES, 
  levels = c(',,D,2400', ',,W,2400', 'T,,D,2400', 'T,,W,2400'),
  labels = c(2, 4, 1, 3)
)

weather_data_subset$SNOW_ATTRIBUTES <- factor(
  weather_data_subset$SNOW_ATTRIBUTES, 
  levels = c(',,D', ',,W', 'T,,D', 'T,,W'),
  labels = c(2, 4, 1, 3)
)

weather_data_subset$TMAX_ATTRIBUTES <- factor(
  weather_data_subset$TMAX_ATTRIBUTES, 
  levels = c(',,D', ',,W'),
  labels = c(1, 2)
)

# fill NA values with 0
weather_data_subset$WT01[is.na(weather_data_subset$WT01)] <- 0
weather_data_subset$WT02[is.na(weather_data_subset$WT02)] <- 0
weather_data_subset$WT03[is.na(weather_data_subset$WT03)] <- 0
weather_data_subset$WT04[is.na(weather_data_subset$WT04)] <- 0
weather_data_subset$WT05[is.na(weather_data_subset$WT05)] <- 0
weather_data_subset$WT06[is.na(weather_data_subset$WT06)] <- 0
weather_data_subset$WT08[is.na(weather_data_subset$WT08)] <- 0
weather_data_subset$WT09[is.na(weather_data_subset$WT09)] <- 0
weather_data_subset$WDF5[is.na(weather_data_subset$WDF5)] <- 0
weather_data_subset$WSF5[is.na(weather_data_subset$WSF5)] <- 0

# scale numeric column to mean = 0 and sd = 1
weather_data_subset$AWND <- scale(weather_data_subset$AWND)
weather_data_subset$PRCP <- scale(weather_data_subset$PRCP)
weather_data_subset$TAVG <- scale(weather_data_subset$TAVG)
weather_data_subset$TMAX <- scale(weather_data_subset$TMAX)
weather_data_subset$TMIN <- scale(weather_data_subset$TMIN)
weather_data_subset$WDF2 <- scale(weather_data_subset$WDF2)
weather_data_subset$WDF5 <- scale(weather_data_subset$WDF5)
weather_data_subset$WSF2 <- scale(weather_data_subset$WSF2)
weather_data_subset$WSF5 <- scale(weather_data_subset$WSF5)

################################################################################

# train-test split
library(rsample)
set.seed(17)

nearZeroVar(weather_data_subset)
weather_data_subset_fixed <- weather_data_subset[,-c(7, 14, 16, 17, 18, 20)]
weather_data_subset_fixed

split <- initial_split(weather_data_subset_fixed, prop = 0.70, strata = weather_condition)
train <- training(split)
test <- testing(split)

write.csv(train, "/Users/ayan/Desktop/BU/Spring 2023/CS699_project/train.csv", row.names=FALSE)
write.csv(test, "/Users/ayan/Desktop/BU/Spring 2023/CS699_project/test.csv", row.names=FALSE)

# train <- read.csv("/Users/ayan/Desktop/BU/Spring 2023/CS699_project/train.csv")
# test <- read.csv("/Users/ayan/Desktop/BU/Spring 2023/CS699_project/test.csv")

################################################################################

# Now, we need to use the 5 feature selection methods on the train and test datasets.

# Feature Selection
# Method 1: Information Gain
install.packages("FSelector")
library(FSelector) #requires JAVA; run in cloud if necessary
weights <- information.gain(weather_condition~., train)
weights

new_weights <- subset(weights, attr_importance > 0)
new_weights

ig_train <- train[, c("PRCP", "PRCP_ATTRIBUTES", "SNOW_ATTRIBUTES", "TAVG", "TMAX", "TMIN", "WT01", "weather_condition")]
ig_test <- test[, c("PRCP", "PRCP_ATTRIBUTES", "SNOW_ATTRIBUTES", "TAVG", "TMAX", "TMIN", "WT01", "weather_condition")]

ctrl <- trainControl(method = "repeatedcv", 
                     number = 10, 
                     repeats = 5, 
                     verboseIter = FALSE,
                     sampling = "up")

##Information Gain - Naive Bayes (1/25)
nb_grid <- expand.grid(usekernel = c(TRUE), fL = 0:5, adjust = seq(0, 5, by = 1))
ig_nb_model <- train(weather_condition ~ .,
                         data = ig_train,
                         method = "nb",
                         preProcess = c("scale", "center"),
                         trControl = ctrl,
                         tuneGrid = nb_grid)
ig_nb_model
test_pred_ig_nb <- predict(ig_nb_model, newdata = ig_test)
CM1 <- confusionMatrix(test_pred_ig_nb, ig_test$weather_condition, mode = "everything")
saveRDS(CM1, file="CM1.RData")

##Information Gain - AdaBoost (2/25)
ada_grid <- expand.grid(iter = 10, maxdepth = 1:10, nu = seq(0.1, 1, by=0.1))
ig_ada_model <- train(weather_condition ~ .,
                          data = ig_train,
                          method = "ada",
                          preProcess = c("scale", "center"),
                          trControl = ctrl,
                          tuneGrid = ada_grid)
ig_ada_model
test_pred_ig_ada <- predict(ig_ada_model, newdata = ig_test)
CM2 <- confusionMatrix(test_pred_ig_ada, ig_test$weather_condition, mode = "everything")
saveRDS(CM2, file="CM2.RData")

##Information Gain - RPart (3/25)
rpart_grid <- expand.grid(cp = seq(0.1, 1, by = 0.1))
ig_rpart_model <- train(weather_condition ~ .,
                            data = ig_train,
                            method = "rpart",
                            preProcess = c("scale", "center"),
                            trControl = ctrl,
                            tuneGrid = rpart_grid)
ig_rpart_model
test_pred_ig_rpart <- predict(ig_rpart_model, newdata = ig_test)
CM3 <- confusionMatrix(test_pred_ig_rpart, ig_test$weather_condition, mode = "everything")
saveRDS(CM3, file="CM3.RData")

##Information Gain - GLM (4/25)
glm_grid <- expand.grid(.parameter = seq(1, 10, 1))
ig_glm_model <- train(weather_condition ~ .,
                          data = ig_train, 
                          method = "glm",
                          preProcess = c("scale", "center"),
                          trControl = ctrl,
                          tuneGrid = glm_grid)
ig_glm_model
test_pred_ig_glm <- predict(ig_glm_model, newdata = ig_test)
CM4 <- confusionMatrix(test_pred_ig_glm, ig_test$weather_condition, mode = "everything")
saveRDS(CM4, file="CM4.RData")

##Information Gain - Random Forest (5/25)
rf_grid <- expand.grid(mtry = 1:9)
ig_rf_model <- train(weather_condition ~ .,
                         data = ig_train, 
                         method = "rf",
                         preProcess = c("scale", "center"),
                         trControl = ctrl,
                         tuneGrid = rf_grid)
ig_rf_model
test_pred_ig_rf <- predict(ig_rf_model, newdata = ig_test)
CM5 <- confusionMatrix(test_pred_ig_rf, ig_test$weather_condition, mode = "everything")
saveRDS(CM5, file="CM5.RData")

#2 Method 2: BORUTA
install.packages("Boruta")
library(Boruta)
library(caret)

set.seed(17)
boruta_output <- Boruta(weather_condition ~ ., data=train, doTrace=0)
boruta_signif <- getSelectedAttributes(boruta_output, withTentative = TRUE)
print(boruta_signif)  

roughFixMod <- TentativeRoughFix(boruta_output)
boruta_signif <- getSelectedAttributes(roughFixMod)
print(boruta_signif)

imps <- attStats(roughFixMod)
imps2 <- imps[imps$decision != 'Rejected', c('meanImp', 'decision')]
imps2[order(-imps2$meanImp), ]
plot(boruta_output, cex.axis=.7, las=2, xlab="", main="Variable Importance")

boruta_train <- train[, c("PRCP","TMAX", "WT01", "TAVG", "TMIN", "SNOW_ATTRIBUTES", "weather_condition")]
boruta_test <- test[, c("PRCP","TMAX", "WT01", "TAVG", "TMIN", "SNOW_ATTRIBUTES", "weather_condition")]

ctrl <- trainControl(method = "repeatedcv", 
                     number = 10, 
                     repeats = 5, 
                     verboseIter = FALSE,
                     sampling = "up")

##Boruta - Naive Bayes (6/25)
nb_grid <- expand.grid(usekernel = c(TRUE), fL = 0:5, adjust = seq(0, 5, by = 1))
boruta_nb_model <- train(weather_condition ~ .,
                         data = boruta_train,
                         method = "nb",
                         preProcess = c("scale", "center"),
                         trControl = ctrl,
                         tuneGrid = nb_grid)
boruta_nb_model
test_pred_boruta_nb <- predict(boruta_nb_model, newdata = boruta_test)
CM6 <- confusionMatrix(test_pred_boruta_nb, boruta_test$weather_condition, mode = "everything")
saveRDS(CM6, file="CM6.RData")
CM6_copy <- readRDS("CM6.RData")
CM6_copy

##Boruta - AdaBoost (7/25)
ada_grid <- expand.grid(iter = 10, maxdepth = 1:10, nu = seq(0.1, 1, by=0.1))
boruta_ada_model <- train(weather_condition ~ .,
                          data = boruta_train,
                          method = "ada",
                          preProcess = c("scale", "center"),
                          trControl = ctrl,
                          tuneGrid = ada_grid)
boruta_ada_model
test_pred_boruta_ada <- predict(boruta_ada_model, newdata = boruta_test)
CM7 <- confusionMatrix(test_pred_boruta_ada, boruta_test$weather_condition, mode = "everything")
saveRDS(CM7, file="CM7.RData")

##Boruta - RPart (8/25)
rpart_grid <- expand.grid(cp = seq(0.1, 1, by = 0.1))
boruta_rpart_model <- train(weather_condition ~ .,
                            data = boruta_train,
                            method = "rpart",
                            preProcess = c("scale", "center"),
                            trControl = ctrl,
                            tuneGrid = rpart_grid)
boruta_rpart_model
test_pred_boruta_rpart <- predict(boruta_rpart_model, newdata = boruta_test)
CM8 <- confusionMatrix(test_pred_boruta_rpart, boruta_test$weather_condition, mode = "everything")
saveRDS(CM8, file="CM8.RData")

##Boruta - GLM (9/25)
glm_grid <- expand.grid(.parameter = seq(1, 10, 1))
boruta_glm_model <- train(weather_condition ~ .,
                          data = boruta_train, 
                          method = "glm",
                          preProcess = c("scale", "center"),
                          trControl = ctrl,
                          tuneGrid = glm_grid)
boruta_glm_model
test_pred_boruta_glm <- predict(boruta_glm_model, newdata = boruta_test)
CM9 <- confusionMatrix(test_pred_boruta_glm, boruta_test$weather_condition, mode = "everything")
saveRDS(CM9, file="CM9.RData")

##Boruta - Random Forest (10/25)
rf_grid <- expand.grid(mtry = 1:9)
boruta_rf_model <- train(weather_condition ~ .,
                         data = boruta_train, 
                         method = "rf",
                         preProcess = c("scale", "center"),
                         trControl = ctrl,
                         tuneGrid = rf_grid)
boruta_rf_model 
test_pred_boruta_rf <- predict(boruta_rf_model, newdata = boruta_test)
CM10 <- confusionMatrix(test_pred_boruta_rf, boruta_test$weather_condition, mode = "everything")
saveRDS(CM10, file="CM10.RData")

# Method 3: Genetic Algorithm
library(caret)
ga_ctrl <- gafsControl(functions = rfGA,  # another option is `caretGA`.
                       method = "repeatedcv",
                       repeats = 3,
)
ga_obj <- gafs(x=train[, -ncol(train)],
               y=train$weather_condition,
               iters = 3,   # normally much higher (100+)
               gafsControl = ga_ctrl)
ga_obj
ga_obj$optVariables

ga_train <- train[, c("PRCP_ATTRIBUTES", "SNOW_ATTRIBUTES", "TAVG", "WT01", "WT03", "weather_condition")]
ga_test <- test[, c("PRCP_ATTRIBUTES", "SNOW_ATTRIBUTES", "TAVG", "WT01", "WT03", "weather_condition")]

ctrl <- trainControl(method = "repeatedcv", 
                     number = 10, 
                     repeats = 5, 
                     verboseIter = FALSE,
                     sampling = "up")

##Genetic Algorithm - Naive Bayes (11/25)
nb_grid <- expand.grid(usekernel = c(TRUE), fL = 0:5, adjust = seq(0, 5, by = 1))
ga_nb_model <- train(weather_condition ~ .,
                         data = ga_train,
                         method = "nb",
                         preProcess = c("scale", "center"),
                         trControl = ctrl,
                         tuneGrid = nb_grid)
ga_nb_model
test_pred_ga_nb <- predict(ga_nb_model, newdata = ga_test)
CM11 <- confusionMatrix(test_pred_ga_nb, ga_test$weather_condition, mode = "everything")
saveRDS(CM11, file="CM11.RData")

##Genetic Algorithm - AdaBoost (12/25)
ada_grid <- expand.grid(iter = 10, maxdepth = 1:10, nu = seq(0.1, 1, by=0.1))
ga_ada_model <- train(weather_condition ~ .,
                          data = ga_train,
                          method = "ada",
                          preProcess = c("scale", "center"),
                          trControl = ctrl,
                          tuneGrid = ada_grid)
ga_ada_model
test_pred_ga_ada <- predict(ga_ada_model, newdata = ga_test)
CM12 <- confusionMatrix(test_pred_ga_ada, ga_test$weather_condition, mode = "everything")
saveRDS(CM12, file="CM12.RData")

##Genetic Algorithm - RPart (13/25)
rpart_grid <- expand.grid(cp = seq(0.1, 1, by = 0.1))
ga_rpart_model <- train(weather_condition ~ .,
                            data = ga_train,
                            method = "rpart",
                            preProcess = c("scale", "center"),
                            trControl = ctrl,
                            tuneGrid = rpart_grid)
ga_rpart_model
test_pred_ga_rpart <- predict(ga_rpart_model, newdata = ga_test)
CM13 <- confusionMatrix(test_pred_ga_rpart, ga_test$weather_condition, mode = "everything")
saveRDS(CM13, file="CM13.RData")

##Genetic Algorithm - GLM (14/25)
glm_grid <- expand.grid(.parameter = seq(1, 10, 1))
ga_glm_model <- train(weather_condition ~ .,
                          data = ga_train, 
                          method = "glm",
                          preProcess = c("scale", "center"),
                          trControl = ctrl,
                          tuneGrid = glm_grid)
ga_glm_model
test_pred_ga_glm <- predict(ga_glm_model, newdata = ga_test)
CM14 <- confusionMatrix(test_pred_ga_glm, ga_test$weather_condition, mode = "everything")
saveRDS(CM14, file="CM14.RData")

##Genetic Algorithm - Random Forest (15/25)
rf_grid <- expand.grid(mtry = 1:9)
ga_rf_model <- train(weather_condition ~ .,
                         data = ga_train, 
                         method = "rf",
                         preProcess = c("scale", "center"),
                         trControl = ctrl,
                         tuneGrid = rf_grid)
ga_rf_model 
test_pred_ga_rf <- predict(ga_rf_model, newdata = ga_test)
CM15 <- confusionMatrix(test_pred_ga_rf, ga_test$weather_condition, mode = "everything")
saveRDS(CM15, file="CM15.RData")

# Method 4: Simulated Annealing
set.seed(17)
sa_ctrl <- safsControl(functions = rfSA,
                       method = "repeatedcv",
                       repeats = 3,
                       improve = 5) # n iterations without improvement before a reset

sa_obj <- safs(x=train[, -ncol(train)],
               y=train$weather_condition,
               safsControl = sa_ctrl)

sa_obj
sa_obj$optVariables

sa_train <- train[, c("SNOW_ATTRIBUTES", "TMAX", "WT08", "PRCP", "TAVG", "weather_condition")]
sa_test <- test[, c("SNOW_ATTRIBUTES", "TMAX", "WT08", "PRCP", "TAVG", "weather_condition")]

ctrl <- trainControl(method = "repeatedcv", 
                     number = 10, 
                     repeats = 5, 
                     verboseIter = FALSE,
                     sampling = "up")

#Simulated Annealing - Naive Bayes (16/25)
nb_grid <- expand.grid(usekernel = c(TRUE), fL = 0:5, adjust = seq(0, 5, by = 1))
sa_nb_model <- train(weather_condition ~ .,
                     data = sa_train,
                     method = "nb",
                     preProcess = c("scale", "center"),
                     trControl = ctrl,
                     tuneGrid = nb_grid)
sa_nb_model
test_pred_sa_nb <- predict(sa_nb_model, newdata = sa_test)
CM16 <- confusionMatrix(test_pred_sa_nb, sa_test$weather_condition, mode = "everything")
saveRDS(CM16, file="CM16.RData")

#Simulated Annealing - AdaBoost (17/25)
ada_grid <- expand.grid(iter = 10, maxdepth = 1:10, nu = seq(0.1, 1, by=0.1))
sa_ada_model <- train(weather_condition ~ .,
                      data = sa_train,
                      method = "ada",
                      preProcess = c("scale", "center"),
                      trControl = ctrl,
                      tuneGrid = ada_grid)
sa_ada_model
test_pred_sa_ada <- predict(sa_ada_model, newdata = sa_test)
CM17 <- confusionMatrix(test_pred_sa_ada, sa_test$weather_condition, mode = "everything")
saveRDS(CM17, file="CM17.RData")

#Simulated Annealing - RPart (18/25)
rpart_grid <- expand.grid(cp = seq(0.1, 1, by = 0.1))
sa_rpart_model <- train(weather_condition ~ .,
                        data = sa_train,
                        method = "rpart",
                        preProcess = c("scale", "center"),
                        trControl = ctrl,
                        tuneGrid = rpart_grid)
sa_rpart_model
test_pred_sa_rpart <- predict(sa_rpart_model, newdata = sa_test)
CM18 <- confusionMatrix(test_pred_sa_rpart, sa_test$weather_condition, mode = "everything")
saveRDS(CM18, file="CM18.RData")

##Simulated Annealing - GLM (19/25)
glm_grid <- expand.grid(.parameter = seq(1, 10, 1))
sa_glm_model <- train(weather_condition ~ .,
                      data = sa_train, 
                      method = "glm",
                      preProcess = c("scale", "center"),
                      trControl = ctrl,
                      tuneGrid = glm_grid)
sa_glm_model
test_pred_sa_glm <- predict(sa_glm_model, newdata = sa_test)
CM19 <- confusionMatrix(test_pred_sa_glm, sa_test$weather_condition, mode = "everything")
saveRDS(CM19, file="CM19.RData")


##Simulated Annealing - Random Forest (20/25)
rf_grid <- expand.grid(mtry = 1:9)
sa_rf_model <- train(weather_condition ~ .,
                     data = sa_train, 
                     method = "rf",
                     preProcess = c("scale", "center"),
                     trControl = ctrl,
                     tuneGrid = rf_grid)
sa_rf_model
test_pred_sa_rf <- predict(sa_rf_model, newdata = sa_test)
CM20 <- confusionMatrix(test_pred_sa_rf, sa_test$weather_condition, mode = "everything")
saveRDS(CM20, file="CM20.RData")

# Method 5: Recursive Feature Elimination
set.seed(17)
subsets <- c(1:5, 10, 14)

ctrl <- rfeControl(functions = rfFuncs,
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

rfe_obj <- rfe(x=train[, -ncol(train)],
               y=train$weather_condition,
               sizes = subsets,
               rfeControl = ctrl)

rfe_obj
rfe_obj$optVariables

rfe_train <- train[, c("PRCP","WT01","TMAX","TMIN","TAVG", "weather_condition")]
rfe_test <- test[, c("PRCP","WT01","TMAX","TMIN","TAVG", "weather_condition")]

ctrl <- trainControl(method = "repeatedcv", 
                     number = 10, 
                     repeats = 5, 
                     verboseIter = FALSE,
                     sampling = "up")

#RFE - Naive Bayes (21/25)
nb_grid <- expand.grid(usekernel = c(TRUE), fL = 0:5, adjust = seq(0, 5, by = 1))
rfe_nb_model <- train(weather_condition ~ .,
                      data = rfe_train,
                      method = "nb",
                      preProcess = c("scale", "center"),
                      trControl = ctrl,
                      tuneGrid = nb_grid)
rfe_nb_model
test_pred_rfe_nb <- predict(rfe_nb_model, newdata = rfe_test)
CM21 <- confusionMatrix(test_pred_rfe_nb, rfe_test$weather_condition, mode = "everything")
saveRDS(CM21, file="CM21.RData")

#RFE - AdaBoost (22/25)
ada_grid <- expand.grid(iter = 10, maxdepth = 1:10, nu = seq(0.1, 1, by=0.1))
rfe_ada_model <- train(weather_condition ~ .,
                       data = rfe_train,
                       method = "ada",
                       preProcess = c("scale", "center"),
                       trControl = ctrl,
                       tuneGrid = ada_grid)
rfe_ada_model 
test_pred_rfe_ada <- predict(rfe_ada_model, newdata = rfe_test)
CM22 <- confusionMatrix(test_pred_rfe_ada, rfe_test$weather_condition, mode = "everything")
saveRDS(CM22, file="CM22.RData")

#RFE - RPart (23/25)
rpart_grid <- expand.grid(cp = seq(0.1, 1, by = 0.1))
rfe_rpart_model <- train(weather_condition ~ .,
                         data = rfe_train,
                         method = "rpart",
                         preProcess = c("scale", "center"),
                         trControl = ctrl,
                         tuneGrid = rpart_grid)
rfe_rpart_model
test_pred_rfe_rpart <- predict(rfe_rpart_model, newdata = rfe_test)
CM23 <- confusionMatrix(test_pred_rfe_rpart, rfe_test$weather_condition, mode = "everything")
saveRDS(CM23, file="CM23.RData")

##RFE - GLM (24/25)
glm_grid <- expand.grid(.parameter = seq(1, 10, 1))
rfe_glm_model <- train(weather_condition ~ .,
                       data = rfe_train, 
                       method = "glm",
                       preProcess = c("scale", "center"),
                       trControl = ctrl,
                       tuneGrid = glm_grid)
rfe_glm_model
test_pred_rfe_glm <- predict(rfe_glm_model, newdata = rfe_test)
CM24 <- confusionMatrix(test_pred_rfe_glm, rfe_test$weather_condition, mode = "everything")
saveRDS(CM24, file="CM24.RData")

##SA - Random Forest (25/25)
rf_grid <- expand.grid(mtry = 1:9)
rfe_rf_model <- train(weather_condition ~ .,
                      data = rfe_train, 
                      method = "rf",
                      preProcess = c("scale", "center"),
                      trControl = ctrl,
                      tuneGrid = rf_grid)
rfe_rf_model
test_pred_rfe_rf <- predict(rfe_rf_model, newdata = rfe_test)
CM25 <- confusionMatrix(test_pred_rfe_rf, rfe_test$weather_condition, mode = "everything")
saveRDS(CM25, file="CM25.RData")

################################################################################
## Data Mining Results + Plots

#Information Gain Accuracies
ig_cms <- list(CM1, CM2, CM3, CM4, CM5)
ig_accuracies <- sapply(ig_cms, function(x) x$overall['Accuracy'])
col_names <- paste0(c("NB", "ADA", "RPART", "GLM", "RF"), "(IG)")
barplot(ig_accuracies, ylim = c(0, 1), xlab = "Classifiers", ylab = "Accuracy", 
        main = "Information Gain Accuracies", names.arg = col_names, 
        col = "blue")

#Boruta Accuracies
boruta_cms <- list(CM6, CM7, CM8, CM9, CM10)
boruta_accuracies <- sapply(boruta_cms, function(x) x$overall['Accuracy'])
col_names <- paste0(c("NB", "ADA", "RPART", "GLM", "RF"), "(B)")
barplot(boruta_accuracies, ylim = c(0, 1), xlab = "Classifiers", ylab = "Accuracy", 
        main = "Boruta Accuracies", names.arg = col_names, 
        col = "blue")

#Genetic Alg Accuracies
ga_cms <- list(CM11, CM12, CM13, CM14, CM15)
ga_accuracies <- sapply(ga_cms, function(x) x$overall['Accuracy'])
col_names <- paste0(c("NB", "ADA", "RPART", "GLM", "RF"), "(GA)")
barplot(ga_accuracies, ylim = c(0, 1), xlab = "Classifiers", ylab = "Accuracy", 
        main = "Genetic Algorithm Accuracies", names.arg = col_names, 
        col = "blue")

#Simulated Annealing Accuracies
sa_cms <- list(CM16, CM17, CM18, CM19, CM20)
sa_accuracies <- sapply(sa_cms, function(x) x$overall['Accuracy'])
col_names <- paste0(c("NB", "ADA", "RPART", "GLM", "RF"), "(SA)")
barplot(sa_accuracies, ylim = c(0, 1), xlab = "Classifiers", ylab = "Accuracy", 
        main = "Simulated Annealing Accuracies", names.arg = col_names, 
        col = "blue")

#Recursive Feature Elimination Accuracies
rfe_cms <- list(CM21, CM22, CM23, CM24, CM25)
rfe_accuracies <- sapply(rfe_cms, function(x) x$overall['Accuracy'])
col_names <- paste0(c("NB", "ADA", "RPART", "GLM", "RF"), "(RFE)")
barplot(rfe_accuracies, ylim = c(0, 1), xlab = "Classifiers", ylab = "Accuracy", 
        main = "Recursive Feature Elimination Accuracies", names.arg = col_names, 
        col = "blue")

###MAX accuracy
all_cms <- mget(paste0("CM", 1:25))
accuracies <- sapply(all_cms, function(x) x$overall['Accuracy'])
max_index <- which.max(accuracies)
highest_acc <- all_cms[[max_index]]
cat("Maximum accuracy is achieved at", accuracies[max_index], "by confusion matrix", names(all_cms)[max_index], "\n")

##Max Accuracy Plot
max_accuracy <- max(accuracies)
max_index <- which(accuracies == max_accuracy)
column <- rep("blue", length(accuracies))
column[max_index] <- "red"
barplot(accuracies, ylim = c(0, 1), xlab = "Classifiers", ylab = "Accuracy",
        main = "Accuracies", names.arg = col_names, col = column)

col_names <- paste0("CM", 1:25)
barplot(accuracies, ylim = c(0, 1), xlab = "Classifiers", ylab = "Accuracy", 
        main = "Accuracies", names.arg = col_names, 
        col = "blue")
