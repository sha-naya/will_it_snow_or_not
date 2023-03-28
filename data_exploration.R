# set working directory
setwd("/Users/ayan/Desktop/BU/Spring 2023/CS699_project")

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

################################################################################

# Now, we need to use the 5 feature selection methods on the train and test datasets.

# feature selection
#1 Information Gain
install.packages("FSelector")
library(FSelector) #requires JAVA; run in cloud if necessary
weights <- information.gain(weather_condition~., train)
weights

new_weights <- subset(weights, attr_importance > 0)
new_weights

ig_train <- train[, c("PRCP", "PRCP_ATTRIBUTES", "SNOW_ATTRIBUTES", "TAVG", "TMAX", "TMAX_ATTRIBUTES", "TMIN", "WT01", "WT04", "WT06", "WT09", "weather_condition")]
ig_test <- test[, c("PRCP", "PRCP_ATTRIBUTES", "SNOW_ATTRIBUTES", "TAVG", "TMAX", "TMAX_ATTRIBUTES", "TMIN", "WT01", "WT04", "WT06", "WT09", "weather_condition")]

#2 BORUTA
install.packages("Boruta")
library(Boruta)

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

nb_grid <- expand.grid(usekernel = c(TRUE), fL = 0:5, adjust = seq(0, 5, by = 1))
boruta_nb_model <- train(weather_condition ~ .,
                           data = boruta_train,
                           method = "nb",
                           preProcess = c("scale", "center"),
                           trControl = ctrl,
                           tuneGrid = nb_grid)
boruta_nb_model
test_pred <- predict(boruta_nb_model, newdata = boruta_test)
confusionMatrix(test_pred, boruta_test$weather_condition)

ada_grid <- expand.grid(iter = 10, maxdepth = 1:10, nu = seq(0.1, 1, by=0.1))
boruta_ada_model <- train(weather_condition ~ .,
                         data = boruta_train,
                         method = "ada",
                         preProcess = c("scale", "center"),
                         trControl = ctrl,
                         tuneGrid = ada_grid)
boruta_ada_model
test_pred <- predict(boruta_ada_model, newdata = boruta_test)
confusionMatrix(test_pred, boruta_test$weather_condition)

rpart_grid <- expand.grid(cp = seq(0.1, 1, by = 0.1))
boruta_rpart_model <- train(weather_condition ~ .,
                          data = boruta_train,
                          method = "rpart",
                          preProcess = c("scale", "center"),
                          trControl = ctrl,
                          tuneGrid = rpart_grid)
boruta_rpart_model
test_pred <- predict(boruta_rpart_model, newdata = boruta_test)
confusionMatrix(test_pred, boruta_test$weather_condition)

#3 Genetic Algorithm
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

ga_train <- train[, c("AWND", "PRCP_ATTRIBUTES", "SNOW_ATTRIBUTES", "TAVG", "TMIN", "WDF5", "WT01", "WT03", "WT04", "WT05", "WT08", "weather_condition")]
ga_test <- test[, c("AWND", "PRCP_ATTRIBUTES", "SNOW_ATTRIBUTES", "TAVG", "TMIN", "WDF5", "WT01", "WT03", "WT04", "WT05", "WT08", "weather_condition")]

#4 Simulated Annealing
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

nb_grid <- expand.grid(usekernel = c(TRUE), fL = 0:5, adjust = seq(0, 5, by = 1))
sa_nb_model <- train(weather_condition ~ .,
                         data = sa_train,
                         method = "nb",
                         preProcess = c("scale", "center"),
                         trControl = ctrl,
                         tuneGrid = nb_grid)
sa_nb_model
test_pred <- predict(sa_nb_model, newdata = sa_test)
confusionMatrix(test_pred, sa_test$weather_condition)

ada_grid <- expand.grid(iter = 10, maxdepth = 1:10, nu = seq(0.1, 1, by=0.1))
sa_ada_model <- train(weather_condition ~ .,
                          data = sa_train,
                          method = "ada",
                          preProcess = c("scale", "center"),
                          trControl = ctrl,
                          tuneGrid = ada_grid)
sa_ada_model
test_pred <- predict(sa_ada_model, newdata = sa_test)
confusionMatrix(test_pred, sa_test$weather_condition)

rpart_grid <- expand.grid(cp = seq(0.1, 1, by = 0.1))
sa_rpart_model <- train(weather_condition ~ .,
                            data = sa_train,
                            method = "rpart",
                            preProcess = c("scale", "center"),
                            trControl = ctrl,
                            tuneGrid = rpart_grid)
sa_rpart_model
test_pred <- predict(sa_rpart_model, newdata = sa_test)
confusionMatrix(test_pred, sa_test$weather_condition)

#5 Recursive Feature Elimination
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

nb_grid <- expand.grid(usekernel = c(TRUE), fL = 0:5, adjust = seq(0, 5, by = 1))
rfe_nb_model <- train(weather_condition ~ .,
                     data = rfe_train,
                     method = "nb",
                     preProcess = c("scale", "center"),
                     trControl = ctrl,
                     tuneGrid = nb_grid)
rfe_nb_model
test_pred <- predict(rfe_nb_model, newdata = rfe_test)
confusionMatrix(test_pred, rfe_test$weather_condition)

ada_grid <- expand.grid(iter = 10, maxdepth = 1:10, nu = seq(0.1, 1, by=0.1))
rfe_ada_model <- train(weather_condition ~ .,
                      data = rfe_train,
                      method = "ada",
                      preProcess = c("scale", "center"),
                      trControl = ctrl,
                      tuneGrid = ada_grid)
rfe_ada_model
test_pred <- predict(rfe_ada_model, newdata = rfe_test)
confusionMatrix(test_pred, rfe_test$weather_condition)

rpart_grid <- expand.grid(cp = seq(0.1, 1, by = 0.1))
rfe_rpart_model <- train(weather_condition ~ .,
                        data = rfe_train,
                        method = "rpart",
                        preProcess = c("scale", "center"),
                        trControl = ctrl,
                        tuneGrid = rpart_grid)
rfe_rpart_model
test_pred <- predict(rfe_rpart_model, newdata = rfe_test)
confusionMatrix(test_pred, rfe_test$weather_condition)

################################################################################