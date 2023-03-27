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

# train-test split
library(rsample)
set.seed(17)
split <- initial_split(weather_data_subset, prop = 0.70, strata = weather_condition)
train <- training(split)
test <- testing(split)

# Now, we need to use the 5 feature selection methods on the train and test datasets.

# feature selection
#1 Information Gain
# selected features: PRCP, PRCP_ATTRIBUTES, SNOW_ATTRIBUTES, TAVG, TMAX, TMAX_ATTRIBUTES, TMIN, WT01, WT04, WT06, WT09
install.packages("FSelector")
library(FSelector) #requires JAVA; run in cloud if necessary
weights <- information.gain(weather_condition~., train)
weights

new_weights <- subset(weights, attr_importance > 0)
new_weights

ig_train <- train[, c("PRCP", "PRCP_ATTRIBUTES", "SNOW_ATTRIBUTES", "TAVG", "TMAX", "TMAX_ATTRIBUTES", "TMIN", "WT01", "WT04", "WT06", "WT09", "weather_condition")]
ig_test <- test[, c("PRCP", "PRCP_ATTRIBUTES", "SNOW_ATTRIBUTES", "TAVG", "TMAX", "TMAX_ATTRIBUTES", "TMIN", "WT01", "WT04", "WT06", "WT09", "weather_condition")]

#2 BORUTA
# selected features: PRCP, TMAX, WT01, TAVG, TMIN, SNOW_ATTRIBUTES, WT09, WT04, WDF5, WT08, WT02, WDF2, WT06, WSF5, WSF2, AWND
install.packages("Boruta")
library(Boruta)

boruta_output <- Boruta(weather_condition ~ ., data=train, doTrace=0)
boruta_signif <- getSelectedAttributes(boruta_output, withTentative = TRUE)
print(boruta_signif)  

roughFixMod <- TentativeRoughFix(boruta_output)
boruta_signif <- getSelectedAttributes(roughFixMod)
print(boruta_signif)

imps <- attStats(roughFixMod)
imps2 <- imps[imps$decision != 'Rejected', c('meanImp', 'decision')]
head(imps2[order(-imps2$meanImp), ])
plot(boruta_output, cex.axis=.7, las=2, xlab="", main="Variable Importance")

boruta_train <- train[, c("PRCP", "TMAX", "WT01", "TAVG", "TMIN", "SNOW_ATTRIBUTES", "WT09", "WT04", "WDF5", "WT08", "WT02", "WDF2", "WT06", "WSF5", "WSF2", "AWND", "weather_condition")]
boruta_test <- test[, c("PRCP", "TMAX", "WT01", "TAVG", "TMIN", "SNOW_ATTRIBUTES", "WT09", "WT04", "WDF5", "WT08", "WT02", "WDF2", "WT06", "WSF5", "WSF2", "AWND", "weather_condition")]

#3 Genetic Algorithm
# selected features: AWND, PRCP_ATTRIBUTES, SNOW_ATTRIBUTES, TAVG, TMIN, WDF5, WT01, WT03, WT04, WT05, WT08
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
# selected features: TAVG, WSF5, WT02, WT04, WT05, WT06, WT09
sa_ctrl <- safsControl(functions = rfSA,
                       method = "repeatedcv",
                       repeats = 3,
                       improve = 5) # n iterations without improvement before a reset

sa_obj <- safs(x=train[, -ncol(train)],
               y=train$weather_condition,
               safsControl = sa_ctrl)

sa_obj
sa_obj$optVariables

sa_train <- train[, c("TAVG", "WSF5", "WT02", "WT04", "WT05", "WT06", "WT09", "weather_condition")]
sa_test <- test[, c("TAVG", "WSF5", "WT02", "WT04", "WT05", "WT06", "WT09", "weather_condition")]

#8 Recursive Feature Elimination
# selected features: PRCP, TMAX, WT01, TMIN, TAVG, SNOW_ATTRIBUTES, WT09, WT04, WT08, WDF5
subsets <- c(1:5, 10, 15, 20)

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

rfe_train <- train[, c("PRCP", "TMAX", "WT01", "TMIN", "TAVG", "SNOW_ATTRIBUTES", "WT09", "WT04", "WT08", "WDF5", "weather_condition")]
rfe_test <- test[, c("PRCP", "TMAX", "WT01", "TMIN", "TAVG", "SNOW_ATTRIBUTES", "WT09", "WT04", "WT08", "WDF5", "weather_condition")]










