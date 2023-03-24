setwd("/Users/ayan/Desktop/BU/Spring 2023/CS699_project")

weather_data <- read.csv("weather_data.csv")
nrow(weather_data)
ncol(weather_data)

# These columns can be used for classification:
# AWND, PRCP, PRCP_ATTRIBUTES, SNOW_ATTRIBUTES, TAVG, TMAX, TMAX_ATTRIBUTES == TMIN_ATTRIBUTES,
# TMIN, WDF2, WDF5, WSF2, WSF5, WT01, WT02, WT03, WT04, WT05, WT06, WT08, WT09, DATE ???
# 
# We can potentially use the DATE column by breaking it down into: SEASON column, MONTH column.
#
# Can we take the same type of data from a different location? That way we can use location as a
# column.
# Yes, that is possible. There are several stations in Boston area.
# We can combine data from all of them for the past 5 years or so.
# This way we get new columns: LOCATION/ELEVATION, YEAR, MONTH

library(dplyr)
dplyr::count(weather_data, weather_condition, sort = TRUE)

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

weather_data_subset <- weather_data[features]

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

weather_data_subset$WT01[is.na(weather_data_subset$WT01)] <- 0
weather_data_subset$WT02[is.na(weather_data_subset$WT02)] <- 0
weather_data_subset$WT03[is.na(weather_data_subset$WT03)] <- 0
weather_data_subset$WT04[is.na(weather_data_subset$WT04)] <- 0
weather_data_subset$WT05[is.na(weather_data_subset$WT05)] <- 0
weather_data_subset$WT06[is.na(weather_data_subset$WT06)] <- 0
weather_data_subset$WT08[is.na(weather_data_subset$WT08)] <- 0
weather_data_subset$WT09[is.na(weather_data_subset$WT09)] <- 0


