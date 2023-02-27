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


