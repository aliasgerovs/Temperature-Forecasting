
library(data.table)
library(inspectdf)
library(dplyr)
library(tidymodels)
library(modeltime)
library(tidyverse)
library(lubridate)
library(timetk)
library(h2o)
library(rsample)
library(forecast)


data <- fread("daily_min_temp.csv")

data %>% glimpse()

data %>% inspect_na()

names(data)

names(data) <- data %>% names() %>% gsub(" ","_",.)

data[data$Date == "7/20/1982","Daily_minimum_temperatures"] <- "0.2"
data[data$Date == "7/21/1982","Daily_minimum_temperatures"] <- "0.8"
data[data$Date == "7/14/1984","Daily_minimum_temperatures"] <- "0.1"


data$Daily_minimum_temperatures <- data$Daily_minimum_temperatures %>% as.numeric()
data$Date <- data$Date %>% as.Date(.,"%m/%d/%Y")


data %>% inspect_na()

# 1. Build h2o::automl()
# For this task:
# prepare data using tk_augment_timeseries_signature()
# set stopping metric to “RMSE”
# set exclude_algos = c("DRF", "GBM","GLM",'XGBoost')

h2o.init(nthreads = -1, max_mem_size = '2g', ip = "127.0.0.1", port = 54321)


data <- data %>% tk_augment_timeseries_signature(Date) %>% select(Daily_minimum_temperatures,everything())

data %>% inspect_na()

data$diff %>% unique()

data$diff %>% table()

data[is.na(data$diff),]$diff <- 86400

data %>% dim()

data %>% glimpse()

data$month.lbl <- data$month.lbl %>% as.character()
data$wday.lbl <- data$wday.lbl %>% as.character()

h2o.init()
h2o_data <- data %>% as.h2o()

h2o_data <- h2o_data %>% h2o.splitFrame(ratios = 0.8, seed = 123)
train <- h2o_data[[1]]
test <- h2o_data[[2]]


target <- data[,1] %>% names()
features <- data[,-1] %>% names()

#MODEL

model <- h2o.automl(
  x = features,y = target,
  training_frame = train,
  validation_frame = test,
  leaderboard_frame = test,
  stopping_metric = "RMSE",
  exclude_algos = c("DRF", "GBM","GLM",'XGBoost'),
  nfolds = 10, seed = 123,
  max_runtime_secs = 480
)

model@leaderboard %>% as.data.frame()
model <- model@leader


# Predicting the Test set results 

y_pred <- model %>% h2o.predict(newdata = test) %>% as.data.frame()
y_pred$predict


test_set <- test %>% as.data.frame()
residuals = test_set$Daily_minimum_temperatures - y_pred$predict

# Calculate RMSE (Root Mean Square Error) ----
RMSE = sqrt(mean(residuals^2))


# Calculate Adjusted R2 (R Squared) ----
y_test_mean = mean(test_set$Daily_minimum_temperatures)

tss = sum((test_set$Daily_minimum_temperatures - y_test_mean)^2) #total sum of squares
rss = sum(residuals^2) #residual sum of squares

R2 = 1 - (rss/tss); R2

n <- test_set %>% nrow() #sample size
k <- features %>% length() #number of independent variables
Adjusted_R2 = 1-(1-R2)*((n-1)/(n-k-1))

tibble(RMSE = round(RMSE),
       R2, Adjusted_R2)



# 2. Build modeltime::arima_reg(). For this task set engine to “auto_arima”

interactive <- FALSE

data <- fread("daily_min_temp.csv")
names(data) <- data %>% names() %>% gsub(" ","_",.)

data[data$Date == "7/20/1982","Daily_minimum_temperatures"] <- "0.2"
data[data$Date == "7/21/1982","Daily_minimum_temperatures"] <- "0.8"
data[data$Date == "7/14/1984","Daily_minimum_temperatures"] <- "0.1"

data$Daily_minimum_temperatures <- data$Daily_minimum_temperatures %>% as.numeric()
data$Date <- data$Date %>% as.Date(.,"%m/%d/%Y")



splits <- initial_time_split(data, prop = 0.8)


model_fit_arima_no_boost <- arima_reg() %>%
  set_engine(engine = "auto_arima") %>%
  fit(Daily_minimum_temperatures ~ Date, data = training(splits))

model_table <- modeltime_table(model_fit_arima_no_boost)


#Calibration

calibration_tbl <- model_table %>% 
  modeltime_calibrate(new_data = testing(splits))

calibration_tbl %>%
  modeltime_forecast(
    new_data    = testing(splits),
    actual_data = data
  ) %>%
  plot_modeltime_forecast(
    .legend_max_width = 25, # For mobile screens
    .interactive = TRUE
  ) 



calibration_tbl %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = TRUE
  )


# 3. Forecast temperatures for next year with model which has lower RMSE.


#Model: auto_arima

model_fit_arima_no_boost <- arima_reg() %>%
  set_engine(engine = "auto_arima") %>%
  fit(Daily_minimum_temperatures ~ Date, data = training(splits))

#Model: arima_boost

model_fit_arima_boosted <- arima_boost(
  min_n = 2,
  learn_rate = 0.015
) %>%
  set_engine(engine = "auto_arima_xgboost") %>%
  fit(Daily_minimum_temperatures ~ Date + as.numeric(Date) + factor(month(Date, label = TRUE), ordered = F),
      data = training(splits))

#Model: ets

model_fit_ets <- exp_smoothing() %>%
  set_engine(engine = "ets") %>%
  fit(Daily_minimum_temperatures ~ Date, data = training(splits))

#Model: Prophet

model_fit_prophet <- prophet_reg() %>%
  set_engine(engine = "prophet") %>%
  fit(Daily_minimum_temperatures ~ Date, data = training(splits))

# Model 5: lm 

model_fit_lm <- linear_reg() %>%
  set_engine("lm") %>%
  fit(Daily_minimum_temperatures ~ as.numeric(Date) + factor(month(Date, label = TRUE), ordered = FALSE),
      data = training(splits))

# Model: Earth

model_spec_mars <- mars(mode = "regression") %>%
  set_engine("earth") 

recipe_spec <- recipe(Daily_minimum_temperatures ~ Date, data = training(splits)) %>%
  step_date(Date, features = "month", ordinal = FALSE) %>%
  step_mutate(date_num = as.numeric(Date)) %>%
  step_normalize(date_num) %>%
  step_rm(Date)

library(earth)


wflw_fit_mars <- workflow() %>%
  add_recipe(recipe_spec) %>%
  add_model(model_spec_mars) %>%
  fit(training(splits))

models_tbl <- modeltime_table(
  model_fit_arima_no_boost,
  model_fit_arima_boosted,
  model_fit_ets,
  model_fit_prophet,
  model_fit_lm,
  wflw_fit_mars
)

models_tbl

#Calibration
calibration_tbl <- models_tbl %>%
  modeltime_calibrate(new_data = testing(splits))

calibration_tbl

calibration_tbl %>%
  modeltime_forecast(
    new_data    = testing(splits),
    actual_data = data
  ) %>%
  plot_modeltime_forecast(
    .legend_max_width = 25, # For mobile screens
    .interactive      = TRUE
  ) 

calibration_tbl %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = TRUE
  )
# We choose LM as it is the lowest RMSE

# 3. Forecast temperatures for next year with model which has lower RMSE 

calibration_tbl <- model_fit_lm %>%
  modeltime_calibrate(new_data = testing(splits))

calibration_tbl %>%
  modeltime_forecast(h = "1 years",
                     new_data    = testing(splits),
                     actual_data = data
  ) %>%
  plot_modeltime_forecast(
    .legend_max_width = 25, # For mobile screens
    .interactive      = TRUE
  ) 




#Github

