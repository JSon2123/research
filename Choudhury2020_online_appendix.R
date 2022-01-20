
## use ## to specify my annotations

## https://www.tidymodels.org/



## 1. Import Libraries and Data

#Libraries
library(tidymodels)
library(tidyverse)
library(pdp)

#Figure size
options(repr.plot.width=8, repr.plot.height=8)

#No warnings
options(warn = -1)

#Set random number selection
set.seed(123)

#Set cores for parallel processing on Kaggle
all_cores <- parallel::detectCores(logical = FALSE)
library(doParallel)
cl <- makePSOCKcluster(all_cores)
registerDoParallel(cl)



## df = read.csv("../input/simulated-data-for-ml-paper/simulated_TECHCO_data.csv")

df = read.csv("./data/simulated_TECHCO_data.csv")




# 2. Preprocess and Partition Data


set.seed(123)

# Make a custom function that splits data into training and test set 
# while respecting emp_id
# If grouping doesn't matter, 
# you can just use the tidymodels `initial_split()` default function

## make a custom function named "initial_split_id"

initial_split_id <- function(data,group,prop = 3/4) {
  group_vfold_cv(data,{{group}},1/(1-prop)) %>%
    head(1) %>%
    pull(1) %>%
    pluck(1)
}

?group_vfold_cv ## group_vfold_cv {rsample}

df_split<-initial_split_id(df,emp_id,prop=0.7)

df_train<-training(df_split)
## dim(df_train)

df_test<-testing(df_split)
## dim(df_test)

# Set parameters that will apply to all the ML algorithms 
# (CV folds, metrics, preprocessing, etc.)

set.seed(123)

#Define the cross-validation folds (10 folds, grouped by emp_id)
folds <- group_vfold_cv(df_train, group='emp_id', v=10)

#Set the metrics by which we will evaluate the model
metrics <- metric_set(mn_log_loss,roc_auc,accuracy)

#Set other control parameters. Here we just turn off verbose model training
ctrl <- control_grid(verbose = FALSE)

#Set preprocessing and model. For the decision tree and neural network, we do minimal pre-processing. We just remove emp_id as a variable. We will add some scaling for the neural network. 
preprocess_recipe <-df_train%>%
  recipe(turnover ~ .)%>%
  step_rm(emp_id)



# 3. Algorithm Implementation and Results

# Decision Tree

?decision_tree #
decision_tree {parsnip}

#Set decision tree model, including the hyperparameters that will be tuned
dt_model <-
  decision_tree(mode="classification",cost_complexity = tune(), tree_depth = tune(), min_n = tune()) %>%
  set_mode("classification") %>%
  set_engine("rpart",parms = list(split = "information"))

#Set the workflow, adding the steps of the preprocessing recipe and the model
dt_wflow<-
  workflow()%>%
  add_recipe(preprocess_recipe)%>%
  add_model(dt_model)

#Set the bounds on the hyperparameters
dt_param <-
  dt_wflow %>%
  parameters() #%>%

#It is possible to update the parameter ranges here
# update(tree_depth = tree_depth(range = c(1,20)),
#   min_n = min_n(range = c(1,100)))

#Create a grid  of 100 random combinations of values for the hyperparameters to be tuned.
dt_grid <- grid_random(dt_param, size = 50)



#Search the hyperparameter space set by the grid (i.e. try fitting a model for the 100 combinations of hyperparameters)
search_dt <-
  tune_grid(
    dt_wflow,
    grid = dt_grid,
    param_info = dt_param,
    resamples = folds,
    metrics = metrics,
    control = ctrl
  )


# Plot the log loss for each hyperparameter 
# to get a sense for the right choice of each



autoplot(search_dt, metric = "mn_log_loss")+theme_classic()

