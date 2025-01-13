# Load necessary libraries
library(xgboost)
library(Matrix)
library(dplyr)
library(data.table)
library(randomForest)
library(caret)

set.seed(1234)  # Set the seed

data_path='C:/Users/asuna/OneDrive/Belgeler/match_data.csv'
match_data <- fread(data_path)
match_data <- match_data %>%
  filter(suspended != "TRUE" & stopped != "TRUE")

# Ensure halftime data is considered when ordering
match_data <- match_data %>%
  mutate(half_time_order = ifelse(halftime == "1st-half", 1, 2)) %>%
  arrange(fixture_id, half_time_order, minute, second)

# Define train-test split based on match date
match_data <- match_data %>% mutate(match_date = as.Date(match_start_datetime))
match_data <- match_data %>%
  mutate(time_in_match = ifelse(halftime == "1st-half", minute + second / 60,
                           45 + minute + second / 60))
train_data <- match_data %>% filter(match_date < as.Date("2024-11-01"))
test_data <- match_data %>% filter(match_date >= as.Date("2024-11-01"))

# Define features (X) and target (y)
X_train <- train_data %>%
  select(
    `1`, `2`, `X`, `minute`, `second`,`time_in_match`,
    `Shots On Target - home`, `Shots On Target - away`,
    `Saves - away`, `Saves - home`, `Goal Attempts - home`, `Goal Attempts - away`,
    `Shots Total - home`, `Shots Total - away`, `Dangerous Attacks - home`,
    `Dangerous Attacks - away`, `Shots Insidebox - home`, `Shots Insidebox - away`,
    `Redcards - away`, `Redcards - home`, `Key Passes - away`, `Key Passes - home`
  ) %>%
  as.matrix()
y_train <- train_data$result

X_test <- test_data %>%
  select(
    `1`, `2`, `X`, `minute`, `second`,`time_in_match`,
    `Shots On Target - home`, `Shots On Target - away`,
    `Saves - away`, `Saves - home`, `Goal Attempts - home`, `Goal Attempts - away`,
    `Shots Total - home`, `Shots Total - away`, `Dangerous Attacks - home`,
    `Dangerous Attacks - away`, `Shots Insidebox - home`, `Shots Insidebox - away`,
    `Redcards - away`, `Redcards - home`, `Key Passes - away`, `Key Passes - home`
  ) %>%
  as.matrix()
y_test <- test_data$result

# Replace 'X' with 0
y_train[y_train == "X"] <- 0
y_test[y_test == "X"] <- 0

# Convert y_train to numeric (if necessary, e.g., if it's a factor or character vector)
y_train <- as.numeric(as.character(y_train))
y_test <- as.numeric(as.character(y_test))


# Create XGBoost DMatrices for training and testing
dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest <- xgb.DMatrix(data = X_test, label = y_test)


#Training
# Define XGBoost parameters
params <- list(
  objective = "multi:softprob",  # Multiclass classification with probabilities
  num_class = 3,                # Number of classes (1: Home Win, 0: Draw, 2: Away Win)
  eval_metric = "mlogloss",
  subsample = 0.8,
  colsample_bytree = 0.8,       # Feature subsampling per tree (80% of columns)
  colsample_bylevel = 0.8,      # Feature subsampling per level (80% of columns per level)
  max_depth = 8,  # Allow deeper trees
  min_child_weight = 0.1,  # More sensitivity to smaller groups
  lambda = 1,                   # L2 regularization
  eta = 0.5                     # Learning rate
)

# Create the group variable (fixture_id) from the original training data
groups <- train_data$fixture_id

# Ensure that groups match the rows of the DMatrix
if (length(groups) != nrow(X_train)) {
  stop("Mismatch between group length and training data rows")
}

# Run cross-validation with grouping
xgb_cv <- xgb.cv(
  params = params,
  data = dtrain,
  nfold = 5,
  nrounds = 300,
  early_stopping_rounds = 10,
  verbose = 0,
  folds = createFolds(groups, k = 5)  # Create grouped folds
)

# Extract the best number of rounds (iterations)
best_nrounds <- xgb_cv$best_iteration

# Train the final model using the best number of rounds
model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = best_nrounds,
  verbose = 1
)

# Predict probabilities on train data
pred_probs_train <- predict(model, newdata = dtrain)
pred_probs_train <- matrix(pred_probs_train, ncol = 3, byrow = TRUE)  # Reshape into a probability matrix

train_data <- train_data %>%
  mutate(
    P_home = pred_probs_train[, 2],
    P_draw = pred_probs_train[, 1],
    P_away = pred_probs_train[, 3],
    
    ev_home = pred_probs_train[, 2] * (`1` - 1) - (1 - pred_probs_train[, 2]),
    ev_draw = pred_probs_train[, 1] * (`X` - 1) - (1 - pred_probs_train[, 1]),
    ev_away = pred_probs_train[, 3] * (`2` - 1) - (1 - pred_probs_train[, 3]),
    
    decision = case_when(
      ev_home > ev_draw & ev_home > ev_away & ev_home > 0 ~ "bet_home",
      ev_draw > ev_home & ev_draw > ev_away & ev_draw > 0 ~ "bet_draw",
      ev_away > ev_home & ev_away > ev_draw & ev_away > 0 ~ "bet_away",
      TRUE ~ "no_action"),
    
    prediction = case_when(
      decision == "bet_home" ~ "1",
      decision == "bet_away" ~ "2",
      decision == "bet_draw" ~ "X",
      decision == "no_action" ~ "0",
      TRUE ~ NA  # Optional: Handle unexpected values with NA
      ),
      
      accuracy = ifelse(prediction == as.character(result), 1, 0),
      
      return = case_when(
        # If accuracy is 1, take the odd of the prediction
        accuracy == 1 & prediction == "1" ~ as.numeric(`1`),
        accuracy == 1 & prediction == "2" ~ as.numeric(`2`),
        accuracy == 1 & prediction == "X" ~ as.numeric(`X`),
        
        # If accuracy is 0 and prediction is 0, return 0
        accuracy == 0 & prediction == "0" ~ 0,
        
        # If accuracy is 0 and prediction is 1, 2, or X, return 0
        accuracy == 0 & prediction %in% c("1", "2", "X") ~ 0,
        
        # Default case (if none of the above conditions apply)
        TRUE ~ NA_real_
      )
  )

    
train_data_summary <- train_data[,c('fixture_id','halftime','minute','second','time_in_match','1','X','2','P_home','P_draw','P_away',
                                        'ev_home','ev_draw','ev_away','result','prediction','accuracy','return')]
    
# Train a Random Forest Model
rf_model <- randomForest(
  x = train_data_summary[, c("ev_home", "ev_draw", "ev_away", "time_in_match",'halftime','minute','second')],  # Features
  y = as.factor(train_data_summary$accuracy),  # Target (convert to factor)
  ntree = 100,  # Number of trees
  mtry = 2,     # Number of features to consider at each split
  importance = TRUE
)

# Find the Optimal Threshold Based on Training Data
thresholds <- seq(0, 1, by = 0.01)

# Calculate accuracy for each threshold on the training data
train_rf_probs <- predict(rf_model, newdata = train_data_summary[, c("ev_home", "ev_draw", "ev_away","time_in_match",'halftime','minute','second')], type = "prob")
train_data_summary$success_prob <- train_rf_probs[, 2]  # Probability for success (class 1)

accuracies <- sapply(thresholds, function(threshold) {
  predicted <- ifelse(train_data_summary$success_prob >= threshold, 1, 0)
  mean(predicted == train_data_summary$accuracy)
})

optimal_threshold <- thresholds[which.max(accuracies)]
print(paste("Final Accuracy on Training Data:", max(accuracies)))
print(paste("Optimal Threshold from Training Data:", optimal_threshold))

#Testing
# Predict probabilities on test data
pred_probs_test <- predict(model, newdata = dtest)
pred_probs_test <- matrix(pred_probs_test, ncol = 3, byrow = TRUE)  # Reshape into a probability matrix

test_data <- test_data %>%
  mutate(
    P_home = pred_probs_test[, 2],
    P_draw = pred_probs_test[, 1],
    P_away = pred_probs_test[, 3],
    
    ev_home = pred_probs_test[, 2] * (`1` - 1) - (1 - pred_probs_test[, 2]),
    ev_draw = pred_probs_test[, 1] * (`X` - 1) - (1 - pred_probs_test[, 1]),
    ev_away = pred_probs_test[, 3] * (`2` - 1) - (1 - pred_probs_test[, 3]),
    
    decision = case_when(
      ev_home > ev_draw & ev_home > ev_away & ev_home > 0 ~ "bet_home",
      ev_draw > ev_home & ev_draw > ev_away & ev_draw > 0 ~ "bet_draw",
      ev_away > ev_home & ev_away > ev_draw & ev_away > 0 ~ "bet_away",
      TRUE ~ "no_action"),
    
    prediction = case_when(
      decision == "bet_home" ~ "1",
      decision == "bet_away" ~ "2",
      decision == "bet_draw" ~ "X",
      decision == "no_action" ~ "0",
      TRUE ~ NA  # Optional: Handle unexpected values with NA
    ),
    
    accuracy = ifelse(prediction == as.character(result), 1, 0),
    
    return = case_when(
      # If accuracy is 1, take the odd of the prediction
      accuracy == 1 & prediction == "1" ~ as.numeric(`1`),
      accuracy == 1 & prediction == "2" ~ as.numeric(`2`),
      accuracy == 1 & prediction == "X" ~ as.numeric(`X`),
      
      # If accuracy is 0 and prediction is 0, return 0
      accuracy == 0 & prediction == "0" ~ 0,
      
      # If accuracy is 0 and prediction is 1, 2, or X, return 0
      accuracy == 0 & prediction %in% c("1", "2", "X") ~ 0,
      
      # Default case (if none of the above conditions apply)
      TRUE ~ NA_real_
    )
  )

test_data_summary <- test_data[,c('fixture_id','halftime','minute','second','time_in_match','1','X','2','P_home','P_draw','P_away',
                                    'ev_home','ev_draw','ev_away','result','prediction','accuracy','return')]

# Predict probabilities on test data
rf_probs <- predict(rf_model, newdata = test_data_summary[, c("ev_home", "ev_draw", "ev_away", "time_in_match",'halftime','minute','second')], type = "prob")
test_data_summary$success_prob <- rf_probs[, 2]  # Probability for success (class 1)

# Apply the Optimal Threshold to Classify Test Data
test_data_summary <- test_data_summary %>%
  mutate(predicted_success = ifelse(success_prob >= optimal_threshold, 1, 0))

# Add decision_time column
test_data_summary <- test_data_summary %>%
  mutate(
    decision_time = ifelse(predicted_success == 1, TRUE, FALSE),
  )

# Extract the first decision time for each match
decision_points <- test_data_summary %>%
  filter(decision_time == TRUE) %>%  # Filter rows where decision_time is TRUE
  group_by(fixture_id) %>%          # Group by match (fixture_id)
  slice_min(order_by = time_in_match, n = 1, with_ties = FALSE) %>%  # Find the earliest time
  select(fixture_id, halftime, minute, second,`1`,`X`,`2`, success_prob, time_in_match,result,prediction,
         accuracy,return)

# Count unique matches using fixture_id
total_num_matches_test <- length(unique(test_data_summary$fixture_id))
num_matches_bet <- length(unique(decision_points$fixture_id))
# Print the result
print(paste("Number of matches in test data:", total_num_matches_test))
print(paste("Number of matches that are bet in test data:", num_matches_bet))


mean_accuracy = mean(decision_points$accuracy)
print(paste("Accuracy in test data:", mean_accuracy))
sum_return <- sum(decision_points$return)
print(paste("Return for decided matches in test data:", sum_return))
