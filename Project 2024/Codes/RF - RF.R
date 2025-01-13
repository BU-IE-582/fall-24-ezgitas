# Load necessary libraries
library(xgboost)
library(Matrix)
library(dplyr)
library(data.table)
library(randomForest)
library(caret)

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
    `1`, `2`, `X`
  ) %>%
  as.matrix()
y_train <- train_data$result

X_test <- test_data %>%
  select(
    `1`, `2`, `X`
  ) %>%
  as.matrix()
y_test <- test_data$result

set.seed(1234)  # Set the seed
# Replace 'X' with 0
y_train[y_train == "X"] <- 0
y_test[y_test == "X"] <- 0

model <- randomForest(
  x = as.data.frame(X_train),
  y = as.factor(y_train),
  ntree = 200,      # Number of trees
  mtry = 3,         # Number of features considered at each split
  importance = TRUE
)

# Predict probabilities on train data
pred_probs_train <- predict(model, newdata = as.data.frame(X_train), type = "prob")

train_data <- train_data %>%
  mutate(
    P_home = pred_probs_train[, "1"], 
    P_draw = pred_probs_train[, "0"],
    P_away = pred_probs_train[, "2"],
    
    ev_home = pred_probs_train[, "1"] * (`1` - 1) - (1 - pred_probs_train[, "1"]),
    ev_draw = pred_probs_train[, "0"] * (`X` - 1) - (1 - pred_probs_train[, "0"]),
    ev_away = pred_probs_train[, "2"] * (`2` - 1) - (1 - pred_probs_train[, "2"]),
    
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
pred_probs_test <- predict(model, newdata = as.data.frame(X_test), type = "prob")


test_data <- test_data %>%
  mutate(
    P_home = pred_probs_test[, "1"],
    P_draw = pred_probs_test[, "0"],
    P_away = pred_probs_test[, "2"],
    
    ev_home = pred_probs_test[, "1"] * (`1` - 1) - (1 - pred_probs_test[, "1"]),
    ev_draw = pred_probs_test[, "0"] * (`X` - 1) - (1 - pred_probs_test[, "0"]),
    ev_away = pred_probs_test[, "2"] * (`2` - 1) - (1 - pred_probs_test[, "2"]),
    
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

