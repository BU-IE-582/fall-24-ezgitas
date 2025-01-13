library(dplyr)
library(data.table)

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

test_data <- match_data %>% filter(match_date >= as.Date("2024-11-01"))

# Extract the last observation for each fixture_id
test_data_last <- test_data %>%
  group_by(fixture_id) %>%  # Group by fixture_id
  slice_tail(n = 1) %>%     # Select the last row of each group
  ungroup()                 # Ungroup to return a regular data frame

test_data_last <- test_data_last %>%
  mutate(prediction = current_state,
         accuracy = ifelse(prediction == result, 1, 0),
         return = accuracy * case_when(
           prediction == 1 ~ `1`,  # Use value in column '1' if prediction is 1
           prediction == 2 ~ `2`,  # Use value in column '2' if prediction is 2
           prediction == "X" ~ `X` # Use value in column 'X' if prediction is "X"
         )
         )

test_data_last_summary <- test_data_last[,c('fixture_id','halftime','minute','second',
                                    'time_in_match','1','X','2','result',
                                    'prediction','accuracy','return')]

# Calculate the sum of the return column
total_return <- sum(test_data_last$return, na.rm = TRUE)

# Print the result
print(total_return)

bet_accuracy<-sum(test_data_last$accuracy)/nrow(test_data_last)

# Print the result
print(bet_accuracy)
