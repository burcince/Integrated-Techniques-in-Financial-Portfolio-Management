## INSTALL AND LOAD PACKAGES ##
install.packages('forecast')
install.packages('rpart')
install.packages('rpart.plot')
install.packages('RColorBrewer')
install.packages('rattle')
install.packages('zoo')
install.packages('fpp2')
install.packages('tidyverse')
install.packages('dplyr')
install.packages('caret')
install.packages('neuralnet')
install.packages('NeuralNetTools')
install.packages('randomForest')
library(forecast)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(zoo)
library(fpp2)
library(tidyverse)
library(dplyr)
library(caret)
library(neuralnet)
library(nnet)
library(NeuralNetTools)
library(MASS)
library(readxl)
library(dplyr)
library(ggplot2)
library(cluster)
library(caret)
library(randomForest)
library(nnet)


### LOAD DATA ###
# Load the data that was preprocessed partially through 'Python'
file_path <- "/Users/burcince/Desktop/Dissertation Data Sheets/Final_Data_Frame.xlsx"
df <- read_excel(file_path)
# Ensure 'Date' column is in Date format, so that it can be used later as an index of the time-series
df$Date <- as.Date(df$Date)
# View the data for verification
head(df)

### CORRELATION ANALYSIS ###
# Selecting numeric columns to conduct correlation analysis
numeric_columns <- df %>% select_if(is.numeric)
# Calculation of correlation matrix
correlation_matrix <- cor(numeric_columns, use = "complete.obs")
# Print the matrix itself for analysis
print(correlation_matrix)
# Plot the correlation matrix with a heat-map format
library(corrplot)
corrplot(correlation_matrix, method = "color", tl.col = "black", tl.cex = 0.7)
# Exporting the matrix to a CSV file, so that there would be no need to calculate it again
write.csv(correlation_matrix, "correlation_matrix.csv", row.names = TRUE)


### PCA ANALYSIS WITH CLUSTERING ###
library(readxl)
library(dplyr)
library(ggplot2)
library(cluster)
library(factoextra)
# Standardization
scaled_data <- scale(numeric_columns)
# Performing PCA Analysis
pca_result <- prcomp(scaled_data, scale = TRUE)
# Getting Principal Components 
pca_data <- as.data.frame(pca_result$x)
# Using Elbow Method to determine optimal number of clusters to attain the principal components
wss <- (nrow(pca_data)-1)*sum(apply(pca_data, 2, var))
for (i in 2:15) wss[i] <- sum(kmeans(pca_data, centers = i)$withinss)
plot(1:15, wss, type = "b", xlab = "Number of Clusters", ylab = "Within groups sum of squares")
# Performin K-Means clustering with the determined number of clusters, using the principal components
set.seed(123)
k <- 8 #Recall Elbow Method
kmeans_result <- kmeans(pca_data, centers = k, nstart = 25)
# Adding the assignments to the original df
df$Cluster <- as.factor(kmeans_result$cluster)
# Plotting clusters based on first 2 Principal Components
ggplot(pca_data, aes(x = PC1, y = PC2, color = df$Cluster)) +
  geom_point() +
  labs(title = "K-Means Clustering", x = "PC1", y = "PC2") +
  theme_minimal()
kmeans_result$centers
summary(pca_result)
# Getting and analyzing the loadings (coefficiencts of columns within each principal component)
loadings <- pca_result$rotation
print(loadings)
# Understanding portion of variances each principal components is responsible for
explained_variance <- pca_result$sdev^2 / sum(pca_result$sdev^2)
cumulative_variance <- cumsum(explained_variance)
print(explained_variance)
print(cumulative_variance)
# Plot the Variance-Principal Component relationships
plot(explained_variance, type = 'b', xlab = 'Principal Component', ylab = 'Proportion of Variance Explained')
plot(cumulative_variance, type = 'b', xlab = 'Principal Component', ylab = 'Cumulative Proportion of Variance Explained')

### K-MEANS CLUSTERING ON THE DATA FOR EQUITY LABELLING AND CLUSTER MEAN ANALYSIS ###
# Performing k-means clustering with the number of clusters chosen with the Elbow Method
set.seed(123)
k <- 8
kmeans_result <- kmeans(scaled_data[,3:31], centers = k, nstart = 25) #data without date and equity in order to ensure non-biased clusters
# Merging cluster results with the dataframe
df$Cluster <- as.factor(kmeans_result$cluster)
# Analyzing the cluster means for understanding how metrics affect return 
library(cluster)
library(factoextra)
# Calculation of the original means 
cluster_means <- df %>%
  group_by(Cluster) %>%
  summarise_all(mean, na.rm = TRUE)
print(cluster_means)
# Exporting the results for computational efficiency
write.csv(cluster_means, "cluster_means.csv", row.names = TRUE)

### LOGISTICS REGRESSION ###

# To ensure consistency among equities and prevent equity bias, partition is made with respect to Date
set.seed(123)
df <- df %>% arrange(Week_Index)
split_point <- round(nrow(df) * 0.8)
train_data <- df[1:split_point, ]
test_data <- df[(split_point + 1):nrow(df), ]
# Veryfring there is no missing data
train_data <- na.omit(train_data)
test_data <- na.omit(test_data)
# Logistic Regression for Short-Term Return
initial_model_short_binary <- glm(Short_Term_Positive_Binary ~ PX_Last + VOLUME + PE_RATIO + Dividend_Yield + 
                       Price2Sales + Price2CashF + EPS + ROE + Debt2Equity + Current_Ratio + 
                       Quick_Ratio + Gross_Margin + Operating_Margin + Net_Profit_Mar + EBITDA + 
                       Free_CF + MA_7 + MA_28 + RSI, 
                     data = train_data, family = binomial)
# Performing backward elimination for getting the best performing model
# and eliminating non-significant independent variables
short_binary <- step(initial_model_short_binary, direction = "backward")
# Summarizing the final model
summary(short_binary)
# Performance evaluation of validation data
log_predictions <- predict(short_binary, test_data, type = "response")
log_pred_class <- ifelse(log_predictions > 0.5, 1, 0)
log_confusion <- confusionMatrix(as.factor(log_pred_class), as.factor(test_data$Short_Term_Positive_Binary))
print(log_confusion)

# Logistic Regression for Mid-Term Return
initial_model_mid_binary <- glm(Mid_Term_Positive_Binary ~ PX_Last + VOLUME + PE_RATIO + Dividend_Yield + 
                                    Price2Sales + Price2CashF + EPS + ROE + Debt2Equity + Current_Ratio + 
                                    Quick_Ratio + Gross_Margin + Operating_Margin + Net_Profit_Mar + EBITDA + 
                                    Free_CF + MA_7 + MA_28 + RSI, 
                                  data = train_data, family = binomial)
mid_binary <- step(initial_model_mid_binary, direction = "backward")
summary(mid_binary)
log_predictions <- predict(mid_binary, test_data, type = "response")
roc_curve_mid <- roc(test_data$Mid_Term_Positive_Binary, log_predictions)
plot(roc_curve_mid)
optimal_threshold_mid <- coords(roc_curve_mid, "best", ret = "threshold")
log_pred_class <- ifelse(log_predictions > 0.6063038, 1, 0)
log_confusion <- confusionMatrix(as.factor(log_pred_class), as.factor(test_data$Mid_Term_Positive_Binary))
print(log_confusion)

# Logistic Regression for Long-Term Return
initial_model_long_binary <- glm(Long_Term_Positive_Binary ~ PX_Last + VOLUME + PE_RATIO + Dividend_Yield + 
                                  Price2Sales + Price2CashF + EPS + ROE + Debt2Equity + Current_Ratio + 
                                  Quick_Ratio + Gross_Margin + Operating_Margin + Net_Profit_Mar + EBITDA + 
                                  Free_CF + MA_7 + MA_28 + RSI, 
                                data = train_data, family = binomial)
Long_binary <- step(initial_model_long_binary, direction = "backward")
summary(Long_binary)
log_predictions <- predict(Long_binary, test_data, type = "response")
log_pred_class <- ifelse(log_predictions > 0.5, 1, 0)
log_confusion <- confusionMatrix(as.factor(log_pred_class), as.factor(test_data$Long_Term_Positive_Binary))
print(log_confusion)

### NEURAL NETWORKS FOR PREDICTION OF RETURNS ###

install.packages("neuralnet")
install.packages("caret")
install.packages("dplyr")
library(neuralnet)
library(caret)
library(dplyr)

# Standardization of numeric values 
preProcValues <- preProcess(train_data, method = c("center", "scale"))
train_data <- predict(preProcValues, train_data)
test_data <- predict(preProcValues, test_data)
# Factorizing response variables
train_data$Short_Term_Positive_Binary <- as.numeric(as.character(train_data$Short_Term_Positive_Binary))
train_data$Mid_Term_Positive_Binary <- as.numeric(as.character(train_data$Mid_Term_Positive_Binary))
train_data$Long_Term_Positive_Binary <- as.numeric(as.character(train_data$Long_Term_Positive_Binary))
test_data$Short_Term_Positive_Binary <- as.numeric(as.character(test_data$Short_Term_Positive_Binary))
test_data$Mid_Term_Positive_Binary <- as.numeric(as.character(test_data$Mid_Term_Positive_Binary))
test_data$Long_Term_Positive_Binary <- as.numeric(as.character(test_data$Long_Term_Positive_Binary))
# Defining significant metrics for each term of return, based on previous analysis
columns_to_include_short <- c(
  "PX_Last", "VOLUME", "Dividend_Yield", "EPS", 
  "ROE", "Current_Ratio", 
  "Operating_Margin", "EBITDA", "Free_CF", "MA_7", 
  "MA_28", "RSI"
)
# Short-term return prediction model
formula_short <- as.formula(paste("Short_Term_Positive_Binary ~", paste(columns_to_include_short, collapse = " + ")))
nn_short <- neuralnet(formula_short, data = train_data, hidden = 5, linear.output = FALSE, threshold = 0.1, stepmax = 1e6, lifesign = 'minimal')

columns_to_include_mid <- c(
  "PX_Last", "VOLUME", "Dividend_Yield", "Price2Sales", "Price2CashF", "EPS", 
  "ROE", "Debt2Equity", "Current_Ratio", "Quick_Ratio", "Gross_Margin", 
  "Operating_Margin", "Net_Profit_Mar", "EBITDA", "Free_CF", "MA_7", 
  "MA_28", "RSI"
)
# Mid-term return prediction model
formula_mid <- as.formula(paste("Mid_Term_Positive_Binary ~", paste(columns_to_include_mid, collapse = " + ")))
nn_model_mid <- neuralnet(formula_mid, data = train_data, hidden = c(5,5), linear.output = FALSE, stepmax = 1e10)

columns_to_include_long <- c(
  "PX_Last", "VOLUME", "Dividend_Yield", "Price2Sales", "Price2CashF", "EPS", 
  "ROE", "Debt2Equity", "Current_Ratio", "Quick_Ratio", "Gross_Margin", 
  "Operating_Margin", "Net_Profit_Mar", "EBITDA", "Free_CF", "MA_7", 
  "MA_28", "RSI"
)
# Long-term return prediction model
formula_long <- as.formula(paste("Long_Term_Positive_Binary ~", paste(columns_to_include_long, collapse = " + ")))
nn_model_long <- neuralnet(formula_long, data = train_data, hidden = c(5,5), linear.output = FALSE, stepmax = 1e10)
# Computation of predictions for each term
predictions_short <- compute(nn_short, test_data[, columns_to_include])$net.result
predictions_mid <- compute(nn_model_mid, test_data[, columns_to_include])$net.result
predictions_long <- compute(nn_model_long, test_data[, columns_to_include])$net.result
library(tidyverse)
library(pROC)
# Calculate the ROC curve for Short Term
roc_curve_short <- roc(test_data$Short_Term_Positive_Binary, predictions_short[,1])
plot(roc_curve_short)
optimal_threshold_short <- coords(roc_curve_short, "best", ret = "threshold")
# Calculate the ROC curve for Mid Term
roc_curve_mid <- roc(test_data$Mid_Term_Positive_Binary, predictions_short[,1])
plot(roc_curve_mid)
optimal_threshold_mid <- coords(roc_curve, "best", ret = "threshold")
# Calculate the ROC curve for Long Term
roc_curve_long <- roc(test_data$Long_Term_Positive_Binary, predictions_short[,1])
plot(roc_curve_long)
optimal_threshold_long <- coords(roc_curve, "best", ret = "threshold")
# Converting probabilities to binary outcomes
# Short-Term
predictions_short<-as.data.frame(predictions_short)
predictions_short_binary <- ifelse(predictions_short > 5.109455e-85, 1, 0)
test_predictions_short_binary <- ifelse(test_data$Short_Term_Positive_Binary > 0, 1, 0)
# Mid-Term
predictions_mid_binary <- ifelse(predictions_mid > optimal_threshold_mid, 1, 0)
test_predictions_short_binary <- ifelse(test_data$Mid_Term_Positive_Binary > 0, 1, 0)
# Long-Term
predictions_long_binary <- ifelse(predictions_long > optimal_threshold_long, 1, 0)
test_predictions_short_binary <- ifelse(test_data$Long_Term_Positive_Binary > 0, 1, 0)
# Evaluating performance based on predictions made
confusionMatrix(as.factor(predictions_short_binary), as.factor(test_predictions_short_binary))
confusionMatrix(as.factor(predictions_mid_binary), as.factor(test_data$Mid_Term_Positive_Binary))
confusionMatrix(as.factor(predictions_long_binary), as.factor(test_data$Long_Term_Positive_Binary))

### MONTE CARLO SIMULATION FOR INCLUDING ADVANCED RISK MEASURES ###

library(dplyr)
library(tidyr)
library(caret)
df <- read_excel(file_path)
# Data Cleaning and Pre-Processing
df <- df %>%
  mutate(
    DATE = as.Date(Date, format="%Y-%m-%d"),
    WEEK_INDEX = as.numeric(Week_Index),
    QUARTER_INDEX = as.numeric(Quarter_Index)
  )
# Normalize or Standardize Data
mean = mean(df$RETURN) # "RETURN" is chosen for short term return, as it is weekly and the short term is defined as 3 weeks
sd = sd(df$RETURN)
preProcess_values <- preProcess(df, method = c("center", "scale"))
df <- predict(preProcess_values, df)
library(PerformanceAnalytics)
# Define Parameters
num_simulations <- 1000
simulation_results <- data.frame() #initialization of results data-frame
initial_portfolio_value<-1000 # to be used later to calucalte money lost/gained
# Run Monte Carlo Simulations
for (i in 1:num_simulations) {
  simulated_returns <- rnorm(n = nrow(df), mean = mean, sd = sd)
  simulation_results <- rbind(simulation_results, simulated_returns)
}
# Analyzing Simulation Results
simulated_risks <- apply(simulation_results, 2, function(x) quantile(x, probs = c(0.05, 0.95)))
summary_simulation_results <- summary(simulation_results)
print(summary_simulation_results)
# Plotting the results 
library(ggplot2)
simulation_df <- as.data.frame(simulation_results)
simulation_df <- (simulation_results)*100
ggplot(simulation_df, aes(x = seq_along(simulation_df[,1]))) +
  geom_line(aes(y = simulation_df[,1]), color = "blue") +
  labs(title = "Monte Carlo Simulation of Portfolio Value",
       x = "SIMULATION RUN",
       y = "RETURN PERCENTAGE") +
  theme_minimal()
# Calculating Advanced Risk Measures
simulation_results <- as.numeric(unlist(simulation_results))
mean_value <- mean(simulation_results, na.rm = TRUE)
median_value <- median(simulation_results, na.rm = TRUE)
VaR_5_percent <- quantile(simulation_results, 0.05, na.rm = FALSE)
VaR_50_percent <- quantile(simulation_results, 0.50, na.rm = TRUE)
VaR_95_percent <- quantile(simulation_results, 0.95, na.rm = TRUE)
CVaR_5_percent <- mean(simulation_results[simulation_results < VaR_5_percent], na.rm = TRUE)
# Creating a summary 
summary_stats <- data.frame(
  Statistic = c("Mean", "Median", "VaR (5%)", "VaR (50%)", "VaR (95%)", "CVaR (5%)"),
  Value = c(mean_value, median_value, VaR_5_percent*initial_portfolio_value/100, VaR_50_percent*initial_portfolio_value/100, VaR_95_percent*initial_portfolio_value/100, CVaR_5_percent*initial_portfolio_value/100)
)
print(summary_stats)
library(ggplot2)
# Plotting the histogram for verification of simulation
ggplot(data.frame(Value = simulation_results), aes(x = Value)) +
  geom_histogram(binwidth = 0.01, fill = "blue", color = "black", alpha = 0.7) +
  geom_vline(aes(xintercept = VaR_5_percent), color = "red", linetype = "dashed", size = 1) +
  geom_vline(aes(xintercept = VaR_95_percent), color = "green", linetype = "dashed", size = 1) +
  labs(title = "Histogram of Simulation Results",
       x = "Portfolio Value % Return",
       y = "Frequency") +
  theme_minimal()
# Same process repeated for Mid-Term Return
df <- read_excel(file_path)
df <- df %>%
  mutate(
    DATE = as.Date(Date, format="%Y-%m-%d"),
    WEEK_INDEX = as.numeric(Week_Index),
    QUARTER_INDEX = as.numeric(Quarter_Index)
  )
mean = mean(df$Mid_Term_Return) # "Mid_Term_Return is chosen as this analysis is going to be for Mid-Term
sd = sd(df$Mid_Term_Return)
preProcess_values <- preProcess(df, method = c("center", "scale"))
df <- predict(preProcess_values, df)
library(PerformanceAnalytics)
num_simulations <- 1000
simulation_results <- data.frame()
initial_portfolio_value<-1000
for (i in 1:num_simulations) {
  simulated_returns <- rnorm(n = nrow(df), mean = mean, sd = sd)
  simulation_results <- rbind(simulation_results, simulated_returns)
}
simulated_risks <- apply(simulation_results, 2, function(x) quantile(x, probs = c(0.05, 0.95)))
summary_simulation_results <- summary(simulation_results)
print(summary_simulation_results)
library(ggplot2)
simulation_df <- as.data.frame(simulation_results)
simulation_df <- (simulation_results)*100
ggplot(simulation_df, aes(x = seq_along(simulation_df[,1]))) +
  geom_line(aes(y = simulation_df[,1]), color = "blue") +
  labs(title = "Monte Carlo Simulation of Portfolio Value",
       x = "SIMULATION RUN",
       y = "RETURN PERCENTAGE") +
  theme_minimal()
simulation_results <- as.numeric(unlist(simulation_results))
mean_value <- mean(simulation_results, na.rm = TRUE)
median_value <- median(simulation_results, na.rm = TRUE)
VaR_5_percent <- quantile(simulation_results, 0.05, na.rm = FALSE)
VaR_50_percent <- quantile(simulation_results, 0.50, na.rm = TRUE)
VaR_95_percent <- quantile(simulation_results, 0.95, na.rm = TRUE)
CVaR_5_percent <- mean(simulation_results[simulation_results < VaR_5_percent], na.rm = TRUE)
summary_stats <- data.frame(
  Statistic = c("Mean", "Median", "VaR (5%)", "VaR (50%)", "VaR (95%)", "CVaR (5%)"),
  Value = c(mean_value, median_value, VaR_5_percent*initial_portfolio_value, VaR_50_percent*initial_portfolio_value, VaR_95_percent*initial_portfolio_value, CVaR_5_percent*initial_portfolio_value)
)
print(summary_stats)
library(ggplot2)
ggplot(data.frame(Value = simulation_results), aes(x = Value)) +
  geom_histogram(binwidth = 0.01, fill = "blue", color = "black", alpha = 0.7) +
  geom_vline(aes(xintercept = VaR_5_percent), color = "red", linetype = "dashed", size = 1) +
  geom_vline(aes(xintercept = VaR_95_percent), color = "green", linetype = "dashed", size = 1) +
  labs(title = "Histogram of Simulation Results",
       x = "Portfolio Value % Return",
       y = "Frequency") +
  theme_minimal()
# Same process for Long-Term Return Analysis
df <- read_excel(file_path)
df <- df %>%
  mutate(
    DATE = as.Date(Date, format="%Y-%m-%d"),
    WEEK_INDEX = as.numeric(Week_Index),
    QUARTER_INDEX = as.numeric(Quarter_Index)
  )
mean = mean(df$Long_Term_Return)
sd = sd(df$Long_Term_Return)
preProcess_values <- preProcess(df, method = c("center", "scale"))
df <- predict(preProcess_values, df)
library(PerformanceAnalytics)
num_simulations <- 1000
simulation_results <- data.frame()
initial_portfolio_value<-1000
for (i in 1:num_simulations) {
  simulated_returns <- rnorm(n = nrow(df), mean = mean, sd = sd)
  simulation_results <- rbind(simulation_results, simulated_returns)
}
simulated_risks <- apply(simulation_results, 2, function(x) quantile(x, probs = c(0.05, 0.95)))
summary_simulation_results <- summary(simulation_results)
print(summary_simulation_results)
library(ggplot2)
simulation_df <- as.data.frame(simulation_results)
simulation_df <- (simulation_results)*100
ggplot(simulation_df, aes(x = seq_along(simulation_df[,1]))) +
  geom_line(aes(y = simulation_df[,1]), color = "blue") +
  labs(title = "Monte Carlo Simulation of Portfolio Value",
       x = "SIMULATION RUN",
       y = "RETURN PERCENTAGE") +
  theme_minimal()
simulation_results <- as.numeric(unlist(simulation_results))
mean_value <- mean(simulation_results, na.rm = TRUE)
median_value <- median(simulation_results, na.rm = TRUE)
VaR_5_percent <- quantile(simulation_results, 0.05, na.rm = FALSE)
VaR_50_percent <- quantile(simulation_results, 0.50, na.rm = TRUE)
VaR_95_percent <- quantile(simulation_results, 0.95, na.rm = TRUE)
CVaR_5_percent <- mean(simulation_results[simulation_results < VaR_5_percent], na.rm = TRUE)
summary_stats <- data.frame(
  Statistic = c("Mean", "Median", "VaR (5%)", "VaR (50%)", "VaR (95%)", "CVaR (5%)"),
  Value = c(mean_value, median_value, VaR_5_percent*initial_portfolio_value, VaR_50_percent*initial_portfolio_value, VaR_95_percent*initial_portfolio_value, CVaR_5_percent*initial_portfolio_value)
)
print(summary_stats)
library(ggplot2)
ggplot(data.frame(Value = simulation_results), aes(x = Value)) +
  geom_histogram(binwidth = 0.01, fill = "blue", color = "black", alpha = 0.7) +
  geom_vline(aes(xintercept = VaR_5_percent), color = "red", linetype = "dashed", size = 1) +
  geom_vline(aes(xintercept = VaR_95_percent), color = "green", linetype = "dashed", size = 1) +
  labs(title = "Histogram of Simulation Results",
       x = "Portfolio Value % Return",
       y = "Frequency") +
  theme_minimal()

### UPDATED PREDICTION MODELS WITH COMBINATION OF ANALYSIS RESULTS ###

# To ensure consistency among equities and prevent equity bias, partition is made with respect to Date
df <- read_excel(file_path)
set.seed(123)
df <- df %>% arrange(Week_Index)
df$Short_Return_New <- ifelse(df$Short_Term_Return >= 0.03, 1, 0)
df$Mid_Return_New <- ifelse(df$Mid_Term_Return >= 0.1, 1, 0)
df$Long_Return_New <- ifelse(df$Long_Term_Return >= 0.2, 1, 0)
sum(df$Short_Return_New)
sum(df$Mid_Return_New)
sum(df$Long_Return_New)
split_point <- round(nrow(df) * 0.8)
train_data <- df[1:split_point, ]
test_data <- df[(split_point + 1):nrow(df), ]
# Veryfring there is no missing data
train_data <- na.omit(train_data)
test_data <- na.omit(test_data)
# Logistic Regression for Short-Term Return
initial_model_short_binary <- glm(Short_Term_Positive_Binary ~ PX_Last + VOLUME + PE_RATIO + Dividend_Yield + 
                                    Price2Sales + Price2CashF + EPS + ROE + Debt2Equity + Current_Ratio + 
                                    Quick_Ratio + Gross_Margin + Operating_Margin + Net_Profit_Mar + EBITDA + 
                                    Free_CF + MA_7 + MA_28 + RSI + GDP + CPI + PPI +LongBond_Interest + PCE_DEFY
                                    + ShortTerm_Int + Market_Index + Rolling_Volatility + Rolling_Beta + Rolling_VaR
                                    + Rolling_Sharpe_Ratio, 
                                  data = train_data, family = binomial)
# Performing backward elimination for getting the best performing model
# and eliminating non-significant independent variables
short_binary <- step(initial_model_short_binary, direction = "backward")
# Summarizing the final model
summary(short_binary)
# Performance evaluation of validation data
log_predictions <- predict(short_binary, test_data, type = "response")
log_pred_class <- ifelse(log_predictions > 0.5, 1, 0)
log_confusion_short <- confusionMatrix(as.factor(log_pred_class), as.factor(test_data$Short_Term_Positive_Binary))
log_confusion_short_advanced <- confusionMatrix(as.factor(log_pred_class), as.factor(test_data$Short_Return_New))
# Confusion Matrix for performance evaluation and comparison to initial models
print(log_confusion_short)
print(log_confusion_short_advanced)

# Logistic Regression for Mid-Term Return
initial_model_mid_binary <- glm(Mid_Term_Positive_Binary ~ PX_Last + VOLUME + PE_RATIO + Dividend_Yield + 
                                  Price2Sales + Price2CashF + EPS + ROE + Debt2Equity + Current_Ratio + 
                                  Quick_Ratio + Gross_Margin + Operating_Margin + Net_Profit_Mar + EBITDA + 
                                  Free_CF + MA_7 + MA_28 + RSI+ GDP + CPI + PPI +LongBond_Interest + PCE_DEFY
                                + ShortTerm_Int + Market_Index + Rolling_Volatility + Rolling_Beta + Rolling_VaR
                                + Rolling_Sharpe_Ratio,  
                                data = train_data, family = binomial)
mid_binary <- step(initial_model_mid_binary, direction = "backward")
summary(mid_binary)
log_predictions <- predict(mid_binary, test_data, type = "response")
log_pred_class <- ifelse(log_predictions > 0.5, 1, 0)
log_confusion_mid <- confusionMatrix(as.factor(log_pred_class), as.factor(test_data$Mid_Term_Positive_Binary))
log_confusion_mid_advanced <- confusionMatrix(as.factor(log_pred_class), as.factor(test_data$Mid_Return_New))
print(log_confusion_mid)
print(log_confusion_mid_advanced)


# Logistic Regression for Long-Term Return
initial_model_long_binary <- glm(Long_Term_Positive_Binary ~ PX_Last + VOLUME + PE_RATIO + Dividend_Yield + 
                                   Price2Sales + Price2CashF + EPS + ROE + Debt2Equity + Current_Ratio + 
                                   Quick_Ratio + Gross_Margin + Operating_Margin + Net_Profit_Mar + EBITDA + 
                                   Free_CF + MA_7 + MA_28 + RSI+ GDP + CPI + PPI +LongBond_Interest + PCE_DEFY
                                 + ShortTerm_Int + Market_Index + Rolling_Volatility + Rolling_Beta + Rolling_VaR
                                 + Rolling_Sharpe_Ratio, 
                                 data = train_data, family = binomial)
Long_binary <- step(initial_model_long_binary, direction = "backward")
summary(Long_binary)
log_predictions <- predict(Long_binary, test_data, type = "response")
log_pred_class <- ifelse(log_predictions > 0.5, 1, 0)
log_confusion_long <- confusionMatrix(as.factor(log_pred_class), as.factor(test_data$Long_Term_Positive_Binary))
log_confusion_long_advanced <- confusionMatrix(as.factor(log_pred_class), as.factor(test_data$Long_Return_New))

print(log_confusion_long)
print(log_confusion_long_advanced)
