# sportsbet
analyse sports betting (and hopefully win!)



We are given a sports betting dataset to analyse and make predictions of. First we will clean the data, by filling in the NA with believable values. Then we shall make several scatterplots to check for correlations. From the dense linear clustering of points I infer that odds and implied_probability are correlated.

Later we use Logistic regression to model because: 

Logistic regression is a powerful tool for analyzing sports bets due to several key features and advantages:

Binary Outcome Modeling: Sports betting often involves binary outcomes (e.g., win/loss, over/under). Logistic regression is specifically designed to handle binary dependent variables, making it a natural choice for this type of analysis.

Probability Estimates: Logistic regression provides probability estimates for each possible outcome. This is particularly useful in sports betting, where understanding the likelihood of different outcomes is crucial for making informed bets.

Interpretability: The coefficients in logistic regression represent the log odds of the outcome for a one-unit change in the predictor variable. This interpretability helps bettors and analysts understand the impact of various factors (e.g., player statistics, team performance, weather conditions) on the probability of a particular outcome.

Handling Multiple Predictors: Logistic regression can incorporate multiple predictor variables, allowing a comprehensive analysis of how different factors influence the outcome of a sports event. These predictors can include quantitative data (e.g., player performance metrics) and qualitative data (e.g., team morale, injuries).

Efficiency with Small to Medium-Sized Datasets: Logistic regression is computationally efficient and works well with small to medium-sized datasets. In many sports betting scenarios, the available data might not be extensive, and logistic regression can still provide reliable results.

Modeling Flexibility: Extensions of logistic regression, such as multinomial logistic regression, allow for modeling outcomes with more than two categories. This is useful if the sports betting involves multiple possible outcomes (e.g., win/draw/loss).
