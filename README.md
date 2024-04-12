# Introduction: The Seed

March—the month of madness, the month of elite college basketball. It's the time when college students rally behind their schools, hoping for a chance to make it to the championship final. But out of the 64 teams that qualify for the tournament, only one will emerge as the champion. The heartbreak and frustration felt by the other 63 teams are palpable.

The 2024 March Madness was my introduction to the world of collegiate basketball, and it did not disappoint. From Cinderella stories to 16-seed upsets and back-to-back dominations, the tournament buzzed with excitement. As I crafted my predictions at the tournament's outset—relying on basic stats, input from friends, and a dash of chance—I realized that my choices were far from concrete.

As the tournament unfolded, I witnessed brackets crumbling left and right. It wasn't until the Oakland Uni. vs. University of Kentucky matchup that my own bracket met its demise. Oakland, an underdog not on my radar, pulled off an unexpected victory. Despite my limited knowledge of collegiate basketball, I felt comfortable with my selections. However, reality soon set in: Teams that defied popular expectations were not guaranteed to win, and by the end of the first round, only a mere 0.005% of perfect brackets remained, as reported by ESPN News. Thus, my interest in predicting a team’s performance within the tournament using a Machine Learning Approach began.


# AP Polls Visualization Commentary

My initial focus was on understanding how a team’s regular-season performance correlates with their tournament performance. To explore this, I decided to visualize the performance of selected No. 1 seeds and gain insights into their seeding based on the AP Polls.


![Image Alt Text](https://github.com/josephguzman03/planting-march-madness/blob/main/Comparison_of_Number_1_Seeds.jpg)


The trends seem to be subtle, with nothing too drastic. The majority of the teams seem to have the right to be nominated as a No. 1 seed, as their performances were known to be within the top 5 throughout most of the season. Yet, their performances did not correlate with their march madness performance, as Kansas lost against No. 8 seed Arkansas, and Purdue faced a major upset against No. 16 seed Fairleigh Dickinson. Notably, Alabama and Miami managed to make it to the Sweet 16 but fell short against teams that surpassed them to reach the final four. If we take another look at the teams' performances that made it to the championship game, their performances differ.   

![Image Alt Text](https://github.com/josephguzman03/planting-march-madness/blob/main/Comparison_of_champ_teams.jpg)


When comparing the performances of the teams that reached the championship game, it's notable that their scores exhibited less consistency throughout the season compared to Alabama. However, these teams showed improvement from their initial Pre scores to their final scores, indicating progression in their performance over time. UConn, the winners of the 2023 March Madness, had a very interesting performance, fluctuating throughout the season. This is important because relying solely on the AP Polls may not be a strong indicator of how these teams perform in the postseason.

While the answer to this question is highly anticipated, it's important to note that AP Polls serve as a supportive metric in classifying seeds for the March Madness bracket. Therefore, there is more than just what the AP Polls tell us.

For now, my goal is to figure out how to predict a team's postseason performance based on their regular-season performance. To understand how teams managed to win, we must also consider how teams are seeded. Thus, in this report, I replicated the seeding process through ML analysis using Linear Regression and Random Forest Regression.

# Seeding Replication Prediction 

## Data Cleaning 

First, I downloaded my datasets from statehead.com and acquired the necessary data points for my analysis to work. There were three vital datasets: history_stats, school_rank, and coach_stats_copy. Each dataset had its respective columns that were either important or repeated. Therefore, after a few rounds of data cleaning, my dataframe 'final' was created.

|    |   Rk | School        |   SRS |   SOS |   FG% |   3P% |   FT% |   ORB |   TRB |   AST |   BLK |   TOV |   PF |   seed |   Yrs |   Overall_W-L% |   SZNCHAMPS |   TORNCHAMPS |   NCAA |   FF |   NATCHAMPS |   SZNC_W-L% |   OC_W-L% |   C_NCAA |   C_FF |   C_CHAMP |\n|---:|-----:|:--------------|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|-----:|-------:|------:|---------------:|------------:|-------------:|-------:|-----:|------------:|------------:|----------:|---------:|-------:|----------:|\n|  0 |    4 | Alabama       | 23.19 |  9.65 | 0.442 | 0.335 | 0.725 |   484 |  1652 |   555 |   189 |   512 |  691 |      1 |   111 |          0.62  |          12 |            9 |     25 |    0 |           0 |       0.838 |     0.689 |        6 |      0 |         0 |\n|  1 |   11 | Arizona       | 19.08 |  8.34 | 0.494 | 0.378 | 0.708 |   356 |  1376 |   662 |   110 |   468 |  593 |      2 |   117 |          0.663 |          26 |            9 |     38 |    4 |           1 |       0.8   |     0.847 |        2 |      0 |         0 |\n|  2 |   12 | Arizona State | 11.29 |  8.18 | 0.421 | 0.322 | 0.689 |   395 |  1315 |   515 |   170 |   422 |  668 |     11 |   109 |          0.53  |           8 |            0 |     17 |    0 |           0 |       0.639 |     0.579 |        4 |      0 |         0 |\n|  3 |   13 | Arkansas      | 15.99 |  9.87 | 0.466 | 0.313 | 0.698 |   373 |  1272 |   462 |   182 |   458 |  694 |      8 |   101 |          0.64  |          26 |            7 |     35 |    6 |           1 |       0.611 |     0.73  |        6 |      0 |         0 |\n|  4 |   17 | Auburn        | 14.35 |  9.29 | 0.439 | 0.315 | 0.696 |   396 |  1231 |   479 |   172 |   414 |  655 |      9 |   118 |          0.542 |           5 |            3 |     13 |    1 |           0 |       0.618 |     0.666 |       12 |      1 |         0 |

## Linear Regression using MSE

When conducting the linear regression model. I had to distringuish my X and y features. Which where:

X = final[['Rk', 'SRS', 'SOS', 'FG%', '3P%', 'FT%', 'ORB', 'TRB', 'AST' 'BLK','TOV', 'PF', 'Yrs', 'Overall_W-L%', 'SZNCHAMPS', 'TORNCHAMPS', 'NCAA', 'FF', 'NATCHAMPS', 'SZNC_W-L%', 'OC_W-L%', 'C_NCAA', 'C_FF','C_CHAMP']]

y = final['seed']

After intizalizing the linear regressional model, and fitting the model to the data. the Minimal Empirical Risk (MSE): 1.9050898129541887. I was a little curious on how the MAE would be implimented, so after running the model, it resulted in Minimal Empirical Risk (MAE) of  1.1165496948751454. 


I began to perform calculate optimal weights using normal equations, augmenting to the feature matrix X. Which began to visualize: 

![Image Alt Text](https://github.com/josephguzman03/planting-march-madness/blob/main/LR_seeds.jpg)


As you can see, most predicted values weren't as consistent as the rest. Therfore, I initialized weights randomly with hyperparameters. I then performed **Gradient Descent** to acquire the optimal lost. Using the new optimal weights, I was able to Initialize a new linear regression model with the optimal weights.

In our analysis, I observed the following performance metrics for the linear regression model predicting seed values based on teams performance on various features:

1. **Mean Absolute Error (MAE):**
   - The model achieved a Mean Absolute Error (MAE) of approximately 1.16. This metric reflects the average magnitude of the differences between the predicted seed values and the actual ones. Lower MAE values indicate better accuracy in predictions.

2. **Mean Squared Error (MSE):**
   - The Mean Squared Error (MSE) for the model was around 2.18. This metric computes the average of the squared differences between predicted and actual values, providing insight into overall accuracy. Lower MSE values signify better model performance.

3. **R-squared (R2) Score:**
   - The model achieved an R-squared (R2) score of approximately 0.90. This score represents the proportion of variance in the seed values that the model can explain based on the features. A score of 1 indicates a perfect fit, while 0 suggests that the model does not explain any variability. The R2 score of 0.90 indicates a strong ability to explain variability.

Interpretation:
- The analysis suggests that the linear regression model performs well in predicting seed values based on the given features. The low MAE and MSE values indicate that predictions are generally close to actual values, with minor variability. Additionally, the high R2 score demonstrates accuracy and explanatory power in explaining the variability in seed values using the selected features. 

But I wanted to see if there was another model that can support my analysis. 

## Random Forest Regression

With the same X and y features, I began to split the data into training and testing sets, with a testing size of 80%. When i made the prediction on the entire dataset X, my MAE was 0.62, MSE was a 0.73 and R2 resulted in a 0.96. I then began to utlize xgboost to see if my model can improve its performance. Using a similar approach, I was able to conduct a prediction on X using Random Forest. Here are the results:

![Image Alt Text](https://github.com/josephguzman03/planting-march-madness/blob/main/RF_seeds.jpg)

Continuing on the analysis, the combined XGBoost and Random Forest model yielded exceptionally high accuracy and performance metrics:

1. **Mean Absolute Error (MAE):**
   - The MAE, at approximately 0.00033, indicates that the model's predictions for seed values were extremely accurate, with minimal average absolute differences between predicted and actual values.

2. **Mean Squared Error (MSE):**
   - With an MSE of approximately 2.48, the model showcased a level of precision, demonstrating negligible squared differences between predicted and actual seed values.

3. **R-squared (R2) Score:**
   - The R-squared score of about 0.9999999912691464 signifies an exceptional ability of the model to explain nearly all variability in seed values using the provided features. This close-to-perfect fit underscores the model's robustness and accuracy.

Interpretation:
- The analysis indicates that the combined XGBoost and Random Forest model delivers outstanding predictive performance, showcasing near-perfect accuracy in predicting seed values. These results affirm the model's reliability and effectiveness in leveraging the provided features to generate highly precise predictions, making it a valuable tool for our project's objectives.

## Outcomes? 


Therefore, using Random Forest with XGBoost and Linear regression using MSE, I was able to perform seed prediction based on the team's performance during the regular season. These seeding placements align with the NCAA committee's judgment of how they seed teams. If we solely predict the seed based on their judgment, we can also take into account the team's performance to achieve that. This model can be applied for any year, as it matched 99% of 2024's seeding placements. However, much research needs to be done, such as player performance, the committee's metrics, venue locations, and even studying the  No. 16 seed team truimpth runs. Overall, my model seems to manage robust claims and make accurate predictions.


Next, using these seeding placements, I'll embark on researching the probabilities of teams reaching the final four based on their seeding placements. Stay Tuned!


## Predicting Seeding Outcomes 

A recent video essay I watched, titled ‘The NCAA Tournament Is a Loser Machine’ by the YouTuber Secret Base, presented several compelling arguments. To begin with, the essay highlighted that ‘your seed does not determine how easy your road through the tournament will be.’ This caught my attention because most basketball playoffs follow a seeding-style tournament, where lower-seeded teams face higher-seeded opponents. Consequently, I decided to explore whether machine learning (ML) could predict a team’s seeding placement based on their season performances.

**This is part two of my project and currently am working on**