# Voice Of Predictives

## Gender Recognition by Voice and Speech Analysis
This database was created to identify a voice as male or female, based upon acoustic properties of the voice and speech. The dataset consists of 3,168 recorded voice samples, collected from male and female speakers. The voice samples are pre-processed by acoustic analysis in R using the seewave and tuneR packages, with an analyzed frequency range of 0hz-280hz (human vocal range).

## Data Set Explained
A dictionary of all variables of the original dataset can be found in Github - [Wiki page](https://github.com/pennysun3/VoiceOfPredictives/wiki).

## Exploratory data analysis
A brief EDA is done in [EDA] (https://github.com/pennysun3/VoiceOfPredictives/blob/master/Training%20Data%20EDA.ipynb). Overall, we have a clean, and balanced dataset. No missing values, no imputation needed. 

## Classification Models
We have run a total of 10 [classification models](https://github.com/pennysun3/VoiceOfPredictives/blob/master/PredictiveII_Voice_Identification_Project.ipynb): XGBoost	
Boosted Tree, SVM, Random Forest, Neural Network, Logistic Regression, CART, GAM, KNN, Naive Bayes. Note, since there is no built in GAM cross valiadation methods in python, we use R to run the [GAM model](https://github.com/pennysun3/VoiceOfPredictives/blob/master/GAM.md) separately.
Among them, XGBoost performs the best with a classification accuracy of %. Since we are making classification between male versus female voices, we believe accruacy is a good metric for evaluation of the results. 
