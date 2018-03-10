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
Among them, XGBoost performs the best with a cross validation classification accuracy of 96.43%. Since we are making classification between male versus female voices, we believe accruacy is a good metric for evaluation of the results. 

       "                       no PCA       PCA\n",
       "Method                                 \n",
       "XGBoost              0.964320  0.891710\n",
       "Boosted Tree         0.962746  0.911283\n",
       "SVM                  0.962436  0.852897\n",
       "Random Forest        0.962114  0.962429\n",
       "Neural Network       0.960216  0.932442\n",
       "Logistic Regression  0.958644  0.844066\n",
       "CART                 0.954844  0.932442\n",
       "GAM                  0.940025  0.927399\n",
       "KNN                  0.937483  0.902445\n",
       "Naive Bayes          0.928325  0.902445"
 
