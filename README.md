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

  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "allscores = [{'Method': 'Logistic Regression', 'no PCA': lr_score_pc, 'PCA': lr_score_npc},\n",
    "         {'Method': 'Neural Network', 'no PCA': nnet_score_pc, 'PCA': nnet_score_npc},\n",
    "         {'Method': 'CART', 'no PCA': cart_score_pc, 'PCA': cart_score_npc},\n",
    "         {'Method': 'KNN', 'no PCA': knn_score_pc, 'PCA': knn_score_npc},\n",
    "         {'Method': 'Boosted Tree', 'no PCA': bt_score_pc, 'PCA': bt_score_npc},\n",
    "         {'Method': 'Random Forest', 'no PCA': rf_score_pc, 'PCA': rf_score_npc},\n",
    "         {'Method': 'XGBoost', 'no PCA': xgb_score_pc, 'PCA': xgb_score_npc},\n",
    "         {'Method': 'SVM', 'no PCA': svm_score_pc, 'PCA': svm_score_npc},\n",
    "         {'Method': 'Naive Bayes', 'no PCA': nb_score_pc, 'PCA': nb_score_npc},\n",
    "         {'Method': 'GAM', 'no PCA':  0.9400253, 'PCA': 0.927399}]\n",
    "df = pd.DataFrame(allscores)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style>\n",
       "    .dataframe thead tr:only-child th {\n",
       "        text-align: right;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: left;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>no PCA</th>\n",
       "      <th>PCA</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Method</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>XGBoost</th>\n",
       "      <td>0.964320</td>\n",
       "      <td>0.891710</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Boosted Tree</th>\n",
       "      <td>0.962746</td>\n",
       "      <td>0.911283</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>SVM</th>\n",
       "      <td>0.962436</td>\n",
       "      <td>0.852897</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Random Forest</th>\n",
       "      <td>0.962114</td>\n",
       "      <td>0.962429</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Neural Network</th>\n",
       "      <td>0.960216</td>\n",
       "      <td>0.932442</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Logistic Regression</th>\n",
       "      <td>0.958644</td>\n",
       "      <td>0.844066</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>CART</th>\n",
       "      <td>0.954844</td>\n",
       "      <td>0.932442</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>GAM</th>\n",
       "      <td>0.940025</td>\n",
       "      <td>0.927399</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>KNN</th>\n",
       "      <td>0.937483</td>\n",
       "      <td>0.902445</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Naive Bayes</th>\n",
       "      <td>0.928325</td>\n",
       "      <td>0.902445</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
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
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.index=df['Method']\n",
    "df[['no PCA','PCA']].sort_values(by=\"no PCA\",ascending=False)"
   ]
  }
