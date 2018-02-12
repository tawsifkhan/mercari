# Kaggle Mercari Price Suggestion Challenge

[Competition link](https://www.kaggle.com/c/mercari-price-suggestion-challenge)

### Methodology 

Dataset contains 5 variables - item name, item condition, category name, brand name, shipping info and item description.

Item name and item descriptions are texts with an impractical amount of levels; so they require some preprocessing. The rest can be just be encoded to a numerical variable. 

![Train_head](/images/train_head.png)

So for item name and description I implemented a TFIDF vectorization process on a stemmed version of the strings to extract 50k feature variables.

    corpus=np.hstack([train.ndesc,test.ndesc])
    sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, max_df=.99,
                                    use_idf=True, smooth_idf=False,
                                    stop_words='english',
                                    sublinear_tf=True,max_features=f_n)
    transformed = sklearn_tfidf.fit_transform(corpus)

Finally I implemented a gradient boosting algorithm and tuned the hyper-parameters with a grid search.


#### Leaderboard score 0.48978 (RMSLE)
