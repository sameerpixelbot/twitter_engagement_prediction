# [Twitter Engagement Prediction](https://github.com/sameerpixelbot/twitter_engagement_prediction) : Project Overview

- Made this **project for** anyone to **predict** how **engaging** a **tweet** will be **based** on the **performance of the tweets of set of users in a bubble**( like a community of people interested in smartphones)
- Scraped over 20,000 tweets over a span of 1 month of 80 twitter users related to tech Youtube community.
- Used ***Google's Base BERT*** for extracting tweet ***text Embeddings.***
- **Made a Metric** that says how **engaging** a particular tweet **based on its likes, retweets, replies, followers.**
- Models used were ***Random Forrest Regressor, XGBoost Regressor.***
- Main project is in [Scraper for individual user branch](https://github.com/sameerpixelbot/twitter_engagement_prediction/tree/scraper_for_individual_user)**.**

## Code and Resourses used :

- **Python version :** 3.7
- **Scraper GitHub :** [twitterscraper](https://github.com/taspinar/twitterscraper)
- **Tensorflow : **2.0 (**GPU Version**)
- **BERT Package :** bert-tensorflow
- **TensorFlow HUB**
- **SciKit Learn**

## How it works :

- - **List** of **twitter users** are **stored** in a file.
  - The **tweets of the users** are **scraped** using **twitterscraper** module which uses 
    **BeautifulSoup.**
- - **Tweets** are passed through ***Base BERT*** for sentence ***Embeddings.***
  - Useful **features** are **extracted** from ***tweets metadata.***
  - The ***Engagement score*** metric is formed **from** the ***likes, retweets, replies, 
    followers*** of each ***tweet.***
- The **BERT Embeddings** and the tweets **metadata** are used to **predict** **Engagement Score**
  using ***Random Forest Regressor.***

## Data Collection :

**Files used :**  [tweets_scraper_users_main](https://github.com/sameerpixelbot/twitter_engagement_prediction/blob/scraper_for_individual_user/tweets_scraper_users_main.ipynb)**,** [add_users](https://github.com/sameerpixelbot/twitter_engagement_prediction/blob/scraper_for_individual_user/add_users.ipynb)**.**

- **add_users** function is used to **add twitter users** in a **file.**
- When **collecting Tweets** of users **everyday,** the tweets which were **tweeted 3 days 
  back** are **collected**, which gives them **enough time to get maximum engagement.**
- The **twitterscraper module** collects the tweets by **querying** through **twitter advanced 
  search,** so a **function is made to output a query format to get the tweets of a user 
  on the given day.**
- After scraping all the tweets are **exported as a .csv file.**

**These were the features extracted :**

- **screen_name**
- **username**
- user_id 
- tweet_id
- tweet_url 
- timestamp
- timestamp_epochs 
- ***text***
- text_html
- links
- hashtags
- **has_media**
- img_urls
- video_url
- ***likes***
- ***retweets***
- ***replies***
- ***is_replied***
- ***is_reply_to***
- parent_tweet_id
- reply_to_users

## Data Cleaning :

**Files used :** [data_cleaning](https://github.com/sameerpixelbot/twitter_engagement_prediction/blob/scraper_for_individual_user/data_cleaning.ipynb)**.**

- Here we **scrape** such that **none** of the **data has any null values**, So 
  their is **no need** 
  to **worry about null values.**
- In this **stage** we have **two functions :**
  - **get_tweet_csv :** 
    - It **concatenates** all the individual **csv** files that are **collected each 
      day.**
    - It **also adds** a **new column** ***'day_year'*** which represents the **day 
      of the year** the 
      **tweet was tweeted.**
  - **get_bert_emb_csv :**
    - It **concatenates** all the individual ***BERT Embeddings csv*** files that 
      are **generated** 
      each day **after collecting the tweets.**

## Feature Engineering :

**Files used :** [extract_bert_embeddings](https://github.com/sameerpixelbot/twitter_engagement_prediction/blob/scraper_for_individual_user/extract_bert_embeddings.ipynb)**,** [feature_eng_funcs](https://github.com/sameerpixelbot/twitter_engagement_prediction/blob/scraper_for_individual_user/feature_eng_funcs.ipynb)**.**

- **Work** in [extract_bert_embeddings](https://github.com/sameerpixelbot/twitter_engagement_prediction/blob/scraper_for_individual_user/extract_bert_embeddings.ipynb) **:**
  - **Here using** ***TensorFlow HUB, TensorFlow 2.0 and Keras*** we will do ***Transfer Learning*** **.**
  - We use the **already learnt** knowledge of ***Google's Base BERT*** to **Extract sentence embeddings** of our tweets**.**
  - Using **Keras** we first make **3 Keras Input Layers for** the **Base BERT Model** **.**
  - Using **TensorFlow HUB** we **Download a trainable Base BERT model** **.**
  - Now we **Deploy** this model **in our GPU**(My GPU : NVIDIA MX110, CUDA compute capability: 5.0, Basic GPU but its fine for this model **😉**)**.**
  - Now this is used to **extract** ***BERT Embeddings Batchwise*** **.** (Batch size=**200 tweets**)
  - We use the **output** of the **first token** for **regression** **.** ( **[CLS]** token is used for regression and classification )
  - Finally all the **Bert Embeddings** (Vector **size** is **768**) of tweets is **made** as a **dataframe** and then **exported** as a **.csv file** **.**
  - The **above process** is done **every time** we **scrape new data** **.**

- **Work** in [feature_eng_funcs](https://github.com/sameerpixelbot/twitter_engagement_prediction/blob/scraper_for_individual_user/feature_eng_funcs.ipynb) **:** In this file there are **2 functions** **.**

```python
def engage_score(likes,retweets,replies,followers):
    
    import numpy as np
    #weights for likes,retweets,replies
    l,rt,rp,f=1,2,3,2
    num=l*np.log2(likes+8)+rt*np.log2(retweets+4)+rp*np.log2(replies+2)
    den=f*np.log10(followers+500+np.random.randint((followers+500)*0.1,(followers+500)*0.2))
    return(num/(den+1))
```

  - **engage_score :**
    - This **Function** is used **to calculate** the **engagement score** of a tweet **based on
      the likes, retweets,replies, followers it has.**
    - The **Quality** of Twitter **engagement** can be said **accurately** by a ***function of likes/impressions, retweets/impressions and replies/impressions.***
    - ***BUT*** the **problem** arises at ***IMPRESSIONS,*** we **can not get** the **data** of how many **impressions a tweet has got** even **by** scraping or using **tweepy.**
    - **Impressions** data can **only** be **known by** the **owner of the tweet,** **cant** be known **by anyone else.**
    - In order to ***overcome this problem,*** if we ***only*** consider ***likes,retweets and replies*** their is a ***bigger problem.***
    - **As their is ***difference between*** people just seeing( impressions = no of people who saw the tweet) the tweet and interacting( actions such as liking, retweeting and repling) with the tweet.**
    - Actions such as **liking, retweeting and replying say that people are more engaged with the tweet, but not just looked at it.**
    - This says that, **more the interactions** such as liking, retweeting and replying **indicates more people are engaged with the tweet.**
    - **But this is not it,** ***What matters is how much percentage of people who looked at the tweet interacted with it.***
    - This **percentage** gives the **right degree** of **engagement a tweet has.**
    - So we **cant** just **only** use **likes, retweets and replies** but we **also need** the amount of **impressions** a tweet has got, so that we can **determine how much engaging a given tweet is.**
    - Thus we **definitely** need either the **impressions** data **or** any **other parameter which is highly correlated with impressions data.**
    - So in this **we** will **use** the no of **followers the user of the tweet has.**
    - Generally **impressions** are **+/-20%** of the no of **followers** the **user of the tweet has.**
    - **So we will calculate impression by this equation followers+500+np.random.randint((followers+500)*0.1,(followers+500)*0.2).**
    - Now we use the **log10(impressions)** to give the **magnitude of impression a tweet has.**
    - Then the **engagement score** is **calculated by the function above.**

- **get_followers_dict :**
  - This is used to **get followers count using tweepy.**

## Model Training :

**Files used :** [hyperparametertuning part 1](https://github.com/sameerpixelbot/twitter_engagement_prediction/blob/scraper_for_individual_user/hyperparameter%20tuning%20part%201.ipynb)**,** [hyperparametertuning part 2](https://github.com/sameerpixelbot/twitter_engagement_prediction/blob/scraper_for_individual_user/hyperparametertuning%20part%202.ipynb)**,** [error checker](https://github.com/sameerpixelbot/twitter_engagement_prediction/blob/scraper_for_individual_user/error%20checker.ipynb)**.**

- **Work** in [hyperparametertuning part 1](https://github.com/sameerpixelbot/twitter_engagement_prediction/blob/scraper_for_individual_user/hyperparameter%20tuning%20part%201.ipynb) **:**

  - In this file we **compared Random forest Regressor and XGBoost Reggressor** 
    with **different features** in their **default parameters.**
  - The performance of **Random forest Regressor** was ***better with mse = 0.38***
  - **Then Ran Random Search CV for *2days and 3nights* on my laptop and still did 
    not finish so have to stop it and did manually in 
    [hyperparametertuning part 2](https://github.com/sameerpixelbot/twitter_engagement_prediction/blob/scraper_for_individual_user/hyperparametertuning%20part%202.ipynb)** **.**

- **Work** in [hyperparametertuning part 2](https://github.com/sameerpixelbot/twitter_engagement_prediction/blob/scraper_for_individual_user/hyperparametertuning%20part%202.ipynb) **:**

  - In this file got the **best performance with default parameter** itself**.**
  - A point to be noted is I was not able to do proper Random Search CV Due to 
    lack of proper hardware, So when I get better hardware The Random Search CV 
    will be done.

- **Work** in [error checker](https://github.com/sameerpixelbot/twitter_engagement_prediction/blob/scraper_for_individual_user/error%20checker.ipynb) **:**

  - Here we did check **if a mse below 0.5 was any good.**
  - By the analysis we can say that **it was alright.**
  - **But getting very low error might take a years worth data as trends in this 
    community  
    change everyday so getting data form a long period of time may give a 
    bigger advantage to the model.(Data on new smartphones leaks and new viral videos made by users and any breaking news change the distribution of data very much so considering that this has performed very well)**

## Future :

- The **model** can be **improved** by using **features** such as the media in tweets, like **images and videos.**
- We can also **fine tune BERT** to our dataset when we get alot **more data.**
