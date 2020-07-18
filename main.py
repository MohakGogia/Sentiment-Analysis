import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import tweepy
import re
import os
from textblob import TextBlob
from wordcloud import WordCloud




#Twitter Authentication
CONSUMER_KEY = "consumer key"
CONSUMER_KEY_SECRET = "comsumer key secret"
ACCESS_TOKEN = "access token"
ACCESS_TOKEN_SECRET = "access token secret"


auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_KEY_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

api = tweepy.API(auth)



searchQuery = input("Enter keyword/hashtag to seach for: ")
noOfTweets = int(input("Enter how many tweets to analyze:"))
# print(searchQuery,noOfTweets)



#Fetching Tweets from twitter and adding them to a Data Frame
tweets = api.search(q = searchQuery, count = noOfTweets, lang = 'en', include_rts = False)
df = pd.DataFrame(data= [tweet.text for tweet in tweets], columns = ['Tweets'])
df['ID'] = np.array([tweet.id for tweet in tweets])
df['TimeStamp'] = np.array([tweet.created_at for tweet in tweets])
df['Source'] = np.array([tweet.source for tweet in tweets])
df['Likes'] = np.array([tweet.favorite_count for tweet in tweets])
df['Retweets'] = np.array([tweet.retweet_count for tweet in tweets])
df['User'] = np.array([tweet.user.screen_name for tweet in tweets])
df.to_csv('tweets.csv')

# df.info()


punctuations = '''()-![]{};:+'"\,<>/$%^*_~'''


#Cleaning tweets to remove unwanted text
def clean_tweets(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)          #Remove @mentions
    text = re.sub(r'RT[\s]+', '', text)                #Remove RT (retweet)
    text = re.sub(r'https?:\/\/\S+', '',text)          #Remove any hyperlink
    text = ''.join([i for i in text if not i in punctuations])
    return text


df['Tweets'] = df['Tweets'].apply(clean_tweets)               

# df.to_csv("cleaned.csv")

#Getting Subjectivity and Polarity from the Tweet
def get_subjectivity(text):
        return TextBlob(text).sentiment.subjectivity

def get_polarity(text):
        return TextBlob(text).sentiment.polarity
    

df['Subjectivity']=df['Tweets'].apply(get_subjectivity)
df['Polarity']=df['Tweets'].apply(get_polarity)



#Finding Sentiment
def get_analysis(score):
    if score<0:
        return 'Negative'
    elif score==0:
        return 'Neutral'
    else:
        return 'Positive'



df['Sentiment']=df['Polarity'].apply(get_analysis)


ptweets = 0
ntweets = 0
neutraltweets = 0


# Displaying tweets categorically
print("-----------------------------------------")
print("-------------POSITIVE TWEETS-------------")
print("-----------------------------------------")
for i in df.index: 
    if df['Sentiment'][i] == 'Positive':
        print(df['Tweets'][i])
        ptweets += 1



print("-----------------------------------------")
print("-------------NEUTRAL TWEETS-------------")
print("-----------------------------------------")
for i in df.index: 
    if df['Sentiment'][i] == 'Neutral':
        print(df['Tweets'][i])
        neutraltweets += 1
       


print("-----------------------------------------")
print("-------------NEGATIVE TWEETS-------------")   
print("-----------------------------------------")    
for i in df.index: 
    if df['Sentiment'][i] == 'Negative':
        print(df['Tweets'][i])
        ntweets += 1



print("---------Tweet Count---------")
print(f'Positive: {ptweets}')
print(f'Neutral: {neutraltweets}')
print(f'Negative: {ntweets}') 



#Visualization Graphs

my_path = os.path.dirname(__file__)


# 1->WordCloud of the words in the fetched tweets
allWords = ''.join([twts for twts in df['Tweets']])


wordCloud = WordCloud(width = 500, height = 300, random_state = 21 , max_font_size = 110).generate(allWords)
plt.imshow(wordCloud,interpolation = 'bilinear')
plt.axis('off')
wordCloud.to_file((my_path + '\\visualization\\wordcloud.png')




#2->Sentiment Count Graph
fig = sns.countplot(data = df, x = 'Sentiment', palette = 'rocket').set_title("Sentiment Count Graph")
fig = fig.get_figure()
fig.savefig(my_path+ '\\visualization\\sentiment_count.png')




#3->Subjectivity VS Polarity Graphs
plt.figure(figsize = (6,6))
plt.scatter(df['Polarity'], df['Subjectivity'], color = 'Green')
plt.title('Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.tight_layout()
plt.savefig(my_path+ '\\visualization\\sentiment_analysis.png')




#4->Sentiment Pie Chart
posi_per = (ptweets*100)/noOfTweets
negi_per = (ntweets*100)/noOfTweets
neut_per = (neutraltweets*100)/noOfTweets

labels = 'Positive', 'Neutral', 'Negative'
sizes = [posi_per,neut_per,negi_per]
colors = ['gold', 'yellowgreen', 'lightcoral']
explode = (0.05, 0.05, 0.05)


plt.figure(figsize = (6,6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True)
plt.axis('equal')
# plt.title("Sentiment Pie Chart")
plt.tight_layout()
plt.savefig(my_path+ '\\visualization\\sentiment_pie_chart.png')





#5->Tags Count 
hashtags = df["Tweets"].str.extractall(r'(\#\w+)')[0].value_counts().reset_index()
hashtags.columns = ["Hashtags", "Count"]


# selecting top 5 most frequent hashtags     
hashtags = hashtags.nlargest(n = 5, columns = "Count") 
plt.figure(figsize = (25,15))
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer = True))
ax = sns.barplot(data = hashtags, x = 'Count', y = 'Hashtags', palette = 'dark')
plt.tight_layout()
# plt.title("Tags Count")
plt.savefig(my_path+ '\\visualization\\tags_count.png')



