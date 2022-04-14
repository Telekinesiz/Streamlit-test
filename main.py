import streamlit as st
import praw
import os
from praw.models import MoreComments
import pandas as pd
from transformers import pipeline
import nltk
from langdetect import detect
import numpy as np
from nltk.corpus import stopwords
import string
import requests
from datetime import datetime
from PIL import Image
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import pyLDAvis.gensim_models
from streamlit import components
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import plotly.express as px
import time

#credentials*********************************************************
reddit_credentials = praw.Reddit(
    user_agent=os.getenv('praw_user_agent'),
    client_id=os.getenv('praw_client_id'),
    client_secret=os.getenv('praw_client_secret'),
    username=os.getenv('praw_username'),
    )

headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36"
    }


def comments_iterator(submission):
    comments = []
    for top_level_comment in submission.comments:
        if isinstance(top_level_comment, MoreComments):
            continue

        comments.append(
            {'Original comment': top_level_comment.body,
             'Comment': top_level_comment.body,
             'Score': top_level_comment.score}
        )
    return comments

# Filter only english
def detect_en(text):
    try:
        return detect(text) == 'en'
    except:
        return False

#lemmetize text
def lemmatize_text(text):
    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

def panda_clear_data(merged_df,column_for_classification):
    pd.options.mode.chained_assignment = None  # default='warn'
    column_name = column_for_classification
    # Remove emojis
    merged_df.astype(str).apply(lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
    # make lowercase
    merged_df[column_name].to_string()
    merged_df[column_name] = merged_df[column_name].str.lower()
    # remove ascii and other special symbols
    merged_df[column_name] = merged_df[column_name].str.replace('\n', '')
    merged_df[column_name] = merged_df[column_name].str.replace('\t', '')
    merged_df[column_name] = merged_df[column_name].str.replace('"', '')
    merged_df[column_name] = merged_df[column_name].str.replace("'", "")
    merged_df[column_name] = merged_df[column_name].str.replace("“", "")
    merged_df[column_name] = merged_df[column_name].str.replace("”", "")
    merged_df[column_name] = merged_df[column_name].str.replace("’", "")
    merged_df[column_name] = merged_df[column_name].str.replace(' {2,}', '', regex=True)
    merged_df[column_name] = merged_df[column_name].str.strip()
    merged_df[column_name] = merged_df[column_name].str.replace('[{}]'.format(string.punctuation), '', regex=True)
    #merged_df[column_name] = merged_df[column_name].str.contains(r'[^\x00-\x7F]', na=False)]
    #merged_df[column_name] = merged_df[column_name].replace(r'[^\x00-\x7F]', '', na=False)
    merged_df[column_name] = merged_df[column_name].replace(r'[^\w\s]|_', '', regex=True)
    merged_df = merged_df[merged_df[column_name].apply(detect_en)]
    # remove stop words
    stop_words = stopwords.words('english')
    merged_df[column_name] = merged_df[column_name].fillna("")
    merged_df[column_name] = merged_df[column_name].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    # Remove URLs
    merged_df[column_name] = merged_df[column_name].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)
    # drop rows with no comments
    merged_df[column_name] = merged_df[column_name].fillna("")
    merged_df[column_name].replace('', np.nan, inplace=True)
    merged_df.dropna(subset=[column_name], how='any', inplace=True)
    # drop duplicates
    merged_df.drop_duplicates(subset=[column_name])
    #lemmatize text
    merged_df[column_name] = merged_df[column_name].apply(lemmatize_text)
    return (merged_df)

def sentiment_classifire(df, column_with_text):
    classifier = pipeline('sentiment-analysis')
    cloumn_with_label = column_with_text + ' label'
    column_with_probability = 'Label probability'
    column_index = 1 + int(df.columns.get_loc(column_with_text))
    #df[column_with_text] = df[column_with_text].apply
    df.insert(column_index, column_with_probability, 'none')
    df.insert(column_index, cloumn_with_label, 'none')
    limitizer_for_test = 0
    for ind in df.index:
        # setting limit for length of input text in order to avoid errors. this model has limit for input
        stemmed_text = df[column_with_text][ind]
        if len(stemmed_text) >= 212:
            stemmed_text = stemmed_text[0:211]
        else:
            pass

        input_text = ' '.join(str(x) for x in stemmed_text)
        print("row number: " + str(limitizer_for_test))
        print('text: ' + input_text)
        print("length of sentence: " + str(len(input_text)))
        result = classifier(input_text)
        label_dictionary = result[0]
        print(label_dictionary)
        label = label_dictionary['label']
        probability = label_dictionary['score']
        df[cloumn_with_label][ind] = label
        df[column_with_probability][ind] = probability
        #print('Label: '+ label)

        limitizer_for_test  += 1
        #if limitizer_for_test  >= 20:
        #    break
    return (df)

def topic_categorizer(df, column_with_text, candidate_labels):
    classifier = pipeline('zero-shot-classification')
    column_with_label = column_with_text + ' category'
    column_with_probability = 'Category probability'
    column_index = 1 + int(df.columns.get_loc(column_with_text))
    #df[column_with_text] = df[column_with_text].apply(literal_eval)
    df.insert(column_index, column_with_probability, 'none')
    df.insert(column_index, column_with_label, 'none')
    limitizer_for_test = 0
    # setting limit for length of input text in order to avoid errors. this model has limit for input
    for ind in df.index:

        stemmed_text = df[column_with_text][ind]
        if len(stemmed_text) >= 212:
            stemmed_text = stemmed_text[0:211]
        else:
            pass

        input_text = ' '.join(str(x) for x in stemmed_text)
        print("row number: " + str(limitizer_for_test))
        print('text: ' + input_text)
        print("length of sentence: " + str(len(input_text)))
        result = classifier(input_text, candidate_labels)

        scores = result['scores']
        labels = result['labels']

        #finding largest score
        largest_score = scores[0]
        for number in scores:
            if number > largest_score:
                largest_score = number

        #finding largest score index and label by undex
        for index, score in enumerate(scores):
            if score == largest_score:
                lab_index = index

        label = labels[lab_index]
        df[column_with_label][ind] = label
        df[column_with_probability][ind] = largest_score
        #print('Label: '+ label)
        limitizer_for_test  += 1
        #if limitizer_for_test  >= 20:
        #    break
    return (df)


# main script
if __name__ == '__main__':
    #Front_end. This paert placed here in order to show something to user while script working
    image = Image.open('Redditpic.png')
    st.image(image, use_column_width=True)

    st.title('Reddit topic Analyzer')

    st.markdown("""
          This app loads topic from Reddit and perform analysis that includes comments classification !
          * **Python libraries:** pandas, streamlit, numpy, matplotlib, seaborn, pyLDAvis, gensim, requests, nltk, praw, transformers, langdetect
          * **Data source:** reddit.com.
          """)
    with st.spinner('Wait for it...'):
        time.sleep(600)



    #backend
    # for classification
    column_for_classification = 'Comment'
    candidate_labels = ["war", "science", "politics", "economy", "amusement", "hobby", 'criminal', 'culture', 'meme',
                        'sport']

    submission_id = st.sidebar.text_input("reddit topic ID", value="tu8l5v")
    reddit = reddit_credentials
    submission = reddit.submission(submission_id)

    print(submission)
    award_price = 0
    for award in submission.all_awardings:
        coin_price = award["coin_price"]
        count = award["count"]
        award_price += count*coin_price
    print(award_price)

    topic_date = submission.created_utc
    topic_name = submission.title
    topic_text = submission.selftext
    topic_score = submission.score
    topic_ratio = submission.upvote_ratio
    Page_url = 'https: // www.reddit.com' + str(submission.permalink)
    json_post_url = 'https://www.reddit.com' + str(submission.permalink) + '.json'

    # will get image or link
    try:
        link_or_image = requests.get(json_post_url, headers=headers).json()
        link_or_image_2 = link_or_image[0]['data']['children'][0]['data']
        link_or_image_3 = link_or_image_2['url_overridden_by_dest']
    except:
        link_or_image_3 = "There is no avialable link or image"
        pass


    comments = comments_iterator(submission)
    merged_df = pd.json_normalize(comments)
    panda_clear_data(merged_df, column_for_classification)
    df = panda_clear_data(merged_df, column_for_classification)
    column_with_text = column_for_classification
    sentiment_classifire(df, column_with_text)
    topic_categorizer(df, column_with_text, candidate_labels)

    dt_object = datetime.fromtimestamp(int(topic_date))
    topic_name_with_url = "["+topic_name+"](https://www.reddit.com/r/place/comments/"+submission_id+"/)"


    graf_df_common = df.groupby(['Comment category','Comment label'], sort=False).size().reset_index(name='Count')
    graf_df_label = df.groupby(['Comment label'], sort=False).size().reset_index(name='Count')
    graf_df_category = df.groupby(['Comment category'], sort=False).size().reset_index(name='Count')



    #diagrams

    fig1 = px.pie(
        hole=0.1,
        labels=graf_df_label['Comment label'],
        names=graf_df_label['Comment label'],
        values=graf_df_label['Count'],
    )

    fig2 = px.pie(
        hole=0.3,
        labels=graf_df_category['Comment category'],
        names=graf_df_category['Comment category'],
        values=graf_df_category['Count'],
    )



    data_words = df[column_for_classification]
    id2word = corpora.Dictionary(data_words)
    corpus = []
    for text in data_words:
        new = id2word.doc2bow(text)
        corpus.append(new)

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=50,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha="auto")

    #pyLDAvis.enable_notebook()

    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word, mds="mmds", R=30)
    #vis
    pyLDAvis.save_html(vis, 'ida.html')
    with open('./ida.html', 'r') as f:
        html_string = f.read()


    # Front_end
    st.success('Done!')


    st.markdown("""***""")
    st.header(topic_name_with_url)
    st.markdown("""Created at: """ + str(dt_object))

    image = st.image(link_or_image_3)
    st.markdown(link_or_image_3)
    st.markdown(topic_text)

    st.markdown("""***""")
    col1, col2, col3 = st.columns(3)
    col1.metric("Topic score", topic_score)
    col2.metric("Score ratio", topic_ratio)
    col3.metric("Topic price*", award_price)
    st.markdown(
        """(Topic price value counts as a summary of prices of all awards, given to the topic and comments in it.)""")


    st.markdown("""***""")
    st.header("Comments categories")
    st.plotly_chart(fig1)

    st.markdown("""***""")
    st.header("Comments labels")
    st.plotly_chart(fig2)

    st.markdown("""***""")
    st.subheader('Interactive comment visualization')
    components.v1.html(html_string, width=1000, height=800, scrolling=False)


    st.markdown("""***""")
    st.subheader('Labeled comments from topic')
    st.dataframe(df)

    st.markdown("""***""")
    st.subheader('Thanks for using. Have a nice day')
    image = Image.open('reddit_bye.png')
    st.image(image, use_column_width=True)