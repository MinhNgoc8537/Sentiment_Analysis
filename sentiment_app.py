# Import library
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import yellowbrick
from streamlit_yellowbrick import st_yellowbrick
from wordcloud import WordCloud
from streamlit_option_menu import option_menu
from sklearn.utils import resample
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from yellowbrick.classifier import ROCAUC
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import pickle
from libs.cleaner import preprocessing_comment
from PIL import Image
# comment
# Header
st.title("Data Science Project")
# Read data
data = pd.read_csv('Shopee_MenFashion_DataPre.csv')

# Split Data
df_train = data.sample(frac = 0.7)
df_test = data.drop(df_train.index)
# Resampling
data_positive = df_train[df_train['label'] == 'Positive']
data_neutral = df_train[df_train['label'] == 'Neutral']
data_negative  = df_train[df_train['label'] == 'Negative']

# Resample Positive Class
data_positive_resample = resample(data_positive,
                                replace = True,
                                n_samples = int(data_positive.shape[0]*0.7),
                                random_state = 42)
# Resample Neutral Class
data_neutral_resample = resample(data_neutral,
                                replace = True,
                                n_samples = int(data_positive.shape[0]*0.7),
                                random_state = 42)
# Resample Negative Class
data_negative_resample = resample(data_negative,
                                replace = True,
                                n_samples = int(data_positive.shape[0]*0.7),
                                random_state = 42)

df_train_balanced = pd.concat([data_positive_resample, data_neutral_resample, data_negative_resample])
# Encode data
df_train_balanced['encoded_label'] = df_train_balanced['label'].apply(lambda x: 1 if x == 'Positive' else 2 if x == 'Neutral' else 3 )
df_test['encoded_label'] = df_test['label'].apply(lambda x: 1 if x == 'Positive' else 2 if x == 'Neutral' else 3 )
df_train_balanced.drop('label', axis = 1, inplace = True)
df_test.drop('label', axis = 1, inplace = True)

# Split Data
X_train, y_train = df_train_balanced.drop('encoded_label', axis = 1), df_train_balanced['encoded_label']
X_test, y_test = df_test.drop('encoded_label', axis = 1), df_test['encoded_label']

# TFIDF
tv = pickle.load(open("ML_TFIDF.pkl", "rb"))
X_train_vec = tv.transform(X_train['comment_pre'])
X_test_vec = tv.transform(X_test['comment_pre'])

# Evaluate model
model = pickle.load(open('ML_SentimentAnalysis_Model.pkl', 'rb'))
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
score_train = model.score(X_train_vec, y_train)
score_test = model.score(X_test_vec, y_test)
precision = precision_score(y_test, y_pred, average = 'macro')
recall = recall_score(y_test, y_pred, average = 'macro')
f1 = f1_score(y_test, y_pred, average = 'macro')

# GUI
menu = ["Company", "About Project", "Evaluation", "Prediction"]
choice = option_menu(
    menu_title = None,
    options= menu,
    menu_icon= 'cast',
    orientation= 'horizontal'
)

if choice == "Company":
    
    # Shopee Information
    st.write("### 1.What Is Shopee?")
    shopee = Image.open("Images/shopee_img.jpg")
    st.image(shopee, width = 700)
    st.write("""##### Shopee is a tech-forward, consumer-focused eCommerce platform in Southeast Asia that has grown in popularity in the Philippines, Singapore, Malaysia, and elsewhere. It aims to give access to a vast array of products and, much like eBay or Amazon in North America, allows both individual sellers and established businesses to sell on their platform. The company first launched in seven countries across the region, including Singapore, Malaysia, Thailand, Taiwan, Indonesia, Vietnam, and the Philippines. Now, it is also used in select countries in Latin America and Europe and has a new presence in India.Shopee originally began as a marketplace, offering consumer-to-consumer (C2C) transactions. It has since transitioned to a hybrid business model that caters to both C2C and business-to-consumer (B2C) transactions.""")
    
    # How Shopee Work
    st.write("### 2.How Does Shopee Work?")
    shopee_work = Image.open("Images/shopee_work.png")
    st.image(shopee_work, width = 700)
    
    # Why Shopee Popular
    st.write("##### Shopee works similarly to other online retailers in that shoppers can enter what they’re looking for into a search engine or browse by category. If a customer is looking for discounts, the platform makes them easy to find by showcasing flash sales and exclusive deals of the day on their homepage. Shoppers can also shop by region. Once on the main page, shoppers can choose their country of residence and be automatically redirected to the storefront that services their country and currency.")
    st.write("### 3.Why Is Shopee So Popular?")
    shopee_popular = Image.open("Images/shopee-popular.jpg")
    st.image(shopee_popular, width = 700)
    st.write("##### Shopee is largely popular due to its social-first, mobile-centric approach, catering to the digital age we live in. Like many startups, Shopee uses social media to expand its reach and attain new customers. The company works with influential brand ambassadors on Instagram and YouTube to increase awareness and offers retailers social media advertising programs. The Shopee app (available on iPhone and Android) also offers real-time shopping and a variety of payment methods.")
elif choice == "About Project":
    # Sentiment Analysis
    st.write("### 1. Sentiment Analysis")
    sentiment_img = Image.open("Images/sentiment_analysis.jpg")
    st.image(sentiment_img, width = 700)
    st.write("##### Sentiment analysis is a type of text research aka mining. It applies a mix of statistics, natural language processing (NLP), and machine learning to identify and extract subjective information from text files, for instance, a reviewer’s feelings, thoughts, judgments, or assessments about a particular topic, event, or a company and its activities as mentioned above. This analysis type is also known as opinion mining (with a focus on extraction) or affective rating.")
    
    # Shopee User Comments Analysis
    st.write("### 2. Shopee User Comments Analysis")
    comment1 = Image.open('Images/comment1.png')
    comment2 = Image.open('Images/comment2.png')
    comment3 = Image.open('Images/comment3.png')
    comment4 = Image.open('Images/comment4.png')
    st.image([comment1, comment2, comment3, comment4], width = 700)
    st.write("##### Shopee is a leading e-commerce website in Vietnam and Southeast Asia. Shopee allows users to easily view product information, reviews, comments and purchases. From customer reviews, the problem is how to make Shopee.vn stores better understand customers, know how they evaluate the store to better improve products/services .")

    # Information of data
    st.write("### 3. EDA")
    
    st.write("#### Some Rows Of Dataset")
    st.dataframe(data.head(), width=700)
    
    # Mean lenth of each class
    st.write('#### Mean Length Of Each Class')
    mean_length = data.groupby('label',).mean('length').round(2)
    st.dataframe(mean_length,width = 700)
    fig, ax = plt.subplots()
    ax = sns.barplot(data = mean_length, x = mean_length.index, y = 'length', palette='Spectral')
    plt.title('Average Length Of Each Type Of Comment')
    plt.xlabel('Label')
    plt.ylabel('Length')
    st.pyplot(fig)
    
    # WordCloud of each class
    st.write("#### WordCloud Of Each Class")
    
    # Positive wordcloud
    st.write("##### Positive")
    text = " ".join(i for i in data[data['label'] == 'Positive'].dropna().comment_pre.values)
    wordcloud = WordCloud( background_color="white", max_words= 30, collocations = False).generate(text)
    fig = plt.figure( figsize=(10,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('WordCloud of Positive Class', fontsize = 16)
    plt.axis("off")
    plt.show()
    st.pyplot(fig, width = 700)
    
    # Neutral wordcloud
    st.write("#### Neutral")
    text = " ".join(i for i in data[data['label'] == 'Neutral'].dropna().comment_pre.values)
    wordcloud = WordCloud( background_color="white", max_words= 30, collocations = False).generate(text)
    fig = plt.figure( figsize=(10,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('WordCloud of Neutral Class', fontsize = 16)
    plt.axis("off")
    plt.show()
    st.pyplot(fig, width = 700)
    
    # Negative wordcloud
    st.write("##### Negative")
    text = " ".join(i for i in data[data['label'] == 'Negative'].dropna().comment_pre.values)
    wordcloud = WordCloud( background_color="white", max_words= 30, collocations = False).generate(text)
    fig = plt.figure( figsize=(10,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('WordCloud of Negative Class', fontsize = 16)
    plt.axis("off")
    plt.show()
    st.pyplot(fig, width = 700)
    
    # Imbalance Data
    st.write("#### Resampling Imablanced Data")
    
    st.write("##### Before Reampling")
    num_cate = pd.DataFrame(data.label.value_counts())
    # Visualize number of values each category
    fig, ax = plt.subplots()
    ax = sns.barplot(data = num_cate, x = num_cate.index, y = 'label', palette='flare')
    plt.title('Number of Values Each Category')
    plt.xlabel('Label')
    plt.ylabel('Values')
    st.pyplot(fig, width = 700)

    st.write("##### After Resampling")
    num_cate = pd.DataFrame(df_train_balanced.label.value_counts())

    # Visualize number of values each category
    fig, ax = plt.subplots()
    ax = sns.barplot(data = num_cate, x = num_cate.index, y = 'label', palette="blend:#7AB,#EDA")
    plt.title('Number of Values Each Category')
    plt.xlabel('Label')
    plt.ylabel('Values')
    st.pyplot(fig, width = 700)
    
    
elif choice == 'Evaluation':
    st.write("### 1.Accuray, Precision, Recall, F1-Score")
    st.code("Accuracy: " + str(round(accuracy,2)))
    st.code("Score train: " + str(round(score_train,2)) + " vs Score test: " +str(round(score_test,2)))
    st.code("Precision: " + str(round(precision,2)))
    st.code("Recall: " + str(round(recall,2)))
    st.code("F1-Score: " + str(round(f1,2)))
    
    st.write("#### 2. ROC curve")
    visualizer = ROCAUC(model, )
    visualizer.fit(X_train_vec, y_train)
    visualizer.score(X_test_vec, y_test)
    st_yellowbrick(visualizer)
    
    st.write("### 3.Confusion Matrix")
    st.write("#### Classification Report")
    st.code(classification_report(y_test, y_pred))
    st.write("#### Heat Map")
    fig, ax= plt.subplots()
    ax = sns.heatmap(confusion_matrix(y_test, y_pred), annot= True,fmt ='d', cmap = plt.cm.Blues)
    plt.show()
    st.pyplot(fig)

    df_prediction = df_test.copy()
    df_prediction['prediction'] = y_pred

    st.write("### WordCloud")
    
    st.write("#### Positive")
    text = " ".join(i for i in df_prediction[df_prediction['prediction'] == 1].dropna().comment_pre.values)
    wordcloud = WordCloud( background_color="white", max_words= 30, collocations = False).generate(text)
    fig= plt.figure(figsize = (10,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('WordCloud of Positive Class', fontsize = 16)
    plt.axis("off")
    plt.show()
    st.pyplot(fig)
    
    st.write("#### Neutral")
    text = " ".join(i for i in df_prediction[df_prediction['prediction'] == 2].dropna().comment_pre.values)
    wordcloud = WordCloud( background_color="white", max_words= 30, collocations = False).generate(text)
    fig= plt.figure(figsize = (10,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('WordCloud of Neutral Class', fontsize = 16)
    plt.axis("off")
    plt.show()
    st.pyplot(fig)
    
    st.write("#### Negative")
    text = " ".join(i for i in df_prediction[df_prediction['prediction'] == 3].dropna().comment_pre.values)
    wordcloud = WordCloud( background_color="white", max_words= 30, collocations = False).generate(text)
    fig= plt.figure(figsize = (10,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title('WordCloud of Negative Class', fontsize = 16)
    plt.axis("off")
    plt.show()
    st.pyplot(fig)
elif choice == "Prediction":
    st.subheader("What Do You Want?")
    lines = None
    option = st.selectbox("",options = ("Input A Comment", "Upload A File"))
    if option == "Input A Comment":
        st.write("Your Selected: ", option)
        comment = st.text_area("Type Your Comment: ")
        if comment != "":
            comment_pre = preprocessing_comment(comment)
            lines = np.array([comment_pre])
            tfidf_comment = tv.transform(lines)
            y_pred_new = model.predict(tfidf_comment)
            if y_pred_new == 1:
                positive = Image.open("Images/positive.png")
                positive = positive.resize((600,600))
                st.image(positive, width = 700)
            elif y_pred_new == 2:
                neutral = Image.open("Images/neutral.png")
                neutral = neutral.resize((600,600))
                st.image(neutral, width = 700)
            else:
                negative = Image.open("Images/negative.png")
                negative = negative.resize((600,600))
                st.image(negative, width = 700)
    if option == "Upload A File":
        st.write("Your Selected: ", option)
        # Upload file
        uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file_1 is not None:
            df = pd.read_csv(uploaded_file_1, header = None)
            df = df.iloc[1:,:]
            st.dataframe(df)
            lines = df[0].apply(lambda x: preprocessing_comment(x))
            st.dataframe(lines)
            x_new = tv.transform(lines)        
            y_pred_new = model.predict(x_new)
            df['prediction'] = y_pred_new
            st.dataframe(df)

        


