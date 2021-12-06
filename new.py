"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
##  Loading Libraries & Dependencies

# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
from PIL import Image

import numpy as np        # Fundamental package for linear algebra and multidimensional arrays
import pandas as pd       # Data analysis and manipulation tool

import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



# Vectorizer
news_vectorizer = open("resources/vect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	#st.title("Climate Change Belief tweet Classifer")
	#image = Image.open('Climatechange.jpg')
	#st.image(image, caption='World Climatechange')
	#st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Home","Introduction to EDA","EDA","Sentiments","Prediction"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Home" page
	if selection == "Home":
		st.title("Climate Change Belief tweet Classifer")
		image = Image.open('Climatechange.jpg')
		st.image(image, caption='World Climatechange')
		st.header("Introduction")
		st.info("This APP is designed to help companies make informed decisions when it comes to climate change. Several companies are built around lessening one’s environmental impact or carbon footprint. This is because they offer products and services that are environmentally friendly and sustainable, in line with their values and ideals. They would like to determine how people perceive climate change and whether or not they believe it is a real threat. This would add to their market research efforts in gauging how their product/service may be received.")
		# You can read a markdown file from supporting resources folder
		st.info("With this context, we have created a Machine Learning model that is able to classify whether or not a person believes in climate change, based on their novel tweet data. This model together will the accompanying app will help Geo-Environmental Consultation companies who are turning to social media to obtain valuable information about job applicants and to monitor the activities of their employees in relation to the values they have towards the company's projects and beliefs surrounding the ever changing global environment.")

	if selection == "Introduction to EDA":
		st.header("Exploratory Data Analysis (EDA)")
		image = Image.open('EDA.jpg')
		st.image(image)
		st.subheader("What is Exploratory Data Analysis?")
		st.markdown("Exploratory Data Analysis (EDA) is an approach/philosophy for data analysis that employs a variety of techniques (mostly graphical) to:")
		st.text("(I)	Maximize insight into a data set")
		st.text("(II)	Uncover underlying structure")
		st.text("(III)	Extract important variables")
		st.text("(IV)	Detect outliers and anomalies")
		st.text("(V)	Test underlying assumptions")
		st.text("(VI)	Develop parsimonious models")
		st.text("(VII)	Determine optimal factor settings")
		

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models") 
	

		# Creating a text box for user input
		tweet_text =st.text_area("Enter Text","Type Here")

		if st.button("Classify with Logistic Regression Model"):
			# Transforming user input with vectorizer
			vect_text =tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/log.pkl"),"rb"))
			prediction = predictor.predict(vect_text)
			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

		if st.button("Classify with Decision Tree Model"):
			# Transforming user input with vectorizer
			vect_text =tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/dt.pkl"),"rb"))
			prediction = predictor.predict(vect_text)
			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))

		if st.button("Classify with K-Nearest Neighbors Model"):
			# Transforming user input with vectorizer
			vect_text =tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/rf.pkl"),"rb"))
			prediction = predictor.predict(vect_text)
			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))
	
	
	# Building out the EDA page
	if selection == "EDA":
		st.subheader("Raw Twitter data and label")
		
		#if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			#st.write(raw) # will write the df to the page
			
			#st.subheader("Sentiments recorded")
			#Distribution = pd.DataFrame(raw['sentiment'].value_counts())
			#st.bar_chart(Distribution)
		if 'number_of_rows' not in st.session_state or 'type' not in st.session_state:
			st.session_state['number_of_rows'] = 5 
			st.session_state['type'] = 'Categorical'
			

		increament = st.button('Show more columns👆')
		if increament:
			st.session_state.number_of_rows = 10


		decrement = st.button('Show fewer columns👇')
		if decrement:
			st.session_state.number_of_rows = 2


		st.table(raw.head(st.session_state['number_of_rows']))

		#types = {'Categorical':['sentiment'], 'Numerical':[]}
		#column = st.selectbox('select a column', types[st.session_state['type']])

		def handle_click(new_type):
				st.session_state.type = new_type

		def handle_click_wo():
			if st.session_state.kind_of_column:
				st.session_state.type = st.session_state.kind_of_column



		type_of_column = st.radio("What kind of analysis",['Categorical','Numerical'], on_change = handle_click_wo,key = 'kind_of_column')
		#change == st.button('Change', on_click=handle_click,args = [type_of_column])
		if st.session_state['type'] =='Categorical':
			Distribution =  pd.DataFrame(raw['sentiment'].value_counts())
			st.bar_chart(Distribution)
		else:
			st.subheader("No Numerical Data")

		st.subheader("Percentage distribution of lables")
		st.text("Figure below shows the percentage distribution of lables for the given data")
		working_df = raw.copy()
		# Labeling the target
		working_df['sentiment'] = [['Negative', 'Neutral', 'Positive', 'News'][x+1] for x in working_df['sentiment']]
		sizes = working_df['sentiment'].value_counts()/working_df.shape[0]
		labels = (working_df['sentiment'].value_counts()/working_df.shape[0]).index
		explode = (0.1, 0.1, 0.1, 0.1)
		fig1, ax1 = plt.subplots()
		ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
		ax1.axis('equal')
		st.pyplot(fig1)

		st.subheader("most popular messages")
		st.text("Table below shows the most popular messages for the given data")
		sel_col,disp_col = st.columns(2)
		choose = sel_col.slider('what number of rows would you like to view?',min_value = 5,max_value = 100,value = 20, step = 5)
		working_df['users'] = [''.join(re.findall(r'@\w{,}', line)) 
        if '@' in line else np.nan for line in working_df.message]
		counts = working_df[['message','users']].groupby('users',as_index=False).count().sort_values(by='message', ascending=False)
		st.table(counts.head(choose))

		st.subheader("Numerical distribution of hashtags")
		st.text("Figure below shows the numerical distribution of hashtags for the given data")
		working_df['hashtags'] = [' '.join(re.findall(r'#\w{,}', line)) 
        if '#' in line else np.nan for line in working_df.message]
		sizes = [sum(np.array(counts['message']) == 1)/len(counts['message']),
        sum(np.array(counts['message']) != 1)/len(counts['message'])]
		labels = ['First Time Tags', 'Repeated Tags']
		explode = (0.1, 0.1)
		fig1, ax1 = plt.subplots()
		ax1.pie(sizes, labels=labels, explode=explode,autopct='%1.1f%%',shadow=True, startangle=90)
		ax1.axis('equal')
		st.pyplot(fig1)

		#wordscloud(write a code that will make the use chose the cloud they would like to see)
		
		
		
		#type_of_sentiment  = st.radio("Which sentiment would you like to see?",['Positive','Negative','Neutral','News'])

		#option = ['Positive','Negative','Neutral','News']
	if selection == "Sentiments":
		st.subheader("Sentiment Clouds")
		st.markdown("The tweets sentiments are divided into four (4) classes:")
		st.text("[ 2 ] - News: tweets links to factual news about climate change")
		st.text("[ 1 ] - Pro: tweets that supports the belief of man-made climate change")
		st.text("[ 0 ] - Neutral: tweets that neither support nor refutes the belief of man-made climate change")
		st.text("[-1 ] - Anti (Negative): tweets that does not believe in man-made climate change")
		st.markdown("Figure below shows the word clouds to the most common words associated with the chosen sentiment for the given data")
		if st.button('Positive Sentiment'):
			#Positive
			df_pos = raw.loc[raw['sentiment'] == 1] #positive reviews
			pos_string = df_pos['message'].str.cat(sep = ' ')
			wordcloud = WordCloud(background_color='white').generate(pos_string)
			# Display the generated image:
			fig1, ax1 = plt.subplots()
			ax1.imshow(wordcloud, interpolation='bilinear')
			ax1.axis("off")
			Positive = st.pyplot(fig1)

		if st.button('Negative Sentiment'):
			#Negative
			df_neg = raw.loc[raw['sentiment'] == -1] #negative reviews
			neg_string = df_neg['message'].str.cat(sep = ' ')
			wordcloud1 = WordCloud(background_color='white').generate(neg_string)
			# Display the generated image:
			fig2, ax2 = plt.subplots()
			ax2.imshow(wordcloud1, interpolation='bilinear')
			ax2.axis("off")
			Negative = st.pyplot(fig2)

		if st.button('Neutral Sentiment'):
			#Neutral
			df_neu = raw.loc[raw['sentiment'] == -1] #neutral reviews
			neu_string = df_neu['message'].str.cat(sep = ' ')
			wordcloud2 = WordCloud(background_color='white').generate(neu_string)
			# Display the generated image:
			fig3, ax3 = plt.subplots()
			ax3.imshow(wordcloud2, interpolation='bilinear')
			ax3.axis("off")
			Negative = st.pyplot(fig3)

		if st.button('News Sentiment'):
			#News
			df_new = raw.loc[raw['sentiment'] == -1] #news reviews
			new_string = df_new['message'].str.cat(sep = ' ')
			wordcloud3 = WordCloud(background_color='white').generate(new_string)
			# Display the generated image:
			fig4, ax4 = plt.subplots()
			ax4.imshow(wordcloud3, interpolation='bilinear')
			ax4.axis("off")
			Negative = st.pyplot(fig4)

		
	


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
