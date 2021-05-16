from wordcloud import WordCloud,STOPWORDS
 
stopwords=set(STOPWORDS)

# WordCloud for Positive
text=open('Positive.txt','r').read()
wc2=WordCloud(background_color='lightgreen',stopwords=stopwords,height=350,width=350)

wc2.generate(text)
wc2.to_file('Poscloud.png')


# WordCloud for Negative
text1=open('Negative.txt','r').read()
wc3=WordCloud(background_color='pink',stopwords=stopwords,height=350,width=350)

wc3.generate(text1)
wc3.to_file('Negcloud.png')


# WordCloud for Netural
text2=open('All.txt','r').read()
wc4=WordCloud(background_color='black',stopwords=stopwords,height=350,width=350)

wc4.generate(text2)
wc4.to_file('All.png')