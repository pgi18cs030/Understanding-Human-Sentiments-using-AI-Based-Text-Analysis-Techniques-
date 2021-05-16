import pickle


clean_review=[]
prediction=[]


def clean_data(data):
    clean_review.append(' '.join(re.sub('[^\w\s\.]',' ',data).split()))



def preprocessing_data():
    df=pd.read_csv('scrappedReviews.csv')
    for review in df['reviews']:
        clean_data(review)

if __name__=='__main__':
    preprocessing_data()
