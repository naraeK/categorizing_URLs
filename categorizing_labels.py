import urllib.request
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import *
from sklearn.metrics import confusion_matrix


def category_URL_zip(url, month_list):
    month_pair = [] # the output of the function
    for month in month_list:
        # only for this assignment
        new_URL = url.replace('index','month-'+month+'-2017')
        # category list for each month
        category_list = []
        response = urllib.request.urlopen(new_URL)
        text = response.read().decode()
        soup = BeautifulSoup(text,'html.parser')
        # category list for each month
        for raw_article in soup.findAll('td', {'class': 'category'}):
            category = raw_article.string.strip()
            if category != 'N/A':
                category_list.append(str(category))
        # URL list for each month
        subURL_list = []
        for raw_article in soup.findAll('a'):
            URL = raw_article.get('href')
            subURL_list.append(str(URL))
        subURL_list = subURL_list[:len(subURL_list)-4]  # last-four elements of every monthly article lists include "footer" in <a>
        zipped = list(zip(category_list,subURL_list))
        month_pair.extend(zipped) # entire pair over all articles
    return month_pair

# download the content
url = 'http://mlg.ucd.ie/modules/COMP41680/archive/index.html'
month_list = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']

# [(catagory1,URL1),(catagory2,URL2),(catagory3,URL3),...]
category_URL_pair = category_URL_zip(url , month_list)
#print(category_URL_pair[0:10])

def extracting_article(category, URL):
    url = "http://mlg.ucd.ie/modules/COMP41680/archive/" + URL  # each article URL
    response = urllib.request.urlopen(url)
    text = response.read().decode()
    # new text file name by their URL
    file_title = URL + '.txt'

    soup = BeautifulSoup(text, 'html.parser')
    title = soup.find('h2').string  # the title of each article
    subtitle = soup.find('b')  # .string # the sub-title of each article
    # creating a new file for each article
    fout = open(file_title, "w")
    fout.write("category: " + str(category) + "\n")  # category for the first line
    fout.write("title: " + str(title) + "\n")  # title for the second line
    fout.write("subtitle: " + str(subtitle) + "\n")  # subtitle for the third line
    for raw_article in soup.findAll('p'):
        if len(raw_article) >= 1:
            if raw_article == soup.find('p', {'class': 'notice'}):  # not a body text
                break
            bodyText = raw_article.string  # .strip()
            bodyText = str(bodyText)  # as plain text
            fout.write(bodyText)  # body text for fourth line
    fout.close()


for c, U in category_URL_pair:
    try:
        extracting_article(c,U) # extracting the article into a new text file
    except UnicodeEncodeError:
        print("Error in", U)
    except TimeoutError:
        print("Time is over. Try it again.")


fout = open('category labels - URL.txt', "w")
for c, U in category_URL_pair:
    fout.write("category: "+ c +"- URL: "+ U +"\n")
fout.close()

def Load_rawDocument(U):
    fin = open(U+'.txt' , 'r')
    raw_documents = fin.readlines()
    fin.close()
    # category = raw_document[0] is not called now.
    # body text is written in the fourth line which is [3] index
    bodyText = raw_documents[3].lower() # convert to lowercase as a part of pre-processing
    return bodyText


# Separate the articles into the appropriate category as one single element
technology_raw_contents = []
business_raw_contents = []
sport_raw_contents = []
for c, U in category_URL_pair:
    bodyText = Load_rawDocument(U) # from the defined function
    if c == 'technology':
        technology_raw_contents.append(bodyText)
    elif c == 'business':
        business_raw_contents.append(bodyText)
    elif c == 'sport':
        sport_raw_contents.append(bodyText)
    else:
        print("This is not defined:", U)

#print(len(technology_raw_contents)) # the number of articles in the technology category
#print(len(business_raw_contents)) # the number of articles in the business category
#print(len(sport_raw_contents)) # the number of articles in the sport category

CVtokenize = CountVectorizer().build_tokenizer()
stopwords = text.ENGLISH_STOP_WORDS

def tokenizing(raw_bodyText):
    # convert to lowercase, then tokenize
    tokens = CVtokenize(bodyText.lower())# before filtering
    filtered_tokens = []
    for token in tokens:
        if not token in stopwords: # pre-processing: removing meaningless words in English
            filtered_tokens.append(token)
    return filtered_tokens

technology_filtered = [] # technology category - each element is each article
business_filtered = [] # business category - each element is each article
sport_filtered = [] # sport category - each element is each article
for c, U in category_URL_pair:
    bodyText = Load_rawDocument(U) # from the defined function
    filtered_tokens = tokenizing(bodyText)
    if c == 'technology':
        technology_filtered.append(filtered_tokens)
    elif c == 'business':
        business_filtered.append(filtered_tokens)
    elif c == 'sport':
        sport_filtered.append(filtered_tokens)
    else:
        print("This is not defined:", U)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(technology_raw_contents) # tokens
#print(X.shape)
#print(list(vectorizer.vocabulary_.keys())[0:35])


# define the function
def stem_tokenizer(text):
    # use the standard scikit-learn tokenizer first
    standard_tokenizer = CountVectorizer().build_tokenizer()
    tokens = standard_tokenizer(text)
    # then use NLTK to perform stemming on each token
    stemmer = PorterStemmer()
    stems = []
    for token in tokens:
        stems.append( stemmer.stem(token) )
    return stems

# Examples
STEMvectorizer = CountVectorizer(stop_words="english",min_df = 3,tokenizer=stem_tokenizer)
'''
X = STEMvectorizer.fit_transform(technology_raw_contents)
Y = STEMvectorizer.fit_transform(business_raw_contents)
Z = STEMvectorizer.fit_transform(sport_raw_contents)
print("The shape of X:",X.shape)
print("The shape of Y:",Y.shape)
print("The shape of Z:",Z.shape)

# display some sample terms
terms = STEMvectorizer.get_feature_names()
vocab = STEMvectorizer.vocabulary_
print(terms[200:220])
print("Vocabulary has %d distinct terms" % len(terms))
'''


# TF-IDF weighting with the preprocessing parameters
TFIDFvectorizer = TfidfVectorizer(stop_words="english",min_df = 3,tokenizer=stem_tokenizer) # "stem_tokenizer" from the pre-processig def


# Example / without train vs. test splitting
'''
X_w = TFIDFvectorizer.fit_transform(technology_raw_contents)
Y_w = TFIDFvectorizer.fit_transform(business_raw_contents)
Z_w = TFIDFvectorizer.fit_transform(sport_raw_contents)

# display some sample weighted values
print(X_w.shape)
'''

# To split the data into train and test models
# data == all article bodytext contents
data =[]
data.extend(technology_raw_contents)
data.extend(business_raw_contents)
data.extend(sport_raw_contents)
print("data length:",len(data)) # len(technology)+len(business)+len(sport): 391+491+526

# target (categories): 0 - technology, 1 - business, 2 - sport
target = [0] * len(technology_raw_contents) + [1] * len(business_raw_contents) + [2] * len(sport_raw_contents)
#print("target length:",len(target))

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2)

# Create document-term matrix from training documents
train_X = TFIDFvectorizer.fit_transform(data_train)
# Create document-term matrix from test documents
test_X = TFIDFvectorizer.transform(data_test)

KNN = KNeighborsClassifier(n_neighbors=3) # k = 3
KNN.fit(train_X, target_train)
KNNpredicted = KNN.predict(test_X)

DTree = DecisionTreeClassifier()
DTree.fit(train_X, target_train)
DTreepredicted = DTree.predict(test_X)

category_accuracy_KNN = accuracy_score(target_test, KNNpredicted)
print("Accuracy of KNN= %.2f" % category_accuracy_KNN ) # seems high enough (> 0.93)

category_accuracy_DTree = accuracy_score(target_test, DTreepredicted)
print("Accuracy of DTree= %.2f" % category_accuracy_DTree ) # seems less high (> 0.85)

print(classification_report(target_test, KNNpredicted, target_names=["technology","business","sport"]))
print(classification_report(target_test, DTreepredicted, target_names=["technology","business","sport"]))

cm_KNN = confusion_matrix(target_test, KNNpredicted, labels=[0,1,2])
print(cm_KNN)


cm_DTree = confusion_matrix(target_test, DTreepredicted, labels=[0,1,2])
print(cm_DTree)

