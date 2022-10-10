import pandas as pnd
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
mensajesTwitter = pnd.read_csv("datas/calentamientoClimatico.csv",delimiter=";") 

print(mensajesTwitter.shape)
print(mensajesTwitter.head(2)) 

mensajesTwitter['CREENCIA'] = (mensajesTwitter['CREENCIA']=='Yes').astype(int)

print(mensajesTwitter.head(100)) 


def normalizacion(mensaje): 
    mensaje = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL', mensaje)
    mensaje = re.sub('@[^\s]+','USER', mensaje)
    mensaje = mensaje.lower().replace("ё", "е")
    mensaje = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', mensaje)
    mensaje = re.sub(' +',' ', mensaje)
    return mensaje.strip() 

text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL', text) 

mensajesTwitter["TWEET"] = mensajesTwitter["TWEET"].apply(normalizacion)
print(mensajesTwitter.head(10)) 

stopWords = stopwords.words('english') 

mensajesTwitter['TWEET'] = mensajesTwitter['TWEET'].apply(lambda  mensaje: ' '.join ([palabra for palabra in mensaje.split() if palabra not in (stopWords)]))
print(mensajesTwitter.head(10)) 

stemmer = SnowballStemmer('english')
mensajesTwitter['TWEET'] = mensajesTwitter['TWEET'].apply(lambda mensaje: ' '.join([stemmer.stem() for palabra in mensaje.split(' ')]))
print(mensajesTwitter.head(10)) 

lemmatizer = WordNetLemmatizer()
mensajesTwitter['TWEET'] = mensajesTwitter['TWEET'].apply(lambda mensaje: ' '.join([lemmatizer.lemmatize(palabra) for palabra in mensaje.split(' ')])) 

X_train, X_test, y_train, y_test = train_test_split(mensajesTwitter['TWEET'].values, mensajesTwitter['CREENCIA'].values,test_size=0.2) 


etapas_aprendizaje = Pipeline([('frequence', CountVectorizer()), ('tfidf', TfidfTransformer()), ('algoritmo',MultinomialNB())]) 

modelo = etapas_aprendizaje.fit(X_train,y_train)
print(classification_report(y_test, modelo.predict(X_test), digits=4)) 

frase = "Why should trust scientists with global warming if they didnt know Pluto wasnt a planet"
print(frase)

#Normalización  
frase = normalizacion(frase)

#Eliminación de las stops words  
frase = ' '.join([palabra for palabra in frase.split() 
if palabra not in(stopWords)])

#Aplicación de stemming  
frase = ' '.join([stemmer.stem() for palabra in frase.split(' ')])

#Lematización  
frase = ' '.join([lemmatizer.lemmatize(palabra) for palabra in frase.split(' ')])
print (frase)

prediccion = modelo.predict([frase])
print(prediccion)
if(prediccion[0]==0):
    print(">> No cree en el calentamiento climático...")
else:
    print(">> Cree en el calentamiento climático...") 

parameters = {'algoritmo__C':(1,2,4,5,6,7,8,9,10,11,12)}
clf = GridSearchCV(etapas_aprendizaje, parameters,cv=2)
clf.fit(X_train,y_train)
print(clf.best_params_) 

etapas_aprendizaje = Pipeline([('frequence',CountVectorizer()),('tfidf', TfidfTransformer()),('algoritmo', SVC(kernel='linear', C=1))])

modelo = etapas_aprendizaje.fit(X_train,y_train)

print(classification_report(y_test, modelo.predict(X_test), digits=4)) 

