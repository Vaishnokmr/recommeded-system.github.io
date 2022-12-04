from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
## Movie 
movie_dict = pickle.load(open('content_recommend\movie.pkl','rb'))
movies = pd.DataFrame(movie_dict)
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similar = cosine_similarity(vectors)


@app.route("/")
def hello_world():
    mov = movies['title'].values
    return render_template('index.html',mov= mov )

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        mov = movies['title'].values
        data= request.form['rest_type']
        movie_index = movies[movies['title']== data].index[0]
        dist = similar[movie_index]
        movie_list = sorted(list(enumerate(dist)),reverse=True, key=lambda x:x[1])[1:6]
        movie_name = []
        for i in movie_list:
            movie_name.append(movies.iloc[i[0]].title)
        return render_template('index.html',select_movie =data ,text = "Recommended Movie",prediction_text= movie_name,mov= mov)
    else:
        return render_template('index.html',mov= mov)

if __name__ =="__main__":
    app.run(debug=True)