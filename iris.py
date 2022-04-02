from doctest import Example
import streamlit as st
import pandas as pd
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from PIL import Image



data = sns.load_dataset('iris')
data = data.sample(frac=1).reset_index(drop=True)

training = data [:105]
test = data[105:]

features = list(zip(training['sepal_length'], training['sepal_width'], 
                    training['petal_length'], training['petal_width']))

target = list(training['species'])

k = 7
if st.sidebar.checkbox('Advanced'):
    k =st.sidebar.slider('User input K', 1, 10, 7)



model = KNeighborsClassifier(n_neighbors= k)
model.fit(features,target)



st.title('Welcome to Iris flower species prediction WebApp')
imgHeader = Image.open ("header.png")
st.image(imgHeader)
st.write('This app predicts the **Iris flower** species using **K-nn algorithm**')


#st.dataframe(pd.DataFrame(training.query('species == "setosa"').reset_index().iloc[0]))
setosaEX = pd.DataFrame(training.query('species == "setosa"').reset_index(drop=True).iloc[0]).transpose()
virginicaEX = pd.DataFrame(training.query('species == "virginica"').reset_index(drop=True).iloc[0]).transpose()
versicolorEX = pd.DataFrame(training.query('species == "versicolor"').reset_index(drop=True).iloc[0]).transpose()

examples = pd.concat([setosaEX, versicolorEX, virginicaEX,]).reset_index(drop=True)

st.dataframe(examples)



st.sidebar.header('User input featuers')

sepal_length =st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
petal_length =st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)

inputFeatuers = [sepal_length,sepal_width,petal_length,petal_width]

st.write(f'''## The input features are 
#### Sepal length = {sepal_length} 
#### Sepal width = {sepal_width}
#### Petal length = {petal_length}
#### petal width = {petal_width} ''')

predicted = str(model.predict([inputFeatuers])).strip('[').strip(']').strip("'")


st.write('## Predicted iris flower is: ')
st.write('###',predicted.upper())

if predicted.lower() == 'setosa'.lower():
    setosa = Image.open('setosa.jpg')
    st.image(setosa)
elif predicted.lower() == 'virginica'.lower():
    virginica = Image.open('virginica.jpg')
    st.image('virginica.jpg')
elif predicted.lower() == 'versicolor'.lower():
    versicolor = Image.open('versicolor.jpg')
    st.image('versicolor.jpg')

