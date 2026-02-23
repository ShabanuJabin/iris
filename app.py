import streamlit as st
import pickle
from PIL import Image

def main():
    st.title("Iris")

    img = Image.open('iris.png')
    st.image(img, width=400)

    # Use number_input (better for ML)
    sepal_length = st.number_input("Sepal length:", min_value=0.0)
    sepal_width = st.number_input("Sepal width:", min_value=0.0)
    petal_length = st.number_input("Petal length:", min_value=0.0)
    petal_width = st.number_input("Petal width:", min_value=0.0)

    model = pickle.load(open('model_iris.sav','rb'))
    scaler = pickle.load(open('scaler_iris.sav','rb'))

    if st.button("Predict"):
        features = [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]]

        result = model.predict(scaler.transform(features))

        if result[0] == 'Iris-setosa':
            st.write("# Setosa flower")
        elif result[0] == 'Iris-versicolor':
            st.write("# Versicolor flower")
        else:
            st.write("# Virginica flower")

    # Simple Clear (reload page)
    if st.button("Clear"):
        st.rerun()

main()