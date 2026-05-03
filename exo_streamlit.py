import streamlit as st
import pandas as pd
import pickle

# --- CHARGEMENT DU MODÈLE ---
# Assurez-vous que le chemin est correct pour votre machine
try:
    with open('modeliris6.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Le fichier modèle 'modeliris6.pkl' n'a pas été trouvé. Vérifiez le chemin.")

st.title("MSDE: ML COURSE")
st.header("IRIS FLOWER PREDICTION APP")
st.subheader("THIS APP PREDICTS THE IRIS FLOWER TYPE")

options = st.multiselect(
    'HOW WOULD YOU LIKE TO USE THE PREDICTION MODEL?', 
    ['Input parameters directly', 'Load a file or data']
)

# --- OPTION 1 : SAISIE MANUELLE ---
if 'Input parameters directly' in options:
    st.sidebar.header("User Input Parameters")
    sepal_length = st.sidebar.slider('Sepal length', 4.0, 8.0, 6.17)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 5.0, 2.56)
    petal_length = st.sidebar.slider('Petal length', 1.0, 7.0, 2.66)
    petal_width = st.sidebar.slider('Petal width', 0.1, 3.0, 2.15)
    
    data = {
        'sepal_length': [sepal_length],
        'sepal_width': [sepal_width],
        'petal_length': [petal_length],
        'petal_width': [petal_width]
    }
    df = pd.DataFrame(data)
    
    st.write("### Selected Parameters")
    st.table(df)

    st.write("### Prediction Results")
    if st.button('Predict'):
        prediction = model.predict(df)
        prediction_proba = model.predict_proba(df)
        st.success(f"The predicted flower type is: **{prediction[0]}**")
        
        st.write("#### Prediction Probability")
        df_proba = pd.DataFrame(prediction_proba, columns=model.classes_)
        st.dataframe(df_proba)

# --- OPTION 2 : CHARGEMENT DE FICHIER ---
if 'Load a file or data' in options:
    st.write("---")
    st.write("### Bulk Prediction from File")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Lecture correcte pour iris_test.csv (sans header)
        input_df = pd.read_csv(uploaded_file, header=None, names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        
        st.write("#### Preview of uploaded data")
        st.dataframe(input_df.head())
        
        if st.button('Predict from File'):
            try:
                predictions = model.predict(input_df)
                
                # Création du tableau de résultats
                result_df = input_df.copy()
                result_df['Predicted_Species'] = predictions
                
                st.write("#### Prediction Results")
                st.dataframe(result_df)
                
                # Bouton de téléchargement
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results as CSV", data=csv, file_name="predictions.csv", mime="text/csv")
                
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")

# Message par défaut si rien n'est sélectionné
if not options:
    st.sidebar.info("Please select a method in the multiselect box to start.")
