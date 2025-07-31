
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Title and Description
st.title("AURA: Advancing Sustainable Business Decisions through Adaptive Utility Ranking Algorithm")
st.markdown('''This application uses the **AURA** method to rank alternatives based on multiple criteria, such as sustainability metrics.
You can upload a dataset with alternative names and their respective scores for each criterion.
The app will normalize the data, apply AURA, and compute rankings for the alternatives.''')

# File upload
uploaded_file = st.file_uploader("Upload your Excel/CSV file", type=["xlsx", "csv"])

# Allow users to input weights for criteria
weights_input = st.text_input("Enter weights for each criterion (comma separated)", "0.2,0.3,0.5")
weights = list(map(float, weights_input.split(',')))

# Default values for alpha
alpha_input = st.slider("Set the Alpha (α) Parameter", 0.0, 1.0, 0.5)

if uploaded_file:
    # Load the dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Clean up the dataset
    df_cleaned = df.dropna(axis=1, how='all').dropna(axis=0, how='all')

    # Extract the alternatives and criteria
    alternatives = df_cleaned.iloc[:, 0].values
    criteria = df_cleaned.columns[1:]

    # Normalize the matrix using vector normalization
    normalized_matrix = df_cleaned.iloc[:, 1:].apply(lambda x: x / np.linalg.norm(x), axis=0)

    # Compute the weighted normalized matrix
    weighted_matrix = normalized_matrix * weights

    # Determine the Ideal Solution (PIS), Negative Ideal Solution (NIS), and Average Solution (AS)
    PIS = weighted_matrix.max(axis=0)
    NIS = weighted_matrix.min(axis=0)
    AS = weighted_matrix.mean(axis=0)

    # Adjust for cost criteria (minimize instead of maximize)
    for i in range(len(criteria)):
        if "Cost" in criteria[i]:  # You can adjust this based on how the user defines cost criteria
            PIS[i] = weighted_matrix.min(axis=0)[i]
            NIS[i] = weighted_matrix.max(axis=0)[i]

    # Compute the Euclidean distance from PIS, NIS, and AS
    distance_to_PIS = np.sqrt(((weighted_matrix - PIS) ** 2).sum(axis=1))
    distance_to_NIS = np.sqrt(((weighted_matrix - NIS) ** 2).sum(axis=1))
    distance_to_AS = np.sqrt(((weighted_matrix - AS) ** 2).sum(axis=1))

    # Compute relative closeness to the ideal solution
    closeness_scores_PIS = distance_to_NIS / (distance_to_PIS + distance_to_NIS)
    closeness_scores_AS = distance_to_AS / (distance_to_PIS + distance_to_AS)

    # Rank the alternatives based on the closeness score
    df_cleaned['Closeness Score (PIS)'] = closeness_scores_PIS
    df_cleaned['Closeness Score (AS)'] = closeness_scores_AS
    df_cleaned['Rank (PIS)'] = df_cleaned['Closeness Score (PIS)'].rank(ascending=False)
    df_cleaned['Rank (AS)'] = df_cleaned['Closeness Score (AS)'].rank(ascending=False)

    # Display the results
    st.subheader("Normalized Matrix")
    st.write(normalized_matrix)

    st.subheader("Weighted Normalized Matrix")
    st.write(weighted_matrix)

    st.subheader("Final Rankings Based on PIS and AS")
    st.write(df_cleaned[['Closeness Score (PIS)', 'Rank (PIS)', 'Closeness Score (AS)', 'Rank (AS)']].sort_values(by='Rank (PIS)'))

    # Visualize the rankings
    fig = px.bar(df_cleaned, x='Rank (PIS)', y='Closeness Score (PIS)', color='Rank (PIS)',
                 labels={'Rank (PIS)': 'Alternatives', 'Closeness Score (PIS)': 'Closeness to Ideal Solution'})
    st.plotly_chart(fig)

    # Option to download the results
    @st.cache
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(df_cleaned)
    st.download_button(
        label="Download Results",
        data=csv,
        file_name='ranking_results.csv',
        mime='text/csv',
    )
else:
    st.info("Please upload a dataset to begin.")
