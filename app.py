
import streamlit as st
import pandas as pd
import numpy as np

# Title and Description
st.title("AURA: Advancing Sustainable Business Decisions through Adaptive Utility Ranking Algorithm")
st.markdown('''This application allows users to rank business alternatives based on multiple criteria using the Adaptive Utility Ranking Algorithm (AURA). 
It accommodates benefit, cost, and target-type criteria in a unified framework to facilitate decision-making.''')

# Upload decision matrix as an Excel file
st.subheader("Step 1: Upload the Decision Matrix")
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])

if uploaded_file:
    # Read the decision matrix from the Excel file
    df = pd.read_excel(uploaded_file, header=[0, 1], index_col=0)
    st.write("Decision Matrix:")
    st.write(df)

    # Input weights and criteria types (benefit, cost, or target)
    st.subheader("Step 2: Set Weights and Criteria Types")
    weights = st.text_area("Enter weights for the criteria (comma-separated, e.g., 0.2, 0.3, 0.5):")
    weights = np.array([float(w) for w in weights.split(',')])

    criterion_types = []
    for col in df.columns.get_level_values(0):
        criterion_type = st.selectbox(f"Select type for criterion '{col}'", ("Benefit", "Cost", "Target"))
        criterion_types.append(criterion_type)

    # Step 3: Normalize the Decision Matrix
    def normalize_matrix(df, weights, criterion_types):
        rj = []
        for i, crit_type in enumerate(criterion_types):
            if crit_type == 'Benefit':
                rj.append(df.iloc[:, i].max())
            elif crit_type == 'Cost':
                rj.append(df.iloc[:, i].min())
            else:  # Target
                rj.append(df.iloc[:, i].mean())  # Assuming target is the average value

        # Normalize matrix based on the reference values (rj)
        rij = (1 - abs(df - rj)) / (df.max() - df.min())
        return rij

    # Call the normalize function
    normalized_matrix = normalize_matrix(df, weights, criterion_types)
    st.subheader("Step 3: Normalized Matrix")
    st.write(normalized_matrix)

    # Step 4: Weighted Normalized Matrix
    def weighted_normalized_matrix(normalized_matrix, weights):
        return normalized_matrix * weights

    # Apply weights
    weighted_matrix = weighted_normalized_matrix(normalized_matrix, weights)
    st.subheader("Step 4: Weighted Normalized Matrix")
    st.write(weighted_matrix)

    # Step 5: Benchmarking (Positive Ideal, Negative Ideal, Average)
    v_plus = weighted_matrix.max()
    v_minus = weighted_matrix.min()
    v_avg = weighted_matrix.mean()

    st.subheader("Step 5: Benchmarking")
    st.write("Positive Ideal Solution (PIS):")
    st.write(v_plus)
    st.write("Negative Ideal Solution (NIS):")
    st.write(v_minus)
    st.write("Average Solution (AVG):")
    st.write(v_avg)

    # Step 6: Compute Distance to Benchmarks
    def compute_distance(matrix, benchmark, p=2):
        distance = np.sum(np.abs(matrix - benchmark)**p, axis=1)**(1/p)
        return distance

    # Compute distances to PIS, NIS, and AVG
    d_plus = compute_distance(weighted_matrix, v_plus)
    d_minus = compute_distance(weighted_matrix, v_minus)
    d_avg = compute_distance(weighted_matrix, v_avg)

    # Final AURA Score
    alpha = st.slider("Alpha (α) Parameter", 0.0, 1.0, 0.5)
    final_scores = 0.5 * (alpha * (d_plus - d_minus) + (1 - alpha) * d_avg)

    st.subheader("Step 6: Final AURA Scores")
    st.write("Final Scores for Each Alternative:")
    st.write(final_scores)

    # Rank Alternatives
    ranking = final_scores.sort_values(ascending=True)
    st.subheader("Ranking of Alternatives Based on AURA")
    st.write(ranking)

    # Button to download ranking results
    st.download_button(
        label="Download Ranking Results",
        data=ranking.to_csv(),
        file_name="aura_ranking_results.csv",
        mime="text/csv"
    )
