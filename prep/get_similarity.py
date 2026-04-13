import numpy as np
import pandas as pd
import os

'''
This file contains functions used to calculate cosine similarity based on word embeddings extracted from a distributional semantic model. Code to actually run functions at bottom.
'''
def create_semantic_matrix_items(
    path_to_embeddings: str,
    path_to_vocab: str,
    out_csv: str):
    """
    Adapted from Kumar et al. (2024)

    Creates an NxN semantic similarity matrix from embeddings and saves with word labels.

    Args:
        path_to_embeddings: CSV of embeddings (assumes D×N: rows = dims, cols = words).
        path_to_vocab: CSV with a single column of word labels (length N).
        out_csv: output CSV path for NxN similarity matrix.

    Returns:
        DataFrame of similarity matrix (words × words).
    """
    # Load vocab and skip first item
    vocab_df = pd.read_csv(path_to_vocab, header=None)
    vocab_df = vocab_df.iloc[1:]  # drop first row
    flat_words = vocab_df[0].tolist()
    N = len(flat_words)

    # Load embeddings and skip first column (before transpose)
    dfE = pd.read_csv(path_to_embeddings, encoding="unicode-escape")
    dfE = dfE.iloc[:, 1:]  # drop first column

    # Transpose so rows = words, cols = dims
    X = dfE.transpose().to_numpy(dtype=float)  # shape N × D

    # Check alignment
    if X.shape[0] != N:
        raise ValueError(
            f"Embeddings ({X.shape[0]} items) and vocab ({N} words) mismatch!"
        )

    # Compute cosine similarity
    D = cdist(X, X, metric="cosine")  # N × N distances
    S = 1.0 - D                       # similarity
    S = np.maximum(S, 0.0)

    # Build dataframe with labels
    dfS = pd.DataFrame(S, index=flat_words, columns=flat_words)

    # Save to CSV
    dfS.to_csv(out_csv, index=True, header=True)

    print(f"Semantic matrix saved to {out_csv} with shape {dfS.shape}")

    return dfS

def create_semantic_vector_label(
    path_to_embeddings: str,
    path_to_vocab: str,
    out_csv: str):
    """
    Adapted from Kumar et al. (2024)

    Creates a 1xN semantic similarity matrix from embeddings. Matrix is for category label x words in that category, saves with word labels.

    Args:
        path_to_embeddings: CSV of embeddings (assumes D×N: rows = dims, cols = words, assumes category label is first word).
        path_to_vocab: CSV with a single column of word labels (length N, assumes category label is first word).
        out_csv: output CSV path for 1xN similarity matrix.

    Returns:
        DataFrame of similarity matrix (category label × words).
    """

    # Load vocab
    vocab_df = pd.read_csv(path_to_vocab, header=None)
    flat_words = vocab_df[0].tolist()
    N = len(flat_words)

    # Load embeddings
    dfE = pd.read_csv(path_to_embeddings, encoding="unicode-escape")
    X = dfE.transpose().to_numpy(dtype=float)   # shape N×D

    if X.shape[0] != N:
        raise ValueError(f"Embeddings ({X.shape[0]} items) and vocab ({N} words) mismatch!")

    # First word vector
    first_vec = X[0].reshape(1, -1)

    # Compute cosine similarity (1 × N)
    D = cdist(first_vec, X, metric="cosine")
    S = 1.0 - D
    S = np.maximum(S, 0.0)

    # ❗ Remove self-similarity (first column)
    S = S[:, 1:]
    flat_words_no_self = flat_words[1:]

    # Build dataframe
    dfS = pd.DataFrame(S, columns=flat_words_no_self, index=[flat_words[0]])

    # Save
    dfS.to_csv(out_csv, index=True, header=True)
    print(f"Semantic vector saved to {out_csv} with shape {dfS.shape}")
    return dfS


def compute_average_similarity(matrix_csv: str) -> float:
    df = pd.read_csv(matrix_csv, index_col=0)
    S = df.apply(pd.to_numeric, errors="coerce").to_numpy()
    """
    Compute average similarity from a supplied semantic matrix (either NxN or 1xN)

    """
    # For N x N inter-item similarity - skip diagonal (ie no self-self similarity)
    if S.shape[0] == S.shape[1]:
        mask = ~np.eye(S.shape[0], dtype=bool)
        return np.nanmean(S[mask])

    # For 1 x N category label similarity
    elif S.shape[0] == 1:
        return np.nanmean(S)

    else:
        raise ValueError(f"Unexpected shape: {S.shape}")


def compute_average_similarity_all_categories(
    base_dir: str,
    matrix_name: str,
    out_csv: str):

    """
    Compute average similarity from a supplied semantic matrix across set of categories. Vocabs and matrices should be stored in subfolders for each category.
    
    """
    results = []

    for cat in os.listdir(base_dir):
        cat_path = os.path.join(base_dir, cat)

        if not os.path.isdir(cat_path):
            continue

        matrix_path = os.path.join(cat_path, matrix_name)

        if not os.path.exists(matrix_path):
            print(f"Skipping {cat}: {matrix_name} not found")
            continue

        try:
            avg_sim = compute_average_similarity(matrix_path)

            results.append({
                "Category": cat,
                "Average_Similarity": avg_sim
            })

            print(f"{cat}: {avg_sim:.4f}")

        except Exception as e:
            print(f"Error in {cat}: {e}")

    results_df = pd.DataFrame(results).sort_values("Category")
    results_df.to_csv(os.path.join(base_dir, out_csv), index=False)

    print(f"\nSaved summary to {os.path.join(base_dir, out_csv)}")
    return results_df


def compute_all_matrix_types(base_dir: str):
    matrix_types = {
        "Interitem": "USE_semantic_matrix_items.csv",
        "Label": "USE_semantic_matrix_category_label.csv"
    }

    """
    Compute average similarity from supplied semantic matrices (BOTH NxN and 1xN)
    
    """

    all_results = []

    for label, matrix_name in matrix_types.items():
        print(f"\nProcessing {label} matrices...\n")

        results = compute_average_similarity_all_categories(
        base_dir=base_dir,
        matrix_name=matrix_name,
        out_csv=f"category_average_similarity_{label}.csv"
    )

        results["Matrix_Type"] = label
        all_results.append(results)

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(os.path.join(base_dir, "category_average_similarity.csv"), index=False)

    print("\nSaved combined results.")
    return final_df


base_dir = "stimuli_by_category"
results = compute_all_matrix_types(base_dir)
print(results)


##########################################################################################################################################

#Get similarity between all items in each category, outputs NxN pairwise similarity matrix

import os
base_dir = "stimuli_by_category"

for cat in os.listdir(base_dir):
    cat_path = os.path.join(base_dir, cat)

    # define file paths
    vocab_path = os.path.join(cat_path, "vocab.csv")
    emb_path   = os.path.join(cat_path, "USE_embeddings.csv")
    out_path   = os.path.join(cat_path, "USE_semantic_matrix_items.csv")
    
    # skip if required files don't exist
    if not os.path.exists(vocab_path) or not os.path.exists(emb_path):
        print(f"Skipping {cat} (missing files)")
        continue
    
    print(f"Processing {cat}...")
    
    create_semantic_matrix_items(
        path_to_embeddings=emb_path,
        path_to_vocab=vocab_path,
        out_csv=out_path
    )

###

# Get similarity between category label and each word in category  (1xN matrix)
import os
base_dir = "stimuli_by_category"

for cat in os.listdir(base_dir):
    cat_path = os.path.join(base_dir, cat)

    # define file paths
    vocab_path = os.path.join(cat_path, "vocab.csv")
    emb_path   = os.path.join(cat_path, "USE_embeddings.csv")
    out_path   = os.path.join(cat_path, "USE_semantic_matrix_category_label.csv")
    
    # skip if required files don't exist
    if not os.path.exists(vocab_path) or not os.path.exists(emb_path):
        print(f"Skipping {cat} (missing files)")
        continue
    
    print(f"Processing {cat}...")

    create_semantic_vector_label(
        path_to_embeddings=emb_path,
        path_to_vocab=vocab_path,
        out_csv=out_path
    )
   
###

#Get average similarities for both types of matrices, output in one file
base_dir = "stimuli_by_category"
results = compute_all_matrix_types("stimuli_by_category")