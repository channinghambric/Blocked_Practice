from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import os


# =========================
# MATRIX CREATION FUNCTIONS
# =========================

def create_semantic_matrix_items(path_to_embeddings, path_to_vocab, out_csv):

    vocab_df = pd.read_csv(path_to_vocab, header=None)
    vocab_df = vocab_df.iloc[1:]  # drop category row if present
    flat_words = vocab_df[0].tolist()
    N = len(flat_words)

    dfE = pd.read_csv(path_to_embeddings, encoding="unicode-escape")
    dfE = dfE.iloc[:, 1:]  # assumes first column is index

    X = dfE.transpose().to_numpy(dtype=float)

    if X.shape[0] != N:
        raise ValueError(f"Mismatch: embeddings {X.shape[0]} vs vocab {N}")

    D = cdist(X, X, metric="cosine")
    S = 1 - D
    S = np.maximum(S, 0.0)

    dfS = pd.DataFrame(S, index=flat_words, columns=flat_words)
    dfS.to_csv(out_csv, index=True, header=True)

    return dfS


def create_semantic_vector_label(path_to_embeddings, path_to_vocab, out_csv):

    vocab_df = pd.read_csv(path_to_vocab, header=None)
    flat_words = vocab_df[0].tolist()
    N = len(flat_words)

    dfE = pd.read_csv(path_to_embeddings, encoding="unicode-escape")
    X = dfE.transpose().to_numpy(dtype=float)

    if X.shape[0] != N:
        raise ValueError(f"Mismatch: embeddings {X.shape[0]} vs vocab {N}")

    first_vec = X[0].reshape(1, -1)

    D = cdist(first_vec, X, metric="cosine")
    S = 1 - D
    S = np.maximum(S, 0.0)

    S = S[:, 1:]
    cols = flat_words[1:]

    dfS = pd.DataFrame(S, index=[flat_words[0]], columns=cols)
    dfS.to_csv(out_csv, index=True, header=True)

    return dfS

# =========================
# SIMILARITY FUNCTIONS
# =========================

def compute_average_similarity(matrix_csv: str):

    df = pd.read_csv(matrix_csv, index_col=0)
    S = df.apply(pd.to_numeric, errors="coerce").to_numpy()

    if S.shape[0] == S.shape[1]:
        mask = ~np.eye(S.shape[0], dtype=bool)
        return np.nanmean(S[mask])

    elif S.shape[0] == 1:
        return np.nanmean(S)

    else:
        raise ValueError(f"Unexpected shape: {S.shape}")


def compute_average_similarity_all_categories(base_dir, matrix_name, out_csv):

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
                "CategoryName": cat,
                "Average_Similarity": avg_sim
            })

            print(f"{cat}: {avg_sim:.4f}")

        except Exception as e:
            print(f"Error in {cat}: {e}")

    results_df = pd.DataFrame(results)

    if not results_df.empty:
        results_df = results_df.sort_values("CategoryName")

    results_df.to_csv(os.path.join(base_dir, out_csv), index=False)

    return results_df


def compute_all_matrix_types(base_dir):

    matrix_types = {
        "Interitem": "USE_semantic_matrix_items.csv",
        "Label": "USE_semantic_matrix_category_label.csv"
    }

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

    return final_df


# =========================
# CLEAN PIPELINE INTERFACE
# =========================

def build_all_matrices(base_dir):

    for cat in os.listdir(base_dir):
        cat_path = os.path.join(base_dir, cat)

        vocab_path = os.path.join(cat_path, "vocab.csv")
        emb_path = os.path.join(cat_path, "USE_embeddings.csv")

        if not os.path.exists(vocab_path) or not os.path.exists(emb_path):
            continue

        print("Creating matrices for:", cat)

        create_semantic_matrix_items(
            emb_path,
            vocab_path,
            os.path.join(cat_path, "USE_semantic_matrix_items.csv")
        )

        create_semantic_vector_label(
            emb_path,
            vocab_path,
            os.path.join(cat_path, "USE_semantic_matrix_category_label.csv")
        )


def compute_all_similarities(base_dir):
    return compute_all_matrix_types(base_dir)

#########  SAMPLE RUN CODE#####################

build_all_matrices("stimuli_by_category")
results = compute_all_similarities("stimuli_by_category")
print(results)