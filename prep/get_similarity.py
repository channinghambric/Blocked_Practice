import numpy as np
import pandas as pd
import os

def compute_average_similarity(matrix_csv: str) -> float:
    df = pd.read_csv(matrix_csv, index_col=0)
    S = df.apply(pd.to_numeric, errors="coerce").to_numpy()

    # For N x N inter-item similarity - skip diagonal
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
    out_csv: str
):
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

#Get similarity between all items in each category
'''
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
'''
###
'''
# Get similarity between category label and all words in category  
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
    '''
###

#Get average similarities
base_dir = "stimuli_by_category"
results = compute_all_matrix_types("stimuli_by_category")