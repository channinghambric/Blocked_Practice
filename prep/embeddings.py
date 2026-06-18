import pandas as pd
import tensorflow_hub as hub
import numpy as np
import os


class embeddings:
    """
    Creates USE embeddings and vocab file from a list of words.
    First row is always category name.
    """

    def __init__(self, words, path_for_lexical_data):

        self.path = path_for_lexical_data + '/USE_embeddings.csv'

        # -----------------------------
        # 1. Separate category label
        # -----------------------------
        category = words[0]

        # -----------------------------
        # 2. Get words
        # -----------------------------
        items = words[1:]

        # -----------------------------
        # 3. Clean words 
        # -----------------------------
        items = [w.lower() for w in items]

        # Remove duplicates BUT preserve order
        seen = set()
        items_clean = []
        for w in items:
            if w not in seen:
                items_clean.append(w)
                seen.add(w)

        # sort ONLY items (category excluded)
        items_clean = sorted(items_clean)

        # -----------------------------
        # 4. Reattach category FIRST
        # -----------------------------
        self.words = [category] + items_clean

        # -----------------------------
        # 5. Write vocab (NO HEADER)
        # -----------------------------
        pd.DataFrame(self.words).to_csv(
            path_for_lexical_data + '/vocab.csv',
            index=False,
            header=False
        )

        # -----------------------------
        # 6. Load USE model
        # -----------------------------
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        model = hub.load(module_url)

        print("USE model loaded")

        # -----------------------------
        # 7. Create embeddings
        # -----------------------------
        embeddings_list = []

        for v in self.words:
            embeddings_list.append(model([v]).numpy()[0])

        # -----------------------------
        # 8. Save embeddings (columns = words)
        # -----------------------------
        self.df = pd.DataFrame(
            dict(zip(self.words, embeddings_list))
        )

        self.df.to_csv(self.path, index=False)

        print(f"Saved embeddings to {self.path}")

        # load USE model
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
        model = hub.load(module_url)
        print ("module %s loaded" % module_url)
        
        embeddings = []
        
        for v in self.words:
            embeddings.append(model([v]).numpy()[0])
        
        # create a dictionary of words and their embeddings without loop
        self.dict = dict(zip(self.words, embeddings))
        # convert dictionary to dataframe with column names as words and each column is the embedding

        self.df = pd.DataFrame(self.dict)
        # save dataframe as csv file
        self.df.to_csv(self.path, index=False)
        
#### SAMPLE RUN CODE ####
import os
import pandas as pd

base_dir = "stimuli_by_category"

for cat in os.listdir(base_dir):
    cat_path = os.path.join(base_dir, cat)

    # skip if not a folder
    if not os.path.isdir(cat_path):
        continue

    vocab_path = os.path.join(cat_path, "vocab.csv")

    # skip missing vocab files
    if not os.path.exists(vocab_path):
        print(f"Skipping {cat} (no vocab.csv)")
        continue

    # load words
    words = pd.read_csv(vocab_path, header=None)[0].tolist()

    print(f"Processing {cat} ({len(words)} items)")

    # run embeddings
    embeddings(
        words,
        cat_path + "/"
    )