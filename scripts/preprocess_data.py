
"""Preprocess synthetic datasets as per documentation."""
import pandas as pd
import json, numpy as np, pathlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load posts
df_posts = pd.read_csv("datasets/synthetic_posts.csv")
# Text features
vec = TfidfVectorizer(stop_words='english')
X_text = vec.fit_transform(df_posts["text"])
y_text = (df_posts["platform"] == "Facebook").astype(int)  # demo binary label

# Split text
X_tr_text, X_te_text, y_tr_text, y_te_text = train_test_split(
    X_text, y_text, test_size=0.2, random_state=42
)

# Save artefacts
pathlib.Path("artefacts").mkdir(exist_ok=True)
import scipy.sparse as sp
sp.save_npz("artefacts/X_train_text.npz", X_tr_text)
sp.save_npz("artefacts/X_test_text.npz", X_te_text)
np.save("artefacts/y_train_text.npy", y_tr_text)
np.save("artefacts/y_test_text.npy", y_te_text)
import pickle
pickle.dump(vec, open("artefacts/vectorizer.pkl","wb"))

# Load image features
df_img = pd.read_csv("datasets/synthetic_image_features.csv")
X_img = df_img[[f"feat{i}" for i in range(1,11)]].values
y_img = df_img["label"].values
scaler = StandardScaler().fit(X_img)
X_img_std = scaler.transform(X_img)

# Split image
X_tr_img, X_te_img, y_tr_img, y_te_img = train_test_split(
    X_img_std, y_img, test_size=0.2, random_state=42
)
np.save("artefacts/X_train_img.npy", X_tr_img)
np.save("artefacts/X_test_img.npy", X_te_img)
np.save("artefacts/y_train_img.npy", y_tr_img)
np.save("artefacts/y_test_img.npy", y_te_img)
pickle.dump(scaler, open("artefacts/scaler.pkl","wb"))

print("Preprocessing complete; artefacts saved in artefacts/") 
