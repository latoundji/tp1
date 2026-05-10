"""
TP1 — Recommendation System : Collaborative Filtering Item-Item Top-N
Implémentation individuelle avec interface Streamlit.

Installation :
    pip install streamlit pandas numpy scikit-learn

Lancement :
    streamlit run tp1_collaborative_filtering.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


# ══════════════════════════════════════════════════════════
# CLASSE PRINCIPALE : ItemItemCollaborativeFilter
# ══════════════════════════════════════════════════════════

class ItemItemCollaborativeFilter:

    def __init__(self, n_neighbors: int = 10):
        self.n_neighbors = n_neighbors
        self.similarity_matrix = None
        self.ratings_matrix = None
        self.items = None
        self.users = None

    def fit(self, ratings_df: pd.DataFrame):
        self.ratings_matrix = ratings_df.pivot_table(
            index="user_id", columns="item_id", values="rating"
        )
        self.users = self.ratings_matrix.index.tolist()
        self.items = self.ratings_matrix.columns.tolist()

        item_means = self.ratings_matrix.mean(axis=0)
        ratings_centered = self.ratings_matrix.subtract(item_means, axis=1).fillna(0)

        sim = cosine_similarity(ratings_centered.T)
        self.similarity_matrix = pd.DataFrame(
            sim, index=self.items, columns=self.items
        )
        return self

    def predict_score(self, user_id, item_id) -> float:
        if user_id not in self.users or item_id not in self.items:
            return 0.0

        user_ratings = self.ratings_matrix.loc[user_id]
        rated_items = user_ratings.dropna().index.tolist()
        if item_id in rated_items:
            rated_items.remove(item_id)
        if not rated_items:
            return 0.0

        similarities = self.similarity_matrix.loc[item_id, rated_items]
        top_neighbors = similarities.nlargest(self.n_neighbors)
        top_neighbors = top_neighbors[top_neighbors > 0]
        if top_neighbors.empty:
            return 0.0

        numerator = sum(
            top_neighbors[i] * user_ratings[i] for i in top_neighbors.index
        )
        denominator = top_neighbors.abs().sum()
        return numerator / denominator if denominator != 0 else 0.0

    def recommend(self, user_id, n: int = 5) -> pd.DataFrame:
        if user_id not in self.users:
            return pd.DataFrame(columns=["item_id", "predicted_score"])

        user_ratings = self.ratings_matrix.loc[user_id]
        unrated_items = user_ratings[user_ratings.isna()].index.tolist()

        scores = [
            (item, self.predict_score(user_id, item))
            for item in unrated_items
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return pd.DataFrame(scores[:n], columns=["item_id", "predicted_score"])

    def similar_items(self, item_id, n: int = 5) -> pd.DataFrame:
        if item_id not in self.items:
            return pd.DataFrame()
        sims = self.similarity_matrix[item_id].drop(item_id).nlargest(n)
        return pd.DataFrame({"item_id": sims.index, "similarity": sims.values})


# ══════════════════════════════════════════════════════════
# DONNÉES DE DÉMONSTRATION
# ══════════════════════════════════════════════════════════

def load_demo_data():
    users = ["Alice", "Bob", "Carol", "David", "Eve",
             "Frank", "Grace", "Hank", "Iris", "Jack"]
    movies = [
        "Inception", "The Matrix", "Interstellar", "Parasite",
        "The Dark Knight", "Spirited Away", "Pulp Fiction",
        "The Shawshank Redemption", "Forrest Gump", "The Godfather",
        "Goodfellas", "Fight Club", "The Silence of the Lambs",
        "Schindler's List", "The Lord of the Rings"
    ]

    np.random.seed(42)
    rows = []
    for user in users:
        rated_movies = np.random.choice(
            movies, size=np.random.randint(8, 13), replace=False
        )
        for movie in rated_movies:
            rating = np.random.choice(
                [1, 2, 3, 4, 5], p=[0.05, 0.1, 0.2, 0.35, 0.3]
            )
            rows.append({
                "user_id": user,
                "item_id": movie,
                "rating": float(rating)
            })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════
# INTERFACE STREAMLIT
# ══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="TP1 — Collaborative Filtering",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Système de Recommandation")
st.subheader("Collaborative Filtering — Item-Item Top-N")

with st.sidebar:
    st.header("⚙️ Paramètres")
    n_neighbors = st.slider("Nombre de voisins (K)", 1, 20, 10)
    top_n = st.slider("Top-N recommandations", 1, 10, 5)
    st.markdown("---")
    st.info("**Données** : Films fictifs (démo)")

df = load_demo_data()
model = ItemItemCollaborativeFilter(n_neighbors=n_neighbors)
model.fit(df)

tab1, tab2, tab3 = st.tabs([
    "🎯 Recommandations",
    "🔗 Items Similaires",
    "📊 Matrice des notes"
])

with tab1:
    st.markdown("### Recommandations pour un utilisateur")
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_user = st.selectbox("Choisir un utilisateur", model.users)
        generer = st.button("🚀 Générer les recommandations")
    with col2:
        if generer:
            recs = model.recommend(selected_user, n=top_n)
            if recs.empty:
                st.warning("Cet utilisateur a tout noté !")
            else:
                st.markdown(f"**Top-{top_n} films recommandés pour {selected_user} :**")
                for _, row in recs.iterrows():
                    score = row["predicted_score"]
                    stars = "⭐" * round(score)
                    st.markdown(
                        f"- **{row['item_id']}** — Score prédit: `{score:.2f}` {stars}"
                    )

    st.markdown("---")
    st.markdown(f"**Films déjà notés par {selected_user} :**")
    user_data = df[df["user_id"] == selected_user][
        ["item_id", "rating"]
    ].sort_values("rating", ascending=False)
    st.dataframe(user_data, use_container_width=True)

with tab2:
    st.markdown("### Items les plus similaires")
    selected_item = st.selectbox("Choisir un film", model.items)
    sims = model.similar_items(selected_item, n=top_n)
    if not sims.empty:
        st.markdown(f"**Films similaires à _{selected_item}_ :**")
        st.dataframe(
            sims.style.background_gradient(subset=["similarity"], cmap="Blues"),
            use_container_width=True
        )

with tab3:
    st.markdown("### Matrice Utilisateur-Item (notes)")
    st.dataframe(
        model.ratings_matrix.style.background_gradient(cmap="RdYlGn", axis=None),
        use_container_width=True
    )
    st.markdown("### Matrice de similarité Item-Item")
    st.dataframe(
        model.similarity_matrix.style.background_gradient(cmap="Blues"),
        use_container_width=True
    )