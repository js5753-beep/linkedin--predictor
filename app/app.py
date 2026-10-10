# Who's a User? Predicting LinkedIn Users
# Ndalo Bantubahle Mukoki
# Run with:  streamlit run app.py

from pdb import run

from flask import app
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# ---------- 1. Data cleaning and model (same steps as the Part 1 notebook) ----------

def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x


@st.cache_resource
def train_model():
    s = pd.read_csv("social_media_usage.csv")

    ss = pd.DataFrame({
        "sm_li":     clean_sm(s["web1h"]),
        "income":    np.where(s["income"] > 9, np.nan, s["income"]),
        "education": np.where(s["educ2"] > 8, np.nan, s["educ2"]),
        "parent":    np.where(s["par"] > 2, np.nan, clean_sm(s["par"])),
        "married":   np.where(s["marital"] > 6, np.nan, clean_sm(s["marital"])),
        "female":    np.where(s["gender"] > 3, np.nan, np.where(s["gender"] == 2, 1, 0)),
        "age":       np.where(s["age"] > 98, np.nan, s["age"])
    }).dropna()

    y = ss["sm_li"]
    X = ss[["income", "education", "parent", "married", "female", "age"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=987
    )

    lr = LogisticRegression(class_weight="balanced")
    lr.fit(X_train, y_train)

    accuracy = lr.score(X_test, y_test)
    return lr, ss, X.columns, accuracy


lr, ss, feature_cols, accuracy = train_model()


# ---------- 2. Labels for the dropdowns (from the codebook) ----------

income_labels = {
    1: "Less than $10,000",
    2: "$10,000 to under $20,000",
    3: "$20,000 to under $30,000",
    4: "$30,000 to under $40,000",
    5: "$40,000 to under $50,000",
    6: "$50,000 to under $75,000",
    7: "$75,000 to under $100,000",
    8: "$100,000 to under $150,000",
    9: "$150,000 or more",
}

education_labels = {
    1: "Less than high school",
    2: "High school incomplete",
    3: "High school graduate",
    4: "Some college, no degree",
    5: "Two-year associate degree",
    6: "Four-year college degree",
    7: "Some postgraduate schooling",
    8: "Postgraduate or professional degree",
}


# ---------- 3. Page layout and user inputs ----------

st.set_page_config(page_title="Who's a LinkedIn User?", page_icon="💼", layout="wide")

st.title("💼 Who's a LinkedIn User?")
st.markdown(
    "Enter a person's characteristics in the sidebar. The logistic regression model "
    "predicts in real time whether they use LinkedIn and how likely it is."
)

st.sidebar.header("Person's characteristics")

income = st.sidebar.selectbox(
    "Household income", options=list(income_labels.keys()),
    format_func=lambda k: f"{k} – {income_labels[k]}", index=7
)
education = st.sidebar.selectbox(
    "Highest education", options=list(education_labels.keys()),
    format_func=lambda k: f"{k} – {education_labels[k]}", index=6
)
age = st.sidebar.slider("Age", min_value=18, max_value=98, value=42)
female = st.sidebar.radio("Gender", ["Female", "Male / other"], horizontal=True)
married = st.sidebar.radio("Married?", ["Yes", "No"], horizontal=True)
parent = st.sidebar.radio("Parent of a child under 18 at home?", ["No", "Yes"], horizontal=True)

person = pd.DataFrame({
    "income":    [income],
    "education": [education],
    "parent":    [1 if parent == "Yes" else 0],
    "married":   [1 if married == "Yes" else 0],
    "female":    [1 if female == "Female" else 0],
    "age":       [age],
})[feature_cols]


# ---------- 4. Prediction ----------

prob = lr.predict_proba(person)[0, 1]
prediction = lr.predict(person)[0]

st.subheader("Prediction")
col1, col2, col3 = st.columns(3)

if prediction == 1:
    col1.success("### ✅ LinkedIn user")
else:
    col1.error("### ❌ Not a LinkedIn user")

col2.metric("Probability of using LinkedIn", f"{prob:.1%}")
col3.metric("Average across survey respondents", f"{ss['sm_li'].mean():.1%}",
            delta=f"{(prob - ss['sm_li'].mean()) * 100:+.1f} pts vs average")

st.progress(float(prob))
st.caption(f"The model classifies anyone with a probability of 50% or more as a user. "
           f"Model accuracy on held-out test data: {accuracy:.0%}.")

st.divider()


# ---------- 5. Visualisations for the marketing team ----------

st.subheader("What drives the prediction?")
left, right = st.columns(2)

# (a) How probability changes with age for this exact profile
with left:
    ages = np.arange(18, 99)
    profile = pd.concat([person] * len(ages), ignore_index=True)
    profile["age"] = ages
    probs = lr.predict_proba(profile[feature_cols])[:, 1]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ages, probs, color="#0A66C2", linewidth=2.5)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="50% threshold")
    ax.scatter([age], [prob], color="red", s=80, zorder=5, label="This person")
    ax.set_xlabel("Age")
    ax.set_ylabel("Probability of using LinkedIn")
    ax.set_ylim(0, 1)
    ax.set_title("Same profile at different ages")
    ax.legend()
    st.pyplot(fig)
    st.caption("Holding everything else fixed, how the probability changes as the person ages.")

# (b) How probability changes with income for this exact profile, at each education level
with right:
    fig, ax = plt.subplots(figsize=(6, 4))
    for edu, colour in zip([3, 6, 8], ["#9ecae1", "#3182bd", "#08306b"]):
        grid = pd.concat([person] * 9, ignore_index=True)
        grid["income"] = range(1, 10)
        grid["education"] = edu
        ax.plot(range(1, 10), lr.predict_proba(grid[feature_cols])[:, 1],
                marker="o", color=colour, label=education_labels[edu])
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.scatter([income], [prob], color="red", s=80, zorder=5, label="This person")
    ax.set_xlabel("Income level (1 = <$10k ... 9 = $150k+)")
    ax.set_ylabel("Probability of using LinkedIn")
    ax.set_ylim(0, 1)
    ax.set_title("Effect of income and education")
    ax.legend(fontsize=8)
    st.pyplot(fig)
    st.caption("Higher income and education both raise the chance of being a LinkedIn user.")

st.divider()

# (c) Actual LinkedIn use in the survey data, by segment
st.subheader("LinkedIn use in the survey data, by segment")
segment = st.selectbox("Break down by", ["income", "education", "age group", "female", "married", "parent"])

data = ss.copy()
data["age group"] = pd.cut(data["age"], bins=[17, 29, 44, 59, 74, 98],
                           labels=["18–29", "30–44", "45–59", "60–74", "75+"])
share = data.groupby(segment, observed=True)["sm_li"].mean().reset_index()
share[segment] = share[segment].astype(str)

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.bar(share[segment], share["sm_li"], color="#0A66C2")
ax.axhline(ss["sm_li"].mean(), color="grey", linestyle="--", linewidth=1, label="Overall average")
ax.set_ylabel("Share using LinkedIn")
ax.set_xlabel(segment)
ax.set_ylim(0, 1)
ax.legend()
st.pyplot(fig)
st.caption(f"Based on {len(ss):,} cleaned survey respondents. For binary segments, 1 = yes and 0 = no.")

# (d) Model coefficients
with st.expander("Model details: logistic regression coefficients"):
    coefs = pd.DataFrame({"Feature": feature_cols, "Coefficient (log-odds)": lr.coef_[0]}).round(3)
    st.dataframe(coefs, hide_index=True)
    st.markdown(
        "Positive coefficients raise the odds of using LinkedIn; negative ones lower them. "
        "**Marketing takeaway:** LinkedIn is the strongest channel for higher-income, more educated, "
        "younger-to-middle-aged audiences."
    )

