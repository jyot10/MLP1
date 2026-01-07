import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

sns.set(style="whitegrid")

# -----------------------------
# Load IRIS dataset
# -----------------------------
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target_names[iris.target]

print("Dataset Head:")
print(df.head())

# -----------------------------------
# A. NUMERICAL DESCRIPTIVE STATISTICS
# -----------------------------------
print("\nDescriptive Statistics:")
print(df.describe())

# Histogram for feature distribution
df.iloc[:, :4].hist(figsize=(10, 6), bins=15, edgecolor="black")
plt.suptitle("Histogram of Numerical Features")
plt.tight_layout()
plt.show()

# Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(data=df.iloc[:, :4])
plt.title("Boxplot for Numerical Features")
plt.show()

# KDE (Density Plot)
plt.figure(figsize=(8, 5))
sns.kdeplot(df["sepal length (cm)"], fill=True)
plt.title("Density Plot of Sepal Length")
plt.show()

# -----------------------------------
# B. CATEGORICAL ANALYSIS
# -----------------------------------
print("\nCategory Counts:")
print(df["species"].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(x="species", data=df)
plt.title("Count Plot of Species")
plt.show()

# Pie chart
df["species"].value_counts().plot(
    kind="pie", autopct="%1.1f%%", figsize=(5, 5)
)
plt.title("Species Distribution")
plt.ylabel("")
plt.show()

# -----------------------------------
# C. MULTIVARIATE ANALYSIS (CORRELATION)
# -----------------------------------
corr = df.iloc[:, :4].corr()
print("\nCorrelation Matrix:")
print(corr)

plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Pairplot
sns.pairplot(df, hue="species")
plt.suptitle("Pairplot of IRIS Dataset", y=1.02)
plt.show()

# -----------------------------------
# D. TIME SERIES ANALYSIS (Synthetic Example)
# -----------------------------------
dates = pd.date_range(start="2024-01-01", periods=50)
sales = np.random.randint(100, 500, size=50)

time_df = pd.DataFrame({
    "date": dates,
    "sales": sales
})

plt.figure(figsize=(10, 4))
plt.plot(time_df["date"], time_df["sales"], marker="o")
plt.title("Sales Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Rolling average
time_df["rolling_mean"] = time_df["sales"].rolling(window=5).mean()

plt.figure(figsize=(10, 4))
plt.plot(time_df["date"], time_df["sales"], label="Actual", marker="o")
plt.plot(
    time_df["date"],
    time_df["rolling_mean"],
    label="Rolling Mean (5)",
    linewidth=3
)
plt.title("Sales Trend with Rolling Average")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
