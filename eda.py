
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

# Load the cleaned dataset
df = pd.read_csv("cleaned_speeches.csv")

# Basic Dataset Insights
print(f"Total speeches: {df.shape[0]}")
print(f"Total unique speakers: {df['Speaker'].nunique()}")
print(df['Speaker'].value_counts().head(10))  # Top 10 speakers with most speeches

# Speech Length Distribution
df['Speech_Length'] = df['Cleaned_Speech'].apply(lambda x: len(str(x).split()))
print(df['Speech_Length'].describe())

# Plot Speech Length Distribution
plt.figure(figsize=(10, 5))
plt.hist(df['Speech_Length'], bins=30, edgecolor='black', alpha=0.7)
plt.title("Distribution of Speech Lengths")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.show()

# Word Frequency Analysis
all_text = " ".join(df['Cleaned_Speech'].dropna())
word_counts = Counter(all_text.split())
most_common_words = word_counts.most_common(20)

# Plot Most Common Words
plt.figure(figsize=(10, 5))
plt.bar(*zip(*most_common_words))
plt.xticks(rotation=45)
plt.title("Top 20 Most Frequent Words in Speeches")
plt.show()

# Generate a Word Cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Speech Content")
plt.show()

#checking if dataset is labeled or not which is unlabeled one
# Display first few rows
print(df.head())

# Check for Bias Labels
print(df.columns)