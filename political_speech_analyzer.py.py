import pandas as pd
import re

# Load the unzipped CSV file

df = pd.read_csv("speeches_bps.csv")

# Drop unnecessary column
df = df.drop(columns=['Unnamed: 0'], errors='ignore')

# Convert Date to standard format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Function to clean speech text
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        return text
    return text

# Apply text cleaning
df['Cleaned_Speech'] = df['Speech'].apply(clean_text)

# Display the first few rows
print(df.head())

# Save the cleaned dataset
df.to_csv("cleaned_speeches.csv", index=False)

#after cleaning we perform eda

