import pandas as pd

# Load the original CSV
df = pd.read_csv("aa_dataset-tickets-multi-lang-5-2-50-version.csv")

# Extract only the 'body' and 'answer' columns
qa_df = df[['body', 'answer']].rename(columns={'body': 'question', 'answer': 'answer'})

# Save to a new CSV
qa_df.to_csv("faq_qa_only.csv", index=False)
