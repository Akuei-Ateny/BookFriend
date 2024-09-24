import pandas as pd

# Load the CSV file
df = pd.read_csv('Assets/books.csv')


# List of columns to keep
columns_to_keep = ['BookID', 'title', 'authors', 'average_rating', 'num_pages', 'ratings_count', 'text_review_counts', 'publisher']  

# Filter the DataFrame
df_cleaned = df[columns_to_keep]

# Check the cleaned data
print(df_cleaned.columns)

# Save the cleaned DataFrame to a new CSV file
df_cleaned.to_csv('books_cleaned.csv', index=False)
