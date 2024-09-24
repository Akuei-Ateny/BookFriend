import pandas as pd
import numpy as np

# Define columns for the dataset
columns = ['user_id', 'title', 'authors', 'average_rating', 'num_pages', 'ratings_count', 'text_review_counts', 'publisher']

# Number of rows
n_rows = 500

# Generate synthetic data for 500 users interacting with books
data = {
    'user_id': np.arange(1, n_rows + 1),  # 500 users
    'title': ['Book ' + chr(65 + i % 26) for i in range(n_rows)],  # Books titled A to Z repeated
    'authors': ['Author ' + chr(65 + i % 26) for i in range(n_rows)],  # Authors named A to Z repeated
    'average_rating': np.random.uniform(1, 5, n_rows),  # Random ratings between 1 and 5
    'num_pages': np.random.randint(100, 500, n_rows),  # Random page counts between 100 and 500
    'ratings_count': np.random.randint(100, 10000, n_rows),  # Random rating count between 100 and 10,000
    'text_review_counts': np.random.randint(10, 1000, n_rows),  # Random review count between 10 and 1000
    'publisher': np.random.choice(['Publisher A', 'Publisher B', 'Publisher C', 'Publisher D', 'Publisher E',
                                   'Publisher F', 'Publisher G', 'Publisher H', 'Publisher I', 'Publisher J'], n_rows)  # Randomly assign one of 10 publishers
}

# Create the DataFrame
df_books = pd.DataFrame(data, columns=columns)

# Display the first few rows of the dataset
print(df_books.head())

# Save the dataset to a CSV file for future use
df_books.to_csv('Assets/synthetic_books_500.csv', index=False)
