import pandas as pd
import numpy as np
import openai

def create_feature_embeddings(df):
    # Assuming df has columns "params" and "results"
    
    # Initialize empty DataFrames for embeddings
    params_embed_df = pd.DataFrame()
    results_embed_df = pd.DataFrame()
    
    # Standardize the "params" and "results" columns
    
    
    # Iteratively create embeddings (for demonstration, we'll use the same values)
    for index, row in df.iterrows():
        params = row['params']
        results = row['results']
        
        # Create embeddings (in this example, simply square the values)
        params_embed = params ** 2
        results_embed = results ** 2
        
        # Append to the respective DataFrames
        params_embed_df = params_embed_df.append({'params_embed': params_embed}, ignore_index=True)
        results_embed_df = results_embed_df.append({'results_embed': results_embed}, ignore_index=True)
    
    # Concatenate the original DataFrame with the embeddings DataFrames
    df = pd.concat([df, params_embed_df, results_embed_df], axis=1)
    
    return df


if __name__ == '__main__':
    # Sample DataFrame
    df = pd.DataFrame({
        'params': np.random.rand(10),
        'results': np.random.rand(10)
    })

    # Create feature embeddings
    df_with_embeddings = create_feature_embeddings(df)
    print(df_with_embeddings.head(3))
