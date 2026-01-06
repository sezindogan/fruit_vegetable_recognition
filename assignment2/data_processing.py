
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def main():
    print("Loading datasets...")
    # Load datasets assuming 'id' is the index
    try:
        df_text = pd.read_csv("text.csv", index_col=0)
        df_image = pd.read_csv("image.csv", index_col=0)
        df_meta = pd.read_csv("meta.csv", index_col=0)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    print(f"Original Text shape: {df_text.shape}")
    print(f"Image shape: {df_image.shape}")
    print(f"Meta shape: {df_meta.shape}")

    # Standardize text features before PCA
    print("Standardizing text features...")
    scaler_text = StandardScaler()
    X_text_scaled = scaler_text.fit_transform(df_text)

    # Apply PCA to text features
    # Retaining 95% variance to reduce dimensions while keeping most information
    print("Applying PCA to text features (95% variance)...")
    pca = PCA(n_components=0.95, random_state=42)
    X_text_pca = pca.fit_transform(X_text_scaled)

    print(f"Reduced Text shape: {X_text_pca.shape} (retained {pca.n_components_} components)")

    # Create DataFrame for reduced text features
    # Naming columns text_pca_0, text_pca_1, ...
    df_text_pca = pd.DataFrame(
        X_text_pca, 
        index=df_text.index, 
        columns=[f'text_pca_{i}' for i in range(X_text_pca.shape[1])]
    )

    # Fuse datasets: Image + Reduced Text + Meta
    print("Fusing datasets (Image + Reduced Text + Meta)...")
    # Join on index to ensure alignment (assuming indices match)
    df_fused_new = pd.concat([df_image, df_text_pca, df_meta], axis=1)

    print(f"New Fused shape: {df_fused_new.shape}")

    # Save to CSV
    output_filename = "fused_reduced.csv"
    df_fused_new.to_csv(output_filename)
    print(f"Saved new fused dataset to '{output_filename}'")

if __name__ == "__main__":
    main()
