# Project 13: Image-Text Hybrid Search

**Difficulty:** Expert  
**Module:** 13 (Multimodal)

## 📌 The Challenge
Implement a cross-modal search paradigm where a user can query an image database using purely textual descriptions, or query a text database using uploaded images. 

## 📖 The Approach
1. **Vector Normalization**: By projecting both images and text into a shared embedding space (e.g. using `CLIP`), we can use cosine similarity accurately across text/image boundaries.
2. **Database Ingestion**: Process a small dataset of images (like Unsplash thumbnails), extract features, and index them into FAISS or ChromaDB.
3. **Query Logic**: When the user searches "a fast red car", embed that string via CLIP text-encoder, and fetch the Top-K nearest image embeddings from the same joint latent space.
4. **Ranking & Validation**: Measure MRR@10 based on standard testing splits. Evaluate how CLIP deals with negative modifiers (e.g., "a car that is NOT red").

## ✅ Checkpoints
- [ ] Set up simple python script to download and reshape 10 sample `.jpg` files.
- [ ] Pass text and image through a pre-trained `CLIPModel` to get unified 512D vectors.
- [ ] Construct the FAISS `IndexFlatIP` and compute similarities.
- [ ] Expose an interactive loop allowing iterative querying.
