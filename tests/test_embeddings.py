from app.embeddings import create_embedding

text = "Apple revenue increased by 15%."

vector = create_embedding(text)

print("Dimensions:", len(vector))
print(vector[:10])