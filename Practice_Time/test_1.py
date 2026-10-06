import tiktoken

text="My Name is Fairoz Sheikh?"

tokenizer=tiktoken.encoding_for_model(model_name="gpt-4")
token_IDs=tokenizer.encode(text)

print("Embedding is as : ",token_IDs)

# convert these tokens back into the original text

print("Original Text",tokenizer.decode(token_IDs))
