from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
model = T5EncoderModel.from_pretrained("google/t5-v1_1-xxl")
