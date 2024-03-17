from transformers import MarianMTModel, MarianTokenizer


class TranslatorModel:
    def __init__(self):
        # Relative path from the `src` directory
        self.model_dir = "./model_ar_en"
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_dir)
        self.model = MarianMTModel.from_pretrained(self.model_dir)

    def translate(self, text):
        # Tokenize the Arabic text for the model using the recommended approach
        model_inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

        # Generate translation using the model
        translated = self.model.generate(**model_inputs)

        # Decode the translated text
        translated_text = self.tokenizer.decode(translated[0], skip_special_tokens=True)

        return translated_text

# from transformers import MarianMTModel, MarianTokenizer
#
#
# class TranslatorModel:
#     def __init__(self, model_name="Helsinki-NLP/opus-mt-ar-en", cache_dir="./model_cache"):
#         # Load the tokenizer and model with a specific cache directory
#         self.tokenizer = MarianTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
#         self.model = MarianMTModel.from_pretrained(model_name, cache_dir=cache_dir)
#
#     def translate(self, text):
#         # Tokenize the Arabic text for the model using the recommended approach
#         model_inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
#
#         # Generate translation using the model
#         translated = self.model.generate(**model_inputs)
#
#         # Decode the translated text
#         translated_text = self.tokenizer.decode(translated[0], skip_special_tokens=True)
#
#         return translated_text
