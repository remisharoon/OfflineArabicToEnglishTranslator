from models import TranslatorModel


def main():
    # Initialize the translator model
    translator = TranslatorModel()

    # Example Arabic text
    arabic_text = "مرحبا بك في عالم الترجمة."

    # Translate the text
    translated_text = translator.translate(arabic_text)

    print("Original Arabic Text:", arabic_text)
    print("Translated English Text:", translated_text)


if __name__ == "__main__":
    main()
