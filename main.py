"""Main script for the Streamlit app."""
import time
from typing import List

import streamlit as st

from model import TranslateModel, HelsinkiItc, FacebookMbart, AwsTranslate, Language


def load_models() -> List[TranslateModel]:
    return [
        HelsinkiItc(),
        FacebookMbart(),
        AwsTranslate()
    ]


def main() -> None:
    """Main function for the Streamlit app."""
    st.set_page_config(layout="wide")
    st.title("Data Day: Translation")

    languages = [Language.spanish, Language.italian, Language.portugees, Language.catalan, Language.galician]
    source_lang = st.selectbox("Target", options=languages, index=0)
    target_lang = st.selectbox("Target", options=[lang for lang in languages if lang != source_lang], index=0)
    models = load_models()

    text_input = st.text_area(
        "Text to translate",
        value="Particular. Vendo Scooter comprada nueva en 2010, "
              "solo un propietario, excelente estado, "
              "con todas sus correspondientes revisiones y Libro de Revisiones, "
              "ultima revisión el pasado 24 ABR 2023, con zapata freno trasero nuevo."
    )

    for model in models:
        st.header(f"Model {model.endpoint_name}")
        start_time = time.time()
        translation = model.predict(text=text_input, source_lang=source_lang, target_lang=target_lang)
        st.subheader(translation)
        st.write("--- %s seconds ---" % round((time.time() - start_time), 3))


if __name__ == "__main__":
    main()
