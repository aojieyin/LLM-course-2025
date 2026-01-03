import streamlit as st
import spacy
import pandas as pd

from util import pdf_utils
from util.embedings_utils import embed_chunks, save_embeddings, embeddings_to_tensor
from util.nlp_utils import sentencize, chunk, chunks_to_text_elems
from util.generator_utils import (
    load_tokenizer,
    tokenize_with_chat,
    tokenize_with_rag_prompt,
    load_gemma,
    generate_answer
)
from util.session_utils import SESSION_VARS, put_to_session, get_from_session, print_session
from util.vector_search_utils import retrieve_relevant_resources

from sentence_transformers import SentenceTransformer



# Language configuration

LANG_CONFIG = {
    "English": {
        "spacy_model": "en_core_web_sm",
        "min_chunk_tokens": 200
    },
    "German": {
        "spacy_model": "de_core_news_sm",
        "min_chunk_tokens": 220
    },
    "Chinese": {
        "spacy_model": "zh_core_web_sm",
        "min_chunk_tokens": 120
    }
}

st.title("Multilingual PDF RAG Demo")

language = st.selectbox(
    "Select document language",
    options=list(LANG_CONFIG.keys()),
    index=0
)

lang_cfg = LANG_CONFIG[language]
min_token_length = lang_cfg["min_chunk_tokens"]

st.write(f"Using language: **{language}**")
st.write(f"Chunk token threshold: **{min_token_length}**")



# Model initialization

st.write("Initializing models")

if not get_from_session(st, SESSION_VARS.LOADED_MODELS):

    # spaCy 
    st.write("Loading spaCy model")
    nlp = spacy.load(lang_cfg["spacy_model"])
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    put_to_session(st, SESSION_VARS.NLP, nlp)

    # Multilingual Embedding 
    st.write("Loading multilingual embedding model")
    embedding_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        device="cpu"
    )
    put_to_session(st, SESSION_VARS.EMBEDDING_MODEL_CPU, embedding_model)

    #  LLM (Gemma) 
    st.write("Loading LLM")
    model_name = "google/gemma-2b-it"
    gemma_model = load_gemma(model_name)
    tokenizer = load_tokenizer(model_name)

    put_to_session(st, SESSION_VARS.MODEL, gemma_model)
    put_to_session(st, SESSION_VARS.TOKENIZER, tokenizer)

    put_to_session(st, SESSION_VARS.LOADED_MODELS, True)
    st.success("Models loaded")

else:
    st.write("Models already loaded")

print_session(st)



# UI

query = st.text_input("Type your query here", "请输入你的问题 / Geben Sie Ihre Frage ein")

gen_variant = st.selectbox(
    "Generation mode",
    ("vanilla", "rag")
)

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type="pdf"
)

button_clicked = st.button("Generate")



# PDF processing

if uploaded_file is not None:

    if uploaded_file.name != get_from_session(st, SESSION_VARS.CUR_PDF_FILENAME):
        put_to_session(st, SESSION_VARS.PROCESSED_DATA, None)
        put_to_session(st, SESSION_VARS.CUR_PDF_FILENAME, uploaded_file.name)

    if not get_from_session(st, SESSION_VARS.PROCESSED_DATA):

        with st.expander("Preprocessing", expanded=True):

            st.write("Reading PDF")
            pages_and_texts = pdf_utils.open_and_read_pdf(uploaded_file)

            st.write("Sentence segmentation")
            sentencize(pages_and_texts, get_from_session(st, SESSION_VARS.NLP))

            st.write("Chunking")
            chunk(pages_and_texts)

            st.write("Converting to chunks")
            pages_and_chunks = chunks_to_text_elems(pages_and_texts)
            df = pd.DataFrame(pages_and_chunks)

            st.write("Filtering chunks")
            pages_and_chunks = df[
                df["chunk_token_count"] >= min_token_length
            ].to_dict(orient="records")

            st.write("Embedding chunks")
            embed_chunks(
                pages_and_chunks,
                get_from_session(st, SESSION_VARS.EMBEDDING_MODEL_CPU)
            )

            st.write("Saving embeddings")
            filename = save_embeddings(pages_and_chunks)

            put_to_session(st, SESSION_VARS.EMBEDDINGS_FILENAME, filename)
            put_to_session(st, SESSION_VARS.PROCESSED_DATA, True)

    
    # Retrieval + Generation

    if get_from_session(st, SESSION_VARS.PROCESSED_DATA):

        st.write("Vector search")
        tensor, pages_and_chunks = embeddings_to_tensor(
            get_from_session(st, SESSION_VARS.EMBEDDINGS_FILENAME)
        )

        scores, indices = retrieve_relevant_resources(
            query,
            tensor,
            get_from_session(st, SESSION_VARS.EMBEDDING_MODEL_CPU),
            st
        )

        context_items = [pages_and_chunks[i] for i in indices]
        for i, item in enumerate(context_items):
            item["score"] = scores[i].cpu()

        with st.expander("Retrieved context"):
            for item in context_items:
                st.write(f"Score: {item['score']:.4f}")
                st.write(item["sentence_chunk"])
                st.write(f"Page: {item['page_number']}")

        with st.expander("Answer"):
            with st.spinner("Generating answer"):
                if gen_variant == "vanilla":
                    input_ids, prompt = tokenize_with_chat(
                        get_from_session(st, SESSION_VARS.TOKENIZER),
                        query
                    )
                else:
                    input_ids, prompt = tokenize_with_rag_prompt(
                        get_from_session(st, SESSION_VARS.TOKENIZER),
                        query,
                        context_items
                    )

                answer = generate_answer(
                    get_from_session(st, SESSION_VARS.MODEL),
                    input_ids,
                    get_from_session(st, SESSION_VARS.TOKENIZER),
                    prompt
                )

                st.write(answer)

        st.success("Done!")
