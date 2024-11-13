from io import BytesIO
import streamlit as st
from audiorecorder import audiorecorder  # type: ignore
from dotenv import dotenv_values
from openai import OpenAI
from hashlib import md5
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams

env = dotenv_values(".env")
  
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072
QDRANT_COLLECTION_NAME = "notes"
AUDIO_TRANSCRIBE_MODEL = "whisper-1"

def get_openai_client():
    return OpenAI(api_key=st.session_state["openai_api_key"])

def transcribe_audio(audio_bytes):
    openai_client = get_openai_client()
    audio_file = BytesIO(audio_bytes)
    audio_file.name = "audio.mp3"
    transcript = openai_client.audio.transcriptions.create(
        file=audio_file,
        model=AUDIO_TRANSCRIBE_MODEL,
        response_format="verbose_json",
    )
    return transcript.text

if not st.session_state.get("openai_api_key"):
    if "OPENAI_API_KEY" in env:
        st.session_state["openai_api_key"] = env["OPENAI_API_KEY"]

    else:
        st.info("Wprowadź swój klucz API, inaczej nie skorzystasz z aplikacji")
        st.session_state["openai_api_key"] = st.text_input("Klucz API", type="password")
        if st.session_state["openai_api_key"]:
            st.rerun()

if not st.session_state.get("openai_api_key"):
    st.stop()

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
        url=env["QDRANT_URL"],
        api_key=env["QDRANT_API_KEY"],  # Используем строку, а не список
    )

def assure_db_collection_exists():
    qdrant_client = get_qdrant_client()
    if not qdrant_client.collection_exists(QDRANT_COLLECTION_NAME):
        print("Tworzę kolekcję")
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
    else:
        print("Kolekcja już istnieje")

def get_embedding(text):
    openai_client = get_openai_client()
    result = openai_client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIM,
    )
    return result.data[0].embedding

def add_note_to_db(note_text):
    qdrant_client = get_qdrant_client()
    count_response = qdrant_client.count(collection_name=QDRANT_COLLECTION_NAME)
    current_count = count_response.count
    point = PointStruct(
        id=current_count + 1,
        vector=get_embedding(text=note_text),
        payload={
            "text": note_text,
        },
    )
    qdrant_client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=[point])

def list_notes_from_db(query=None):
    qdrant_client = get_qdrant_client()
    if not query:
        notes = qdrant_client.scroll(collection_name=QDRANT_COLLECTION_NAME, limit=10)[0]
        result = []
        for note in notes:
            result.append({
                "text": note.payload["text"],
                "score": None,
            })
        return result
    else:
        notes = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=get_embedding(text=query),
            limit=10,
        )
        result = []
        for note in notes:
            result.append({
                "text": note.payload["text"],
                "score": note.score,
            })
        return result

# Inicjalizacja stanu sesji
if "note_audio_bytes_md5" not in st.session_state:
    st.session_state["note_audio_bytes_md5"] = None
if "note_audio_bytes" not in st.session_state:
    st.session_state["note_audio_bytes"] = None
if "note_text" not in st.session_state:
    st.session_state["note_text"] = ""
if "note_audio_text" not in st.session_state:
    st.session_state["note_audio_text"] = ""

st.title("Zoe LAB 😘")
assure_db_collection_exists()

# Zakładki "Dodaj notatkę" i "Wyszukaj notatkę"
add_tab, search_tab = st.tabs(["Dodaj notatkę", "Wyszukaj notatkę"])

# Zakładka "Dodaj notatkę"
with add_tab:
    note_audio = audiorecorder(
        start_prompt="Nagraj",
        stop_prompt="Stop",
        pause_prompt="Pauza",
        show_visualizer=True,
        key="audio_recorder"
    )

    if note_audio:
        audio = BytesIO()
        note_audio.export(audio, format="mp3")
        st.session_state["note_audio_bytes"] = audio.getvalue()
        current_md5 = md5(st.session_state["note_audio_bytes"]).hexdigest()
        if st.session_state["note_audio_bytes_md5"] != current_md5:
            st.session_state["note_audio_text"] = ""
            st.session_state["note_text"] = ""
            st.session_state["note_audio_bytes_md5"] = current_md5

        st.audio(st.session_state["note_audio_bytes"], format="audio/mp3")
        st.write("Nagranie zostało ukończone")

        note_title = st.text_input("Podaj nazwę nagrania", "Bez nazwy")

    uploaded_file = st.file_uploader("Wgraj plik audio (MP3)", type=["mp3"])
    if uploaded_file is not None:
        st.session_state["note_audio_bytes"] = uploaded_file.read()
        st.audio(st.session_state["note_audio_bytes"], format="audio/mp3")
        st.write("Wgrano plik audio.")
        note_title = st.text_input("Podaj nazwę wgranego nagrania", "Bez nazwy")

    if st.button("Zapisz notatkę głosową"):
        with open(f"{note_title}.mp3", "wb") as f:
            f.write(st.session_state["note_audio_bytes"])
            st.toast("Notatka głowa zapisana", icon="🎉")

        st.download_button(
            label="Ściągnij notatkę",
            data=st.session_state["note_audio_bytes"],
            file_name=f"{note_title}.mp3",
            mime="audio/mp3"
        )

        current_md5 = md5(st.session_state["note_audio_bytes"]).hexdigest()
        if st.session_state["note_audio_bytes_md5"] != current_md5:
            st.session_state["note_audio_text"] = ""
            st.session_state["note_audio_bytes_md5"] = current_md5

    if st.button("Transkrybuj audio"):
        st.session_state["note_audio_text"] = transcribe_audio(st.session_state["note_audio_bytes"])

    if st.session_state["note_audio_text"]:
        st.session_state["note_text"] = st.text_area("Edytuj notatkę tekstową", value=st.session_state["note_audio_text"])

    if st.session_state["note_text"] and st.button("Zapisz notatkę tekstową", disabled=not st.session_state["note_text"]):
        add_note_to_db(note_text=st.session_state["note_text"])
        st.toast("Notatka tekstowa zapisana", icon="🎉")

        st.download_button(
            label="Ściągnij notatkę tekstową",
            data=st.session_state["note_text"],
            file_name=f"{note_title}.txt",
            mime="text/plain"
        )

# Zakładka "Wyszukaj notatkę"
with search_tab:
    query = st.text_input("Wyszukaj notatkę")
    if st.button("Szukaj"):
        search_results = list_notes_from_db(query=query)
        for note in search_results:
            st.markdown(f"**Notatka:** {note['text']}")
            if note["score"]:
                st.markdown(f"*Dopasowanie:* {note['score']}")