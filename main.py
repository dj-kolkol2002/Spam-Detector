import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import gradio as gr

# --- 1. TRENOWANIE MODELU ---
print("Trwa ładowanie i trenowanie modelu. Proszę czekać...")
# Wczytujemy plik z Kaggle (wymagane kodowanie latin-1)
df = pd.read_csv('spam.csv', encoding='latin-1')

# Formatowanie danych pod nasz model
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

# Dzielenie danych i wektoryzacja tekstu
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label_num'], test_size=0.2, random_state=42)
vectorizer = CountVectorizer()
X_train_dtm = vectorizer.fit_transform(X_train)

# Trening modelu Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_dtm, y_train)
print("Model jest gotowy do pracy!")

# --- 2. SŁOWNIK TŁUMACZEŃ ---
translations = {
    "English": {
        "title": "# SMS Spam Detector 📩🤖",
        "desc": "Paste a suspicious message to check if our AI model classifies it as spam.",
        "input_label": "Message Content",
        "input_placeholder": "Paste or type the SMS content here...",
        "output_label": "AI Verdict",
        "btn_submit": "Check message",
        "btn_clear": "Clear",
        "spam_msg": "🚨 WARNING! This is most likely SPAM.",
        "ham_msg": "✅ Normal, safe message (HAM).",
        "empty_msg": "Please enter a message!"
    },
    "Polski": {
        "title": "# Wykrywacz Spamu SMS 📩🤖",
        "desc": "Wklej podejrzaną wiadomość i sprawdź, czy nasz model AI uzna ją za spam.",
        "input_label": "Treść wiadomości",
        "input_placeholder": "Wklej lub wpisz treść SMS-a tutaj...",
        "output_label": "Werdykt AI",
        "btn_submit": "Sprawdź wiadomość",
        "btn_clear": "Wyczyść",
        "spam_msg": "🚨 UWAGA! To najprawdopodobniej SPAM.",
        "ham_msg": "✅ Zwykła, bezpieczna wiadomość.",
        "empty_msg": "Wpisz jakąś wiadomość!"
    }
}


# --- 3. FUNKCJE LOGICZNE ---
def predict_spam(message, lang):
    """Ocenia tekst z użyciem modelu i zwraca wynik w wybranym języku"""
    texts = translations[lang]
    if not message.strip():
        return texts["empty_msg"]

    # Przetwarzamy tekst przez wektoryzator i zgadujemy
    text_dtm = vectorizer.transform([message])
    prediction = nb_model.predict(text_dtm)[0]

    if prediction == 1:
        return texts["spam_msg"]
    else:
        return texts["ham_msg"]


def update_ui_language(lang):
    """Odświeża teksty w interfejsie graficznym po zmianie języka"""
    texts = translations[lang]
    return (
        gr.update(value=texts["title"]),
        gr.update(value=texts["desc"]),
        gr.update(label=texts["input_label"], placeholder=texts["input_placeholder"]),
        gr.update(label=texts["output_label"]),
        gr.update(value=texts["btn_submit"]),
        gr.update(value=texts["btn_clear"])
    )


# --- 4. BUDOWA INTERFEJSU GRAFICZNEGO ---
# Kod CSS ukrywający wbudowaną stopkę (footer) Gradio
custom_css = """
footer {display: none !important;}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as app:
    # Wybór języka
    lang_dropdown = gr.Dropdown(choices=["English", "Polski"], value="English", label="Language / Język",
                                interactive=True)

    # Nagłówki
    title_text = gr.Markdown(translations["English"]["title"])
    desc_text = gr.Markdown(translations["English"]["desc"])

    # Główne okna na tekst
    with gr.Row():
        msg_input = gr.Textbox(lines=5, label=translations["English"]["input_label"],
                               placeholder=translations["English"]["input_placeholder"])
    output_text = gr.Text(label=translations["English"]["output_label"])

    # Przyciski
    with gr.Row():
        submit_btn = gr.Button(translations["English"]["btn_submit"], variant="primary")
        clear_btn = gr.Button(translations["English"]["btn_clear"])

    # --- AKCJE I ZDARZENIA ---
    # Co się dzieje po zmianie języka na liście:
    lang_dropdown.change(
        fn=update_ui_language,
        inputs=[lang_dropdown],
        outputs=[title_text, desc_text, msg_input, output_text, submit_btn, clear_btn]
    )

    # Co się dzieje po kliknięciu "Sprawdź":
    submit_btn.click(
        fn=predict_spam,
        inputs=[msg_input, lang_dropdown],
        outputs=[output_text]
    )

    # Co się dzieje po kliknięciu "Wyczyść":
    clear_btn.click(
        fn=lambda: "",
        inputs=[],
        outputs=[msg_input]
    )

# --- 5. URUCHOMIENIE APLIKACJI ---
if __name__ == "__main__":
    app.launch()