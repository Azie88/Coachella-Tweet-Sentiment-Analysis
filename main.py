from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax
import gradio as gr

# Load model and tokenizer
model_path = "Azie88/Coachella_sentiment_analysis_roberta"

tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Preprocessing to clean up usernames and links
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Sentiment prediction returning styled HTML
def sentiment_analysis(text):
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores_ = output[0][0].detach().numpy()
    scores_ = softmax(scores_)

    labels = ['Negative', 'Neutral', 'Positive']
    emojis = ['😠', '😐', '😊']
    colors = ['red', 'gray', 'green']

    top_idx = scores_.argmax()
    label = labels[top_idx]
    emoji = emojis[top_idx]
    color = colors[top_idx]
    confidence = round(scores_[top_idx] * 100, 2)

    # Styled HTML output
    result = f"""
    <div style="text-align: center; font-size: 2rem;">
        {emoji} <span style="color: {color};">{label}</span><br>
        <small style="font-size: 1rem;">Confidence: {confidence}%</small>
    </div>
    """
    return result

# Gradio UI
demo = gr.Interface(
    fn=sentiment_analysis,
    inputs=gr.Textbox(placeholder="Type a tweet about #Coachella (e.g., 'Lineup is 🔥🔥🔥')", lines=3, label="Tweet Text"),
    outputs=gr.HTML(),
    theme=gr.themes.Base(),
    examples=[
        ["OMG the Coachella lineup is 🔥"],
        ["This lineup is so trash…"],
        ["Coachella starts tomorrow."]
    ],
    title='🎶 Coachella Tweet Sentiment Analyzer',
    description="Analyze if a tweet related to the #Coachella festival has a Positive, Neutral, or Negative sentiment."
)

demo.launch()

