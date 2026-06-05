from flask import Flask, render_template, request, jsonify
from googletrans import Translator, LANGUAGES

app = Flask(__name__)
translator = Translator()

@app.route('/')
def index():
    languages = {code: name.title() for code, name in LANGUAGES.items()}
    return render_template('index.html', languages=languages)

@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    text = data.get('text', '')
    src_lang = data.get('src_lang', 'auto')
    tgt_lang = data.get('tgt_lang', 'en')

    if not text.strip():
        return jsonify({'error': 'Please enter some text.'}), 400

    try:
        result = translator.translate(text, src=src_lang, dest=tgt_lang)
        return jsonify({
            'translated_text': result.text,
            'detected_language': LANGUAGES.get(result.src, result.src).title()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)