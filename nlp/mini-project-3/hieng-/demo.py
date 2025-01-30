import numpy as np
import tensorflow as tf
from khmernltk import word_tokenize
from flask import Flask, request, jsonify, render_template
import utils


def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model


word_to_index, index_to_word, vocabs = utils.word_indexing("word_to_index.npy")
extended_word_to_index, extended_index_to_word, extended_vocabs = utils.word_indexing("extended_word_to_index.npy")

skip_model1 = load_model("model_word_prediction1.keras")
skip_model2 = load_model("model_word_prediction2.keras")
scratch_model1 = load_model("model_scratch_word_prediction1.keras")
scratch_model2 = load_model("model_scratch_word_prediction2.keras")
extended_model1 = load_model("pre_model_word_prediction1.keras")

app = Flask(__name__)


@app.route("/predict/<model>/<sentence>", methods=["GET"])
def predict(model, sentence):
    if model == "skip1":
        return jsonify(
            utils.predict_next_word(
                skip_model1, sentence, word_to_index, index_to_word, vocabs
            )
        )
    elif model == "scratch1":
        return jsonify(
            utils.predict_next_word(
                scratch_model1, sentence, word_to_index, index_to_word, vocabs
            )
        )
    elif model == "skip2":
        return jsonify(
            utils.predict_next_word(
                skip_model2, sentence, word_to_index, index_to_word, vocabs
            )
        )
    elif model == "scratch2":
        return jsonify(
            utils.predict_next_word(
                scratch_model2, sentence, word_to_index, index_to_word, vocabs
            )
        )
    elif model == "extended1":
        return jsonify(
            utils.predict_next_word(
                extended_model1, sentence, extended_word_to_index, extended_index_to_word, extended_vocabs, "<ចម>"
            )
        )
    else:
        return "Model not found", 404


# Serve the index page
@app.route("/")
def index():
    return render_template("./demo.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
