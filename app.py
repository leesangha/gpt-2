import os
import io
import uuid
import shutil
import sys

import fire
import json
import tensorflow as tf
import model, sample, encoder

from flask import Flask, render_template, flash, send_file, request, jsonify, url_for
import numpy as np
from werkzeug.utils import secure_filename


def interact_model(
    model_name='774M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=40,
    temperature=1,
    top_k=40,
    top_p=1,
    models_dir='models',
    raw_text='',
):
    print('raw _text' +raw_text)
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        print(type(raw_text))
        print(raw_text)
        context_tokens = enc.encode(raw_text)
        generated = 0
        for _ in range(nsamples // batch_size):
            out = sess.run(output, feed_dict={
                context: [context_tokens for _ in range(batch_size)]
            })[:, len(context_tokens):]
            for i in range(batch_size):
                generated += 1
                text = enc.decode(out[i])
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)

    return text.encode('ascii','ignore').decode('ascii')

app = Flask(__name__, template_folder="templates", static_url_path="/static")

@app.route("/")
def main():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        text=request.form["message"]
        result = fire.Fire(interact_model(raw_text=text))
        return jsonify({"message": result}), 200
    except Exception as e:
        print(e)

        return jsonify({"message": "Error! "}), 400



@app.route("/health")
def health():
    return res.sendStatus(200)


if __name__ == "__main__":
    from waitress import serve

    serve(app, host="0.0.0.0", port=80)
