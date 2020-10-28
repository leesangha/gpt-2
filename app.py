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

from queue import Empty,Queue
import threading
import time

###################
requests_queue=Queue()
BATCH_SIZE = 1
CHECK_INTERVAL = 0.1
###################
#status=0

def makeModel(text,leng,k):
    try:
        model_name='774M'
        seed=None
        nsamples=1
        batch_size=1
        length=int(leng)
        temperature=1
        top_k=int(k)
        top_p=1
        models_dir='models'
        raw_text = text
        print('makeModel')
        
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

            print('raw_text in make model      :  '+raw_text)
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
                    print(text)
        return text.encode('ascii','ignore').decode('ascii')
    except Exception as e:
        print(e)
        return 500

#Queueing
def handle_requests_by_batch():
    try:
        while True:
            requests_batch=[]

            while not(len(requests_batch) >= BATCH_SIZE):
                try:
                    requests_batch.append(
                        requests_queue.get(timeout=CHECK_INTERVAL)
                    )
                except Empty:
                    continue
        
            batch_outputs=[]

            for request in requests_batch:
                print('run')
                #global status
                #status=1
                batch_outputs.append(
                    makeModel(request["input"][0],request["input"][1],request["input"][2])
                )
            
            for request, output in zip(requests_batch,batch_outputs):
                request["output"]=output

    except Exception as e:
        while not requests_queue.empty():
            requests_queue.get()
        print(e)

#Thread Start
threading.Thread(target=handle_requests_by_batch).start()

app = Flask(__name__, template_folder="templates", static_url_path="/static")

@app.route("/")
def main():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        print('predict')
        #global status
        #if status ==1:
        #    return jsonify({"message":"Too Many Requests"}),429
        
        if requests_queue.qsize() >=1:
            return jsonify({"message":"Too Many Requests"}),429
        
        text=request.form["message"]
        length=request.form["length"]
        top_k=request.form["top_k"]
        print('receive  ' + text +' ' + length + ' ' + top_k)

        req={"input":[text,length,top_k]}
        requests_queue.put(req)

        #Thread output response
        while "output" not in req:
            time.sleep(CHECK_INTERVAL)

        if req["output"] == 500:
            return jsonify({"error": "Error output is something wrong"}), 500
        #status=0
        result=text+req["output"]
        return jsonify({"message": result}), 200

    except Exception as e:
        print(e)

        return jsonify({"message": e}), 400



@app.route("/health")
def health():
    return res.sendStatus(200)


if __name__ == "__main__":
    from waitress import serve

    serve(app, host="0.0.0.0", port=80)
