# pip install   tensorflow==2.3.0
# pip install  tensorflow-text==2.3.0
# pip install  tf-models-official==2.3.0
import os
import re

import easyocr
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

text
from official import nlp
from official.nlp import bert
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.run_classifier
import official.nlp.bert.tokenization
import official.nlp.data.classifier_data_lib
import official.nlp.modeling.losses
import official.nlp.modeling.models
import official.nlp.modeling.networks
from utils.constant import Constant


class BertDataset:
    def __init__(self, x, y, label_map):
        self.x = x
        self.y = y
        self.label_map = label_map


class OCRBertClassifier:
    bert_config_dict = {
        "attention_probs_dropout_prob": 0.1,
        "directionality": "bidi",
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pooler_fc_size": 768,
        "pooler_num_attention_heads": 12,
        "pooler_num_fc_layers": 3,
        "pooler_size_per_head": 128,
        "pooler_type": "first_token_transform",
        "type_vocab_size": 2,
        "vocab_size": 21128
    }

    def __init__(self, task_config, train_dataset, val_dataset, test_dataset):
        self.task_config = task_config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.ocr_reader = easyocr.Reader(['ch_sim', 'en'])

    def _preprocess(self, x):
        # wget https://tfhub.dev/tensorflow/bert_zh_preprocess/3
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
        preprocessor = hub.KerasLayer(Constant.BERT_PREPROCESS_MODEL)
        encoder_inputs = preprocessor(text_input)
        preprocess_model = tf.keras.Model(text_input, encoder_inputs)
        return preprocess_model(x)

    def _generate_dataset(self, dataset_path):
        def remove_special_chars(input):
            input = re.sub('[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~\s]+', "", input)
            return re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", "", input)

        x_array = []
        y_array = []
        label_map = {}
        for label_index, label in enumerate(os.listdir(dataset_path)):
            label_map[label] = label_index
            label_dir = os.path.join(dataset_path, label)
            for img in os.listdir(label_dir):
                img = os.path.join(label_dir, img)
                # TODO batch input
                bounds = self.ocr_reader.readtext(img)
                orig_str = ''.join([ret[1] for ret in bounds])
                curr_str = remove_special_chars(orig_str)
                x_array.append(curr_str)
                y_array.append(label_index)
        bert_dataset = BertDataset(x=self._preprocess(tf.constant(x_array)), y=tf.constant(y_array),
                                   label_map=label_map)

        return bert_dataset

    def generate_bert_dataset(self):
        return self._generate_dataset(self.train_dataset), self._generate_dataset(
            self.val_dataset), self._generate_dataset(self.test_dataset)

    def fit(self, epochs=5, batch_size=32, lr=2e-5):
        train_dataset, val_dataset, test_dataset = self.generate_bert_dataset()
        bert_config = bert.configs.BertConfig.from_dict(self.bert_config_dict)
        bert_classifier, bert_encoder = bert.bert_models.classifier_model(bert_config,
                                                                          num_labels=len(train_dataset.label_map))
        checkpoint = tf.train.Checkpoint(encoder=bert_encoder)
        # pip install gsutil
        # gsutil cp -R gs://cloud-tpu-checkpoints/bert/v3/chinese_L-12_H-768_A-12.tar.gz chinese_L-12_H-768_A-12
        checkpoint.read(Constant.BERT_CHECKPOINT)
        # .assert_consumed()

        train_data_size = len(train_dataset.x)
        steps_per_epoch = int(train_data_size / batch_size)
        num_train_steps = steps_per_epoch * epochs
        warmup_steps = int(epochs * train_data_size * 0.1 / batch_size)

        # creates an optimizer with learning rate schedule
        optimizer = nlp.optimization.create_optimizer(lr, num_train_steps=num_train_steps,
                                                      num_warmup_steps=warmup_steps)

        metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        bert_classifier.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        bert_classifier.fit(
            train_dataset.x, train_dataset.y,
            validation_data=(val_dataset.x, val_dataset.y),
            batch_size=batch_size,
            epochs=epochs)
        tf.saved_model.save(bert_classifier, export_dir=self.task_config.get("output_path", "./saved_model"))
