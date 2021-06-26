# pip install   tensorflow==2.3.0
# pip install  tensorflow-text==2.3.0
# pip install  tf-models-official==2.3.0
# pip install jieba
import os
import re

import easyocr
import numpy
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from autogluon.vision import ImagePredictor
from gluoncv.auto.data.dataset import ImageClassificationDataset

text
from official import nlp
from official.nlp import bert
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
from utils.constant import Constant

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class BertDataset:
    def __init__(self, x, y, label_map, empty_label=[], empty_images=[], data_size=1):
        self.x = self._preprocess(tf.constant(x))
        self.y = tf.constant(y)
        self.label_map = label_map
        self.classes = len(label_map)
        self.empty_images = empty_images
        self.empty_label = empty_label
        self.data_size = data_size

    def _preprocess(self, x):
        # wget https://tfhub.dev/tensorflow/bert_zh_preprocess/3
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
        preprocessor = hub.KerasLayer(Constant.BERT_PREPROCESS_MODEL)
        encoder_inputs = preprocessor(text_input)
        preprocess_model = tf.keras.Model(text_input, encoder_inputs)
        return preprocess_model(x)


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
        self.ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)

    def generate_bert_dataset(self, dataset_path, data_type="train"):
        def remove_special_chars(input):
            input = re.sub('[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~\s]+', "", input)
            return re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", "", input)

        x_array = []
        y_array = []
        label_map = {}
        empty_label = []
        empty_images = []
        data_size = 0
        for label_index, label in enumerate(sorted(os.listdir(dataset_path))):
            label_map[label] = label_index
            label_dir = os.path.join(dataset_path, label)
            for img_index, img in enumerate(os.listdir(label_dir)):
                data_size += 1
                img = os.path.join(label_dir, img)
                # TODO batch input
                bounds = self.ocr_reader.readtext(img)
                orig_str = ''
                for ret in bounds:
                    text = remove_special_chars(ret[1])
                    # data augmentation for trainval ,
                    # need vocab length > 1
                    if data_type != "test" and len(orig_str) > 1:
                        x_array.append(text)
                        y_array.append(label_index)
                    orig_str += text
                # need vocab length > 1
                # when testdata vocab is None, add the image to the empty list for image_cls inference
                if len(orig_str) < 1:
                    if data_type == "test":
                        empty_label.append(label_index)
                        empty_images.append(img)
                else:
                    x_array.append(orig_str)
                    y_array.append(label_index)
        bert_dataset = BertDataset(x=x_array, y=y_array, label_map=label_map, empty_label=empty_label,
                                   empty_images=empty_images, data_size=data_size)

        return bert_dataset

    # self._generate_dataset(self.train_dataset,"train"), self._generate_dataset(
    #             self.val_dataset,"val"),

    def fit(self, train_dataset, val_dataset, test_dataset, epochs=5, batch_size=32, lr=2e-5):
        bert_config = bert.configs.BertConfig.from_dict(self.bert_config_dict)
        bert_classifier, bert_encoder = bert.bert_models.classifier_model(bert_config,
                                                                          num_labels=len(train_dataset.label_map))
        checkpoint = tf.train.Checkpoint(encoder=bert_encoder)
        # pip install gsutil
        # gsutil cp -R gs://cloud-tpu-checkpoints/bert/v3/chinese_L-12_H-768_A-12.tar.gz chinese_L-12_H-768_A-12
        checkpoint.read(Constant.BERT_CHECKPOINT).assert_consumed()

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
        results = bert_classifier.evaluate(test_dataset.x, test_dataset.y)
        print(results)
        tf.saved_model.save(bert_classifier, export_dir=self.task_config.get("output_path", "./saved_model"))

    def ensemble_eval(self, bert_model_path='/home/kaiyuan.xu/pycharm/framework/expert/saved_model-skip-empty',
                      image_model_path='/home/kaiyuan.xu/pycharm/framework/expert/Store-type-recognition/default/20210626-1024/checkpoint/predictor.ag'):
        # generator image cls dataset and predictor for non-text data
        test_bert_data = self.generate_bert_dataset(self.test_dataset, "test")
        test_dataset = ImageClassificationDataset({'image': test_bert_data.empty_images})
        predictor = ImagePredictor.load(image_model_path)
        cls_model_result = predictor.predict(test_dataset)
        print(cls_model_result)
        top1 = []
        topk = 5 if test_bert_data.classes > 5 else test_bert_data.classes
        for i in range(0, len(cls_model_result), topk):
            top1.append(i)
        cls_model_result = cls_model_result.iloc[top1]
        a=(cls_model_result["class"] == test_bert_data.empty_label).value_counts()[0]

        # only predict for text data
        bert_model = tf.saved_model.load(bert_model_path)
        # my_examples
        bert_result = bert_model([test_bert_data.x['input_word_ids'],
                                  test_bert_data.x['input_mask'],
                                  test_bert_data.x['input_type_ids']], training=False).numpy()
        y_pred = bert_result.argmax(axis=1)
        y_label = test_dataset.y.numpy()
        acc = numpy.sum((y_label == y_pred) != 0)
        print(acc)
