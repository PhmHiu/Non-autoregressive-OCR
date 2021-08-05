
class Hparams:
    def __init__(self):
        ### data and save path
        self.train_record_path = 'ocr/tfRecord_dataset/image_info.train'
        self.num_train_sample = 1690388
        self.valid_record_path = 'ocr/tfRecord_dataset/image_info.valid'
        self.charset_path = 'Non-autoregressive-OCR/charsets/charset_size=94.txt'
        self.num_valid_sample = 7249
        self.save_path = 'training_checkpoints'
        self.save_best = False
        self.max_to_keep = 1000
        self.augment = False

        ### model name
        self.model_name = 'cnn_sa_ctc' #'cnn_ctc'

        ### input params
        self.image_shape = (50, 250, 3)
        self.nul_code = 93
        self.charset_size = 94
        self.max_char_length = 15

        ### conv_tower params
        # base model from tf.keras.application, or custom instance of tf.keras.Model
        # check for new models from https://www.tensorflow.org/api_docs/python/tf/keras/applications
        # check for newest model from tf-nightly version
        self.base_model_name = 'InceptionResNetV2'
        # last convolution layer from base model which extract features from
        # inception v3: mixed2 (mixed_5d in tf.slim inceptionv3)
        # inception resnet v2: (mixed_6a in tf.slim inception_resnet_v2)
        self.end_point = 'mixed_6a'
        # endcode cordinate feature to conv_feature
        self.use_encode_cordinate = False

        ### decoder
        if self.model_name in ['cnn_sa_ctc']:
            self.num_heads = 8
            self.num_layers = 4
            self.model_dim = 128
            self.ff_dim = 512 # model_dim * 4
            self.dropout_rate = 0.1

        ### training params
        self.batch_size = 128
        self.max_epochs = 1000
        self.lr = 0.0001

hparams = Hparams()
