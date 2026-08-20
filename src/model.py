from tensorflow.keras import layers, models
import tensorflow as tf

def conv_block(x, filters, kernel_size, strides=1, use_batchnorm=False):
    x = layers.Conv1D(filters, kernel_size=kernel_size, strides=strides,
                       padding='same', use_bias=not use_batchnorm)(x)
    if use_batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def build_cnn_feature_extractor(input_shape, fs, dropout_cnn=0.5, use_batchnorm=False):
    inputs = layers.Input(shape=input_shape)
    s = conv_block(inputs, 64, fs // 2, strides=6, use_batchnorm=use_batchnorm)
    s = layers.MaxPooling1D(pool_size=8, strides=8)(s)
    s = layers.Dropout(dropout_cnn)(s)
    s = conv_block(s, 128, 8, use_batchnorm=use_batchnorm)
    s = conv_block(s, 128, 8, use_batchnorm=use_batchnorm)
    s = layers.MaxPooling1D(pool_size=4, strides=4)(s)
    s = layers.Flatten()(s)
    l = conv_block(inputs, 64, fs * 4, strides=50, use_batchnorm=use_batchnorm)
    l = layers.MaxPooling1D(pool_size=4, strides=4)(l)
    l = layers.Dropout(dropout_cnn)(l)
    l = conv_block(l, 128, 6, use_batchnorm=use_batchnorm)
    l = conv_block(l, 128, 6, use_batchnorm=use_batchnorm)
    l = layers.MaxPooling1D(pool_size=2, strides=2)(l)
    l = layers.Flatten()(l)
    merged = layers.Concatenate()([s, l])
    merged = layers.Dropout(dropout_cnn)(merged)
    return models.Model(inputs, merged, name="cnn_feature_extractor")

def build_cnn_lstm(seq_len, sample_shape, fs=100, n_classes=5, learning_rate=1e-3,
                    dropout_cnn=0.5, dropout_lstm=0.3, lstm_units=128, use_batchnorm=False):
    cnn = build_cnn_feature_extractor(sample_shape, fs, dropout_cnn=dropout_cnn, use_batchnorm=use_batchnorm)
    seq_input = layers.Input(shape=(seq_len, *sample_shape))
    features = layers.TimeDistributed(cnn, name="td_cnn")(seq_input)
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True, dropout=dropout_lstm), name="bi_lstm"
    )(features)
    x = layers.Dropout(dropout_lstm)(x)
    outputs = layers.TimeDistributed(
        layers.Dense(n_classes, activation='softmax'), name="td_output"
    )(x)
    model = models.Model(seq_input, outputs, name="cnn_lstm")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
