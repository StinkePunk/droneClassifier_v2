import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import librosa
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import os
from tensorflow.keras.applications import (
    VGG16, InceptionV3, MobileNet, Xception, EfficientNetB0, ResNet50, DenseNet121
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler

os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "10"  # Timeout auf 10 Sekunden setzen
os.environ["TF_NUM_INTEROP_THREADS"] = "4" # Für Unterstützung von MultiThreadding
os.environ["TF_NUM_INTRA_THREADS"] = "4"   # Für Unterstützung von MultiThreadding

def get_base_model(model_name, input_shape=(128, 128, 3)):
    model_switch = {
        "VGG16": VGG16,
        "InceptionV3": InceptionV3,
        "MobileNet": MobileNet,
        "Xception": Xception,
        "EfficientNetB0": EfficientNetB0,
        "ResNet50": ResNet50,
        "DenseNet121": DenseNet121
    }
    return model_switch.get(model_name, MobileNet)(input_shape=input_shape, include_top=False, weights='imagenet')

def focal_loss(gamma=2., alpha=0.25):
    """
    Focal Loss für unbalancierte Klassifikation.

    Parameter:
    - gamma: Fokussierungsparameter. Höhere Werte fokussieren mehr auf schwierige Beispiele.
    - alpha: Gewichtung für die Minderheitsklasse.

    Returns:
    - Funktion zur Berechnung des Losses.
    """
    def focal_loss_fixed(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())  # Clipping, um Division durch 0 zu vermeiden
        pt = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)  # Wahrscheinlichkeiten für richtige Klassen
        return -K.mean(alpha * K.pow(1. - pt, gamma) * K.log(pt))  # Focal Loss-Formel
    return focal_loss_fixed

def train_classifier(train_features, y_train, val_features, y_val, 
                     model_name="MobileNet", trainable_layers=20, 
                     dropout_rate=0.5, loss_function="binary_crossentropy",
                     batchSize=64):
    num_classes = 1  # Annahme für die Anzahl der Drohnenklassen

    # base_model.trainable = False  # Deaktiviere alle Schichten zuerst
    base_model = get_base_model(model_name)
    for layer in base_model.layers:
        layer.trainable = False

    # Aktiviere nur die letzten Schichten
    if trainable_layers > 0:
        for layer in base_model.layers[-trainable_layers:]:
            layer.trainable = True

    # Anpassen des Inputs für VGG (3 Kanäle für das vortrainierte Modell)
    inputs = tf.keras.Input(shape=(128, 128, 1))
    x = layers.Conv2D(3, (3, 3), padding='same')(inputs)  # Anpassen an Eingabe
    x = layers.BatchNormalization()(x)  # BatchNormalization nach der Convolution-Schicht
    x = layers.Activation('relu')(x)  # Aktivierungsfunktion nach Conv2D

    # Hinzufügen der VGG-Schichten
    x = base_model(x, training=False)

    # Klassifikationsschichten hinzufügen
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Ausgabeschicht
    x = layers.Dense(num_classes, activation='sigmoid')(x)

    # Modell zusammenbauen
    model = models.Model(inputs, x)

    # Anpassung der Form für das Modell
    # train_features = np.expand_dims(train_features, axis=-1)
    # val_features = np.expand_dims(val_features, axis=-1)

    # Datenkonvertierung in float32
    train_features = np.expand_dims(train_features, axis=-1).astype('float32')
    val_features = np.expand_dims(val_features, axis=-1).astype('float32')

    # Normalisierung der Wertebereiche
    # Librosa Normalisierung
    #
    # train_features = librosa.util.normalize(train_features)
    # val_features = librosa.util.normalize(val_features)
    #
    # Z-Schore Normalisierung
    # train_features = (train_features - np.mean(train_features)) / np.std(train_features)
    # val_features = (val_features - np.mean(val_features)) / np.std(val_features)
    #
    # Min-Max-Skalierung: fit auf Train, apply auf Val; Per-Sample-Standardisierung (stabiler für Log-Mel)
    def _standardize(x):
        m = np.mean(x, axis=(1,2,3), keepdims=True)
        s = np.std(x,  axis=(1,2,3), keepdims=True) + 1e-8
        return (x - m) / s
    train_features = _standardize(train_features)
    val_features   = _standardize(val_features)
    scaler = None

    # Anpassung auf 4 Dimenionale Arrays
    # train_features = np.squeeze(train_features, axis=(3, 4, 5))
    # val_features = np.squeeze(val_features, axis=(3, 4, 5))

    # Labels kommen bereits als 0/1-Ints aus DroneClassifier.train()
    y_train = np.asarray(y_train, dtype='int32')
    y_val   = np.asarray(y_val,   dtype='int32')
    print("Encoded Labels - Train:", np.unique(y_train))
    print("Encoded Labels - Validation:", np.unique(y_val))

    # Da es wesentlich mehr "Drone" als "No drone" Daten gibt:
    
    # Berechnung der Klassengewichte
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    # Umwandeln in ein Dictionary
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print("Class weights:", class_weight_dict)
    
    # Fixe Klassengweichte
    # class_weight_dict = {0: 0.8, 1: 0.2}

    # Lernrate anpassen mit ExponentialDecay (funktioniert nicht zusammen mit ReduceLROnPlateau)
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #    initial_learning_rate=5e-4,  # Startwert der Lernrate
    #    decay_steps=10000,           # Nach wie vielen Schritten die Lernrate reduziert wird
    #    decay_rate=0.9               # Faktor der Reduktion
    #)

    # Adam-Optimizer mit LearningRateSchedule
    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule) # In Kombination mit tf.keras.optimizers.schedules.ExponentialDecay
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)      # In Kombination mit ReduceLROnPlateau

    # Kompilieren des Modells
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(curve="PR", name="auprc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

   
    # Reduziert die Lernrate, wenn sich der Validierungsverlust für 5 Epochen nicht verbessert (funktioniert nicht zusammen mit )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_auprc', mode='max',  # Beobachtet den Validierungsverlust
        factor=0.5,                       # Reduziert die Lernrate um 50 % (z. B. 1e-3 → 5e-4)
        patience=2,                       # Nach 4 Epochen ohne Verbesserung wird die Lernrate gesenkt
        min_lr=1e-6                       # Setzt eine Untergrenze für die Lernrate (damit sie nicht zu klein wird)
    )

    # Stoppt das Training, wenn sich der Validierungsverlust für 10 Epochen nicht verbessert
    early_stopping = EarlyStopping(
        monitor='val_auprc', mode='max',
        patience=5,         # Gibt dem Modell 10 Epochen Zeit zur Verbesserung
        restore_best_weights=True  # Stellt die besten Gewichte wieder her
    )


    # Modell trainieren
    history = model.fit(train_features, y_train, 
                        epochs=30, 
                        batch_size=batchSize,
                        validation_data=(val_features, y_val),
                        class_weight=class_weight_dict,
                        callbacks=[reduce_lr, early_stopping], # Early Stopping aktivieren
                        verbose = 5
    )

    return model, history, scaler
