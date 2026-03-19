import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from script06_utils import load_and_preprocess_data

def get_model(model_type, input_shape, n_classes=10):
    """
    Construye y devuelve un modelo funcional basado en el tipo especificado ("cnn" o "mlp").
    """
    inputs = Input(shape=input_shape, name='input_layer')
    
    if model_type == "cnn":
        # Arquitectura CNN - Basado en LeNet-5
        x = Conv2D(filters=6, kernel_size=(5, 5), padding='same', activation='relu', name='conv2d_1')(inputs)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pooling2d_1')(x)
        x = Conv2D(filters=16, kernel_size=(5, 5), activation='relu', name='conv2d_2')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pooling2d_2')(x)
        x = Flatten(name='flatten')(x)
        x = Dense(120, activation='relu', name='dense_1')(x)
        x = Dense(84, activation='relu', name='dense_2')(x)
        
    elif model_type == "mlp":
        # Arquitectura MLP Original
        x = Flatten(name='flatten')(inputs)
        x = Dense(64, activation='relu', name='dense_1')(x)
        x = Dense(32, activation='relu', name='dense_2')(x)
        x = Dense(16, activation='relu', name='dense_3')(x)
    else:
        raise ValueError("model_type debe ser 'cnn' o 'mlp'")

    # Capa de Clasificación
    outputs = Dense(n_classes, activation='softmax', name='output')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name=f'{model_type.upper()}_funcional')
    return model


def main():
    print("Python version:", sys.version.split(' ')[0])
    print("TensorFlow version:", tf.__version__)
    print("Keras version:", keras.__version__)
    
    # 1. Elección de modelo por argumento de consola
    model_type = "cnn"  # Por defecto usará CNN
    if len(sys.argv) > 1:
        arg_type = sys.argv[1].lower()
        if arg_type in ["cnn", "mlp"]:
            model_type = arg_type
        else:
            print("[WARN] Argumento no reconocido. Use 'cnn' o 'mlp'. Usando CNN por defecto.")
            
    print(f"\n[INFO] Configurando entrenamiento para modelo tipo: {model_type.upper()}")

    # 2. Carga del conjunto de datos y normalización centralizada
    (X_train_norm, y_train), (X_test_norm, y_test) = load_and_preprocess_data()
    
    # 3. Construir Arquitectura Dinámica
    input_shape = X_train_norm.shape[1:]
    n_classes = 10
    
    model = get_model(model_type, input_shape, n_classes)
    model.summary()
    
    # 4. Compilación del modelo
    lr = 0.001
    model.compile(
        loss='sparse_categorical_crossentropy', 
        optimizer=Adam(lr), 
        metrics=['accuracy']
    )
    
    # 5. Entrenamiento
    print("\n[INFO] Iniciando entrenamiento...")
    
    # Configurar parámetros particulares según el modelo originario
    epochs = 15 if model_type == "cnn" else 20
    validation_split = 0.2 if model_type == "cnn" else 0.7 
    
    history = model.fit(
        X_train_norm,
        y_train,
        epochs=epochs,
        batch_size=256,
        validation_split=validation_split,
        verbose=1
    )
    
    # 6. Gráficas de entrenamiento
    plt.figure(figsize=(10,3))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy during training ({model_type.upper()})')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss during training ({model_type.upper()})')
    plt.legend()
    
    plt.tight_layout()
    
    os.makedirs("figures", exist_ok=True)
    plt.savefig(os.path.join("figures", f"training_loss_accuracy_{model_type}.png"))
    plt.close()
    
    # 7. Evaluación
    print("\n[INFO] Evaluando en conjunto de test...")
    test_loss, test_acc = model.evaluate(X_test_norm, y_test)
    print(f"Test accuracy: {test_acc:.4f}\n")
    
    # 8. Reporte y Matriz de Confusión
    y_pred = model.predict(X_test_norm)
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    report_classification = classification_report(y_test, y_pred_labels, digits=4)
    print("Reporte de clasificación:\n", report_classification)
    
    cm = confusion_matrix(y_test, y_pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Purples')
    plt.title(f'Matriz de confusión - {model_type.upper()}-based model')
    plt.savefig(os.path.join("figures", f"confusion_matrix_{model_type}.png"))
    plt.close()
    
    # 9. Guardar el modelo en disco
    model_save_path = os.path.join(os.getcwd(), "modelos")
    os.makedirs(model_save_path, exist_ok=True)
    
    model_name = f"modelo_{model_type.upper()}_imagenes_fashion.h5"
    full_model_path = os.path.join(model_save_path, model_name)
    
    model.save(full_model_path)
    
    print("===================================================")
    print("Modelo guardado correctamente")
    print("Ruta:", full_model_path)
    print("===================================================")

if __name__ == "__main__":
    main()
