import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Path ke dataset yang sudah diunduh (pastikan path ini sesuai dengan struktur folder Anda)
dataset_dir = './dataset_split'  # Sesuaikan dengan path tempat dataset diunduh

train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'validation')

# Data augmentation untuk training dataset
train_datagen = ImageDataGenerator(
    rescale=1.0/255,           # Normalisasi citra
    rotation_range=20,         # Rotasi gambar acak
    #width_shift_range=0.2,     # Geser gambar secara horizontal
    #height_shift_range=0.2,    # Geser gambar secara vertikal
    #shear_range=0.3,           # Pemotongan gambar
    #zoom_range=0.2,            # Zoom in/out gambar
    horizontal_flip=True,       # Membalik gambar secara horizontal
    fill_mode='nearest'       # Pengisian area kosong setelah transformasi
)

# Normalisasi untuk validation dataset
val_datagen = ImageDataGenerator(rescale=1.0/255)

# Membaca data latih
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),    # Ukuran gambar yang akan diproses
    batch_size=32,             # Ukuran batch
    class_mode='categorical'   # Label kategori
)

# Membaca data validasi
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),    # Ukuran gambar yang akan diproses
    batch_size=32,             # Ukuran batch
    class_mode='categorical'   # Label kategori
)

# Ambil batch pertama dari train_generator
images, labels = next(train_generator)

# Tentukan jumlah gambar yang ingin ditampilkan
num_images = 6

# Plot gambar-gambar tersebut
plt.figure(figsize=(10, 10))

for i in range(num_images):
    plt.subplot(2, 3, i + 1)  # Menampilkan dalam grid 2x3
    plt.imshow(images[i])      # Menampilkan gambar
    plt.title(f"Label: {np.argmax(labels[i])}")  # Menampilkan label
    plt.axis('off')            # Menyembunyikan sumbu
plt.show()

# Fungsi untuk membuat Squeeze and Excitation Block
def squeeze_and_excitation_block(input_tensor, reduction_ratio=16):
    channel_axis = -1  # Last axis is the channel dimension
    se_tensor = layers.GlobalAveragePooling2D()(input_tensor)  # Squeeze
    se_tensor = layers.Reshape((1, 1, input_tensor.shape[-1]))(se_tensor)  # Reshape ke (1, 1, channels)
    
    # Fully connected layer (Excitation)
    se_tensor = layers.Dense(input_tensor.shape[-1] // reduction_ratio, activation='relu')(se_tensor)
    se_tensor = layers.Dense(input_tensor.shape[-1], activation='sigmoid')(se_tensor)
    
    # Kalikan input dengan bobot channel
    output_tensor = layers.multiply([input_tensor, se_tensor])  # Mengalikan input dengan pembobot
    return output_tensor

# Membangun model CNN dengan SENet
input_tensor = Input(shape=(224, 224, 3))  # Mendefinisikan input layer pertama

x = Conv2D(32, (3, 3), activation='relu')(input_tensor)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Menambahkan SE Block
x = squeeze_and_excitation_block(x)

x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Menambahkan SE Block lagi
x = squeeze_and_excitation_block(x)

# Global Average Pooling dan Fully Connected Layer
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
output_tensor = Dense(train_generator.num_classes, activation='softmax')(x)  # Kelas = jumlah kelas pada dataset

model = Model(inputs=input_tensor, outputs=output_tensor)  # Definisikan model dengan input dan output yang benar

model.summary()

# Kompilasi model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

from tensorflow.keras.callbacks import ModelCheckpoint

# Menyimpan model terbaik berdasarkan akurasi validasi
checkpoint = ModelCheckpoint('best_model.keras',  # Ganti dengan ekstensi .keras
                             save_best_only=True,
                             monitor='val_loss',  # Menyimpan model terbaik berdasarkan loss validasi
                             mode='min',
                             verbose=1)

# Melatih model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=30,  # Sesuaikan dengan jumlah epoch yang diinginkan
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    callbacks=[checkpoint],  # Menambahkan callback untuk menyimpan model terbaik
    verbose=1
)

# Evaluasi model pada data validasi
val_loss, val_accuracy = model.evaluate(val_generator, steps=val_generator.samples // val_generator.batch_size)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_accuracy}')

# Menyimpan model setelah pelatihan
model.save('/tmp/model/palmprint_model_final.keras')

# Memuat model yang sudah dilatih
model = load_model('/tmp/model/palmprint_model_final.keras')

# Memuat gambar baru untuk prediksi
img_path = 'dataset_split/train/001/001_F_L_30.JPG'  # Ganti dengan path gambar yang ingin diuji
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Menambah dimensi batch
img_array = img_array / 255.0  # Normalisasi gambar

# Melakukan prediksi
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)

print(f'Predicted class: {predicted_class[0]}')

from sklearn.metrics import f1_score, confusion_matrix

# Prediksi pada data validasi
val_pred = model.predict(val_generator)
val_pred_classes = np.argmax(val_pred, axis=1)

# Menghitung F1-Score
val_true_classes = val_generator.classes
f1 = f1_score(val_true_classes, val_pred_classes, average='macro')
print(f"F1-Score: {f1}")

# Plotting akurasi pelatihan dan validasi
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Plotting loss pelatihan dan validasi
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# Menghitung prediksi pada data validasi
val_pred = model.predict(val_generator)
val_pred_classes = np.argmax(val_pred, axis=1)

# Menghitung confusion matrix
cm = confusion_matrix(val_generator.classes, val_pred_classes)

# Menampilkan confusion matrix menggunakan heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=val_generator.class_indices.keys(), yticklabels=val_generator.class_indices.keys())
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Metrik evaluasi seperti precision, recall, F1-score
report = classification_report(val_generator.classes, val_pred_classes, target_names=val_generator.class_indices.keys())
print(report)

# Menghitung akurasi
accuracy = accuracy_score(val_generator.classes, val_pred_classes)
print(f'Validation Accuracy: {accuracy}')

