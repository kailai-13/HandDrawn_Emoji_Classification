import tensorflow as tf
import tensorflow_datasets as tfds
import os

# Define the EmojiDrawings dataset class
class EmojiDrawings(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('0.1.0')
    emoji_names = [
        "beaming-face.png",
        "cloud.png",
        "face-spiral.png",
        "flushed-face.png",
        "grimacing-face.png",
        "grinning-face.png",
        "grinning-squinting.png",
        "heart.png",
        "pouting-face.png",
        "raised-eyebrow.png",
        "relieved-face.png",
        "savoring-food.png",
        "smiling-heart.png",
        "smiling-horns.png",
        "smiling-sunglasses.png",
        "smiling-tear.png",
        "smirking-face.png",
        "tears-of-joy.png",
    ]

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            supervised_keys=('image', 'label'),
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(shape=(400, 400, 4)),  # RGBA images
                'label': tfds.features.ClassLabel(names=EmojiDrawings.emoji_names),
            }),
        )

    def _split_generators(self, dl_manager):
        extracted_path = dl_manager.download_and_extract(
            f'https://github.com/FlafyDev/emoji-drawings/releases/download/v{EmojiDrawings.VERSION}/emoji_drawings_images.zip'
        )
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={'data_dir': extracted_path}
            ),
        ]

    def _generate_examples(self, data_dir):
        for filename in os.listdir(data_dir):
            if not filename.endswith(".png"):
                continue

            # Extract the emoji index from the filename (before the first '-')
            emoji_index = int(filename.split("-")[0])  # Extract the first number

            # Ensure the index is valid
            if 0 <= emoji_index < len(EmojiDrawings.emoji_names):
                image_path = os.path.join(data_dir, filename)
                yield filename, {
                    'image': image_path,
                    'label': EmojiDrawings.emoji_names[emoji_index],  # Map index to label
                }


# Register the dataset
tfds.core.DatasetBuilder.register(EmojiDrawings)

# Load the dataset from TensorFlow Datasets
(ds_train, ds_test), ds_info = tfds.load("emoji_drawings", split=['train[:80%]', 'train[80%:]'], as_supervised=True, with_info=True)

# Preprocessing function
def preprocess(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)  # Normalize (0,1)
    image = tf.image.resize(image, (64, 64))  # Resize to fit CNN
    return image, label

# Apply preprocessing
ds_train = ds_train.map(preprocess).batch(32).shuffle(1000).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

# Build CNN Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 4)),  # 4 channels (RGBA)
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Prevent overfitting
    tf.keras.layers.Dense(ds_info.features['label'].num_classes, activation='softmax')
])

# Compile Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train Model
history = model.fit(ds_train, validation_data=ds_test, epochs=10)

# Save the Model
model.save("emoji_recognition_cnn.h5")

# Display Model Summary
model.summary()
