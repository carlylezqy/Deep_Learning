import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,            # データセット全体の平均値を取得
    featurewise_std_normalization=True, # データを標準化する
    width_shift_range=0.1,  # 横サイズの0.1の割合でランダムに水平移動
    height_shift_range=0.1, # 縦サイズの0.1の割合でランダムに垂直移動
    rotation_range=10,      # 10度の範囲でランダムに回転させる
    zoom_range=0.1,         # ランダムに拡大
    horizontal_flip=True    # 左右反転
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,# データセット全体の平均値を取得
    featurewise_std_normalization=True, # データを標準化する
)

def prepare_data(data_generator=False):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train.astype('float32'), x_test.astype('float32')

    y_train, y_test = to_categorical(y_train), to_categorical(y_test)
    
    if(data_generator):
        train_datagen.fit(x_train)
        test_datagen.fit(x_test)

    return x_train, x_test, y_train, y_test

@tf.function
def train_step(x, t):
    with tf.GradientTape() as tape:
        outputs = model(x, training=True)
        tmp_loss = loss_fn(t, outputs)
        
    grads = tape.gradient(tmp_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    train_loss(tmp_loss)
    train_accuracy(t, outputs)

@tf.function
def valid_step(val_x, val_y):
    pred = model(val_x, training = False)
    tmp_loss = loss_fn(val_y, pred)
    val_loss(tmp_loss)
    val_accuracy(val_y, pred)

class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
    
        weight_decay = 1e-4

        self.std1 = tf.keras.layers.BatchNormalization()
        self.std2 = tf.keras.layers.BatchNormalization()
        self.std3 = tf.keras.layers.BatchNormalization()
        self.std4 = tf.keras.layers.BatchNormalization()
        self.std5 = tf.keras.layers.BatchNormalization()
        self.std6 = tf.keras.layers.BatchNormalization()

        # 第1層: 畳み込み層1 正則化を行う
        # (バッチサイズ,32,3,3) -> (バッチサイズ,32,32,32)
        self.conv2D_1 = tf.keras.layers.Conv2D(
            filters=32,                   # フィルター数32
            kernel_size=(3, 3),           # 3×3のフィルター
            padding='same',               # ゼロパディング
            input_shape=x_train.shape[1:], # 入力データの形状
            kernel_regularizer=tf.keras.regularizers.l2(
                weight_decay), # 正則化
            activation='relu'             # 活性化関数はReLU
            )

        # 第2層: 畳み込み層2: 正則化を行う
        # (バッチサイズ,32,32,32) ->(バッチサイズ,32,32,32)
        self.conv2D_2 = tf.keras.layers.Conv2D(
            filters=32,                   # フィルター数32
            kernel_size=(3, 3),           # 3×3のフィルター
            padding='same',               # ゼロパディング
            input_shape=x_train[0].shape, # 入力データの形状
            kernel_regularizer=tf.keras.regularizers.l2(
                weight_decay), # 正則化
            activation='relu'             # 活性化関数はReLU
            )

        # 第3層: プーリング層1: ウィンドウサイズは2×2
        # (バッチサイズ,32,32,32) -> (バッチサイズ,16,16,32)
        self.pool1 = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2))             # 縮小対象の領域は2×2
        # ドロップアウト1：ドロップアウトは20％
        self.dropput1 = tf.keras.layers.Dropout(0.2)

        # 第4層: 畳み込み層3　正則化を行う
        # (バッチサイズ,16,16,32) ->(バッチサイズ,16,16,64)
        self.conv2D_3 = tf.keras.layers.Conv2D(
            filters=64,                  # フィルターの数は64
            kernel_size=(3, 3),           # 3×3のフィルターを使用
            padding='same',               # ゼロパディングを行う
            kernel_regularizer=tf.keras.regularizers.l2(
                weight_decay),           # 正則化
            activation='relu'             # 活性化関数はReLU
            )
        
        # 第5層: 畳み込み層4: 正則化を行う
        # (バッチサイズ,64,16,16) ->(バッチサイズ,64,16,16)
        self.conv2D_4 = tf.keras.layers.Conv2D(
            filters=64,                  # フィルターの数は256
            kernel_size=(3, 3),           # 3×3のフィルターを使用
            padding='same',               # ゼロパディングを行う
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay), # 正則化
            activation='relu'             # 活性化関数はReLU
            )

        # 第6層: プーリング層2: ウィンドウサイズは2×2
        # (バッチサイズ,16,16,64) -> (バッチサイズ,8,8,64)
        self.pool2 = tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2))             # 縮小対象の領域は2×2
        # ドロップアウト2：ドロップアウトは30％
        self.dropput2 = tf.keras.layers.Dropout(0.3)

        # 第7層: 畳み込み層5: 正則化を行う
        # (バッチサイズ,8,8,64) -> (バッチサイズ,8,8,128)
        self.conv2D_5 = tf.keras.layers.Conv2D(
            filters=128,                  # フィルターの数は64
            kernel_size=(3, 3),           # 3×3のフィルターを使用
            padding='same',               # ゼロパディングを行う
            kernel_regularizer=tf.keras.regularizers.l2(
                weight_decay),           # 正則化
            activation='relu'             # 活性化関数はReLU
            )
        
        # 第8層: 畳み込み層6: 正則化を行う
        # (バッチサイズ,8,8,128) -> (バッチサイズ,8,8,128)
        self.conv2D_6 = tf.keras.layers.Conv2D(
            filters=128,                  # フィルターの数は256
            kernel_size=(3, 3),           # 3×3のフィルターを使用
            padding='same',               # ゼロパディングを行う
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay), # 正則化
            activation='relu'             # 活性化関数はReLU
            )

        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.dropput3 = tf.keras.layers.Dropout(0.4)
        self.flatten = tf.keras.layers.Flatten()

        self.fc1 =  tf.keras.layers.Dense(128, activation='relu')
        self.dropput4 = tf.keras.layers.Dropout(0.4)

        # 第11層: 出力層
        # (バッチサイズ,128) -> (バッチサイズ,10)
        self.fc2 =  tf.keras.layers.Dense(
            10,                           # 出力層のニューロン数は10
            activation='softmax')         # 活性化関数はソフトマックス

    @tf.function
    def call(self, x, training=None):
        x = self.std1(self.conv2D_1(x))
        x = self.pool1(self.std2(self.conv2D_2(x)))
        if training:
            x = self.dropput1(x)

        x = self.std3(self.conv2D_3(x))
        x = self.pool2(self.std4(self.conv2D_4(x)))
        if training:
            x = self.dropput2(x)

        x = self.std5(self.conv2D_5(x))
        x = self.pool3(self.std6(self.conv2D_6(x)))
        if training:
            x = self.dropput3(x)

        x = self.flatten(x)
        x = self.fc1(x)

        if training:
            x = self.dropput4(x)

        x = self.fc2(x)
        return x

loss_fn = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

train_loss = tf.keras.metrics.Mean()
val_loss = tf.keras.metrics.Mean()
train_accuracy = tf.keras.metrics.CategoricalAccuracy()
val_accuracy = tf.keras.metrics.CategoricalAccuracy()

epochs = 120
batch_size = 64

x_train, x_test, y_train, y_test = prepare_data(data_generator=True)
train_steps, val_steps = x_train.shape[0]//batch_size, x_test.shape[0]//batch_size

model = CNN()

history = {'train_loss':[],'train_accuracy':[], 'val_loss':[], 'val_accuracy':[]}

train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
validation_generator = test_datagen.flow(x_test, y_test, batch_size=batch_size)
 
for epoch in range(epochs): 
    step_counter = 0  
    for x_batch, t_batch in train_generator:
        train_step(x_batch, t_batch)
        step_counter += 1
        if step_counter >= train_steps:
            break
    
    v_step_counter = 0
    for x_val_batch, t_val_batch  in validation_generator:
        valid_step(x_val_batch, t_val_batch)
        v_step_counter += 1
        if v_step_counter >= val_steps:
            break
    
    avg_train_loss = train_loss.result()
    avg_train_acc = train_accuracy.result()
    avg_val_loss = val_loss.result()
    avg_val_acc = val_accuracy.result()

    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['train_accuracy'].append(avg_train_acc)
    history['val_accuracy'].append(avg_val_acc)
 
    if (epoch + 1) % 1 == 0:
        print(
        'epoch({}) train_loss: {:.4} train_acc: {:.4} val_loss: {:.4} val_acc: {:.4}'
        .format(epoch+1,avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc)
    )
 
model.summary()


'''
import matplotlib.pyplot as plt
# %matplotlib inline

# 学習結果（損失）のグラフを描画
plt.plot(history['loss'], marker='.', label='loss (Training)')
plt.plot(history['val_loss'], marker='.', label='loss (Validation)')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# 学習結果（精度）のグラフを描画
plt.plot(history['accuracy'], marker='.', label='accuracy (Training)')
plt.plot(history['val_accuracy'], marker='.', label='accuracy (Validation)')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
'''