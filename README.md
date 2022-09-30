# FlowPredictor

Библиотека для запуска сохраненных моделей Tensorflow.

На данный момент библиотека принимает в качестве признаков вектор из `double` произвольной длины.
Результат вычислений должен быть типом `float`.

### Зависимости
* C++17
* cppflow
* libtensorflow C API


### Дополнительно

Имя input-слоя должно быть `serving_default_input_input`, имя output-слоя `StatefulPartitionedCall`.
Это соответствует модели, построенной с помощью кода Keras:
```
model = models.Sequential()
model.add(layers.Normalization( name="input") )
...
model.add(layers.Dense(1, activation='sigmoid', name="output"))
model.compile(...)
```