import tensorflow as tf
import sys
import os

if sys.version_info.major >= 3:
    import pathlib
else:
    import pathlib2 as pathlib



saved_models_root = "/workspace1/zigangzhao/TensorFlowCar/models/official/mnist/mnist_saved_model"

saved_model_dir = str(sorted(pathlib.Path(saved_models_root).glob("*"))[-1])
print(saved_model_dir)


converter =  tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
#print(tflite_model)


tflite_models_dir = pathlib.Path("./mnist_tflite_models/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

tflite_model_file = tflite_models_dir/"mnist_model.tflite"
tflite_model_file.write_bytes(tflite_model)



converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
converter.post_training_quantize=True

tflite_fp16_model = converter.convert()
tflite_model_fp16_file = tflite_models_dir/"mnist_model_quant_fp16.tflite"
#tflite_model_fp16_file = tflite_models_dir/"mnist_model_quant_int8.pb"
tflite_model_fp16_file.write_bytes(tflite_fp16_model)

###test  ---error happend
import numpy as np
_, mnist_test = tf.keras.datasets.mnist.load_data()
images, labels = tf.cast(mnist_test[0], tf.float32)/255.0, mnist_test[1]
print(images)
mnist_ds = tf.data.Dataset.from_tensor_slices((images, labels)).batch(1)

# Load data for quantized model
images_uint8 = tf.cast(mnist_test[0], tf.uint8)
mnist_ds_uint8 = tf.data.Dataset.from_tensor_slices((images_uint8, labels)).batch(1)

interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()

interpreter_fp16 = tf.lite.Interpreter(model_path=str(tflite_model_fp16_file))
interpreter_fp16.allocate_tensors()

#interpreter_quant = tf.lite.Interpreter(model_path=str(tflite_model_quant_file))
#interpreter_quant.allocate_tensors()

def eval_model(interpreter, mnist_ds):
  total_seen = 0
  num_correct = 0

  input_index = interpreter.get_input_details()[0]["index"]
  print(input_index)
  output_index = interpreter.get_output_details()[0]["index"]
  for img, label in mnist_ds:
    total_seen += 1
    interpreter.set_tensor(input_index, img)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_index)
    if predictions == label.numpy():
      num_correct += 1

    if total_seen % 500 == 0:
      print("Accuracy after %i images: %f" %
            (total_seen, float(num_correct) / float(total_seen)))

  return float(num_correct) / float(total_seen)
# Create smaller dataset for demonstration purposes
mnist_ds_demo = mnist_ds.take(2000)

print(eval_model(interpreter, mnist_ds_demo))