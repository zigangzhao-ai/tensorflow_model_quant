import tensorflow as tf
import sys
import os

if sys.version_info.major >= 3:
    import pathlib
else:
    import pathlib2 as pathlib



saved_models_root = "/workspace/zigangzhao/TensoFlowBDD/models/official/mnist/mnist_saved_model/"

saved_model_dir = str(sorted(pathlib.Path(saved_models_root).glob("*"))[-1])
#print(saved_model_dir)

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
#print(tflite_model)

tflite_models_dir = pathlib.Path("./mnist_tflite_models/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

tflite_model_file = tflite_models_dir/"mnist_model.tflite"
tflite_model_file.write_bytes(tflite_model)


tf.logging.set_verbosity(tf.logging.INFO)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.compat.v1.lite.constants.FLOAT16]

tflite_fp16_model = converter.convert()
tflite_model_fp16_file = tflite_models_dir/"mnist_model_quant_f16.tflite"
tflite_model_fp16_file.write_bytes(tflite_fp16_model)