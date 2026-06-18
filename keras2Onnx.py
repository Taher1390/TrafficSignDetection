import tensorflow as tf

model = tf.keras.models.load_model("traffic_sign.h5")

tf.saved_model.save(model, "saved_model")

# python -m tf2onnx.convert \
# --saved-model saved_model \
# --output traffic_sign.onnx \
# --opset 13