import logging
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import signature_constants

def get_func_from_saved_model(saved_model_dir):
    saved_loaded_model = tf.saved_model.load(saved_model_dir)
    graph_func = saved_loaded_model.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    return graph_func, saved_loaded_model
  
if __name__ == "__main__":
    writer = tf.summary.create_file_writer(logdir)
    tf.summary.trace_on(graph=True, profiler=True)
    logdir = './logs'
    # XXX: This assumes that TF->TFTRT model conversion has already been done. Use the location
    # of the converted saved model.
    converted_model_dir = './checkpoints/saved_model/converted'
    model_func, _ = get_func_from_saved_model(converted_model_dir)
    ## Perform a forward pass to instantiate the model graph.
    import numpy as np
    # XXX: use the same dimensions as the input size in (N, H, W, C) format.
    # For instance, this model uses images from the mnist database, so the
    # input size is (28x28x1).
    model_func(input_1=np.random.random((1, 28, 28, 1)).astype(np.float32))
    with writer.as_default():
        tf.summary.trace_export(
          name="my_func_trace",
          step=0,
          profiler_outdir=logdir)
    
    print("Model Imported. Visualize by running: "
          "tensorboard --logdir={}".format(logdir))      
