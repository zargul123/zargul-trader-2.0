import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Enable oneDNN optimizations
os.environ['OMP_NUM_THREADS'] = '6'        # Match Ryzen 5 cores

def optimize_for_amd():
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('CPU')
    tf.config.set_logical_device_configuration(
        physical_devices[0],
        [tf.config.LogicalDeviceConfiguration() for _ in range(6)]
    )