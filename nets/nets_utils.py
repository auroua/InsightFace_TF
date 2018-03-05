from tensorflow.python import pywrap_tensorflow
import collections
import numpy as np


var_stat = collections.namedtuple('stats', ['mean', 'median', 'std'])


def get_variables_in_checkpoint_file(file_name):
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        var_to_shape_map = reader.get_variable_to_shape_map()
        return var_to_shape_map
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed "
                  "with SNAPPY.")


def get_tensor_static_val(file_name, all_tensors, all_tensor_names):
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    vars_dict = {}
    if all_tensors or all_tensor_names:
      var_to_shape_map = reader.get_variable_to_shape_map()
      for key in sorted(var_to_shape_map):
        if all_tensors:
          vars_dict[key] = var_stat(np.mean(reader.get_tensor(key)), np.median(reader.get_tensor(key)),
                                    np.std(reader.get_tensor(key)))
    return vars_dict