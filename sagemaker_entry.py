# ---------------------------------------------------------------------------- #
# Training functions                                                           #
# ---------------------------------------------------------------------------- #
from handwriting_recognition import run_main


def train(
        hyperparameters,
        input_data_config,
        channel_input_dirs,
        output_data_dir,
        model_dir,
        num_gpus,
        num_cpus,
        hosts,
        current_host,
        **kwargs):
    """
    [Required]

    Runs Apache MXNet training. Amazon SageMaker calls this function with information
    about the training environment. When called, if this function returns an
    object, that object is passed to a save function.  The save function
    can be used to serialize the model to the Amazon SageMaker training job model
    directory.

    The **kwargs parameter can be used to absorb any Amazon SageMaker parameters that
    your training job doesn't need to use. For example, if your training job
    doesn't need to know anything about the training environment, your function
    signature can be as simple as train(**kwargs).

    Amazon SageMaker invokes your train function with the following python kwargs:

    Args:
        - hyperparameters: The Amazon SageMaker Hyperparameters dictionary. A dict
            of string to string.
        - input_data_config: The Amazon SageMaker input channel configuration for
            this job.
        - channel_input_dirs: A dict of string-to-string maps from the
            Amazon SageMaker algorithm input channel name to the directory containing
            files for that input channel. Note, if the Amazon SageMaker training job
            is run in PIPE mode, this dictionary will be empty.
        - output_data_dir:
            The Amazon SageMaker output data directory. After the function returns, data written to this
            directory is made available in the Amazon SageMaker training job
            output location.
        - model_dir: The Amazon SageMaker model directory. After the function returns, data written to this
            directory is made available to the Amazon SageMaker training job
            model location.
        - num_gpus: The number of GPU devices available on the host this script
            is being executed on.
        - num_cpus: The number of CPU devices available on the host this script
            is being executed on.
        - hosts: A list of hostnames in the Amazon SageMaker training job cluster.
        - current_host: This host's name. It will exist in the hosts list.
        - kwargs: Other keyword args.

    Returns:
        - (object): Optional. An Apache MXNet model to be passed to the model
            save function. If you do not return anything (or return None),
            the save function is not called.
    """
    run_main(log_dir=output_data_dir, checkpoint_dir_path=model_dir)


def save(model, model_dir):
    """
    [Optional]

    Saves an Apache MXNet model after training. This function is called with the
    return value of train, if there is one. You are free to implement this to
    perform your own saving operation.

    Amazon SageMaker provides a default save function for
    Apache MXNet models. The default save function serializes 'Apache MXNet
    Module <https://mxnet.incubator.apache.org/api/python/module.html>' models.
    To rely on the default save function, omit a definition of
    'save' from your script. The default save function is discussed in more
    detail in the Amazon SageMaker Python SDK GitHub documentation.

    If you are using the Gluon API, you should provide your own save function,
    or save your model in the train function and let the train function
    complete without returning anything.

    Arguments:
       - model (object): The return value from train.
       - model_dir: The Amazon SageMaker model directory.
    """
    pass
