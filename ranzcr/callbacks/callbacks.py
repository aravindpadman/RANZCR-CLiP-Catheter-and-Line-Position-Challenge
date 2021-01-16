from typing import List

class Callback:
    """Abstract base class used to build new callbacks"""
    def on_epoch_begin(self, model, **kwargs):
        """Called at the start of an epoch"""
        pass
    def on_epoch_end(self, model, **kwargs):
        """Called at the end of an epoch"""
        pass 
    def on_train_batch_begin(self, model, **kwargs):
        """Called at the beginning of a training batch"""
        pass
    def on_train_batch_end(self, model, **kwargs):
        """Called at the end of a training batch"""
        pass
    def on_validation_batch_begin(self, model, **kwargs):
        """Called at the beginning of a validation batch"""
        pass
    def on_validation_batch_end(self, model, **kwargs):
        """Called at the end of a validation batch"""
        pass
    def on_test_batch_begin(self, model, **kwargs):
        """Called at the beginning of a test batch"""
        pass
    def on_test_batch_end(self, model, **kwargs):
        """Called at the end of a test batch"""
        pass
    def on_train_begin(self, model, **kwargs):
        """Called at the beginning of training"""
        pass
    def on_train_end(self, model, **kwargs):
        """Called at the end of training"""
        pass
    def on_validation_begin(self, model, **kwargs):
        """Called at the beginning of  validation or evaluation"""
        pass
    def on_validation_end(self, model, **kwargs):
        """Called at the end of  validation or evaluation"""
        pass
    def on_test_begin(self, model, **kwargs):
        """Called at the start of  test or inference"""
        pass
    def on_test_end(self, model, **kwargs):
        """Called at the end of  test or inference"""
        pass

class CallbackHandler(object):
    """A container for abstracting a list of callbacks"""
    def __init__(self, callbacks: List[Callback], model):
        self.callbacks = callbacks
        self.model = model
    def __call__(self, state, **kwargs):
        for cb in self.callbacks:
            _ = getattr(cb, state.value)(self.model, **kwargs)

if __name__ == '__main__':
    "write some tests here"
    pass



