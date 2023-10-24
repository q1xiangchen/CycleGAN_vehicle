import os
from torch.utils.tensorboard import SummaryWriter

class Logger(object):
    """Tensorboard logger."""
    
    def __init__(self, log_dir, experiment_name):
        """Initialize summary writer."""
        log_dir = os.path.join(log_dir, experiment_name)
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        self.writer.add_scalar(tag, value, step)
