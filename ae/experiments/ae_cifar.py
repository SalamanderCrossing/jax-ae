import mate
from ..models import AE
from ..data_loaders.cifar10_ae import get_loaders
from ..trainers.ae_trainer import TrainerModule

train_loader, val_loader, test_loader = get_loaders(
    dataset_path="data", train_batch_size=256, test_batch_size=256
)

c_hid = 32
latent_dim = 128
ae = AE(c_hid=c_hid, latent_dim=latent_dim)

trainer = TrainerModule(mate.default_checkpoint_location, ae, c_hid=32, latent_dim=128)
