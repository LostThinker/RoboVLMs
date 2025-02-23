from torch.utils.data import DataLoader
from lightning.pytorch.utilities.combined_loader import CombinedLoader

iterables = [DataLoader(range(6), batch_size=4), DataLoader(range(15), batch_size=5)]
combined_loader = CombinedLoader(iterables, 'max_size_cycle')
_ = iter(combined_loader)



for batch, batch_idx, dataloader_idx in combined_loader:
    print(f"{batch}, {batch_idx=}, {dataloader_idx=}")