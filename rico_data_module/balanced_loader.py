import torch
from torch.utils.data import TensorDataset, DataLoader

# Building a balanced batch sampler
##################################

class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    """
    A custom batch sampler that generates balanced batches from real and synthetic indices.

    Args:
        real_indices (list): List of indices for real samples.
        synth_indices (list): List of indices for synthetic samples.
        batch_size (int): The desired batch size.

    Raises:
        ValueError: If either real_indices or synth_indices is None.

    Returns:
        An iterator that yields balanced batches of indices, where each batch contains a mix of real and synthetic samples.
    """

    def __init__(self, real_indices: list, synth_indices: list, batch_size:int):
        super().__init__()

        # Handling the case where one of the two datasets is empty
        if real_indices == None or synth_indices == None:
            raise ValueError("Real indices or synthetic indices are None")

        self.batch_size = batch_size

        # Calculate the number of real and synthetic samples in each batch
        n_reals = self.batch_size / (1 + len(synth_indices)/len(real_indices))
        assert n_reals >= 1, "Not enough real samples to create a uniform batch"
        n_reals = int(n_reals)
        n_synths = self.batch_size - n_reals

        # Randomly permute the real and synthetic indices
        real_perm = torch.randperm(len(real_indices))
        real_indices_permuted = real_indices[real_perm]
        self.real_batches = torch.split(real_indices_permuted, n_reals)

        synth_perm = torch.randperm(len(synth_indices))
        synth_indices_permuted = synth_indices[synth_perm]
        self.synth_batches = torch.split(synth_indices_permuted, n_synths)

    def __iter__(self):
        """
        Returns an iterator that yields balanced batches of indices.

        Yields:
            torch.Tensor: A batch of indices, where each batch contains a mix of real and synthetic samples.
        """
        for real_batch, synth_batch in zip(self.real_batches, self.synth_batches):
            batch = torch.cat((real_batch, synth_batch))
            batch = batch[torch.randperm(batch.size(0))]  # Uncomment to shuffle batches, comment for debugging
            yield batch

    def __len__(self):
        """
        Returns the minimum length between the real and synthetic batches.

        Returns:
            int: The minimum length between the real and synthetic batches.
        """
        return min(len(self.real_batches), len(self.synth_batches))
    

class UniqueDataSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data, batch_size:int):
        self.data = data
        self.batch_size = batch_size
        self.indices = torch.randperm(len(self.data))
    
    def __iter__(self):
        for batch in torch.split(self.indices, self.batch_size):
            yield batch
    
    def __len__(self):
        if len(self.data) % self.batch_size != 0:
            return len(self.data) // self.batch_size + 1
        else:
            return len(self.data) // self.batch_size
        
if __name__ == "__main__":


    Ns = 60
    Nr = 25
    L = 8


    # Building real and synth datas...
    ##################################
    real = torch.zeros(int(Nr), L, 1)

    synth = torch.arange(1,Ns +1)
    synth = torch.unsqueeze(synth, dim=1)
    synth = synth.repeat(1, L).unsqueeze(2) # (N, L, 1)


    # Building a dataset and a loader
    ##################################
    data = torch.cat([real, synth], dim=0)

    dataset = TensorDataset(data)
    real_indices = list(range(len(real)))
    synth_indices = list(range(len(real), len(data)))

    batch_sampler = BalancedBatchSampler(real_indices=real_indices, synth_indices=synth_indices, batch_size=12)
    loader = DataLoader(dataset, batch_sampler=batch_sampler)

    for i, batch in enumerate(loader):
        break

    print(batch[0].squeeze())



