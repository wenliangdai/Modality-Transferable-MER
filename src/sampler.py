import torch


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset.labels))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            print(label)
            exit(1)
            for __label in label:
                if __label in label_to_count:
                    label_to_count[__label] += 1
                else:
                    label_to_count[__label] = 1

        # weight for each sample
        # weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
        weights = []
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            this_weight = 0.0
            for __label in label:
                this_weight += 1.0 / label_to_count[__label]
            this_weight /= len(label)
            weights.append(this_weight)
        print(weights)
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, id_):
        label = dataset.labels[id_]
        label = label.nonzero().squeeze()
        return label

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples