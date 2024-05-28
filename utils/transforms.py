import torch
from torch import nn
import torchvision as T
import torchvision.transforms.functional as TF
import numpy as np
import re
from torch._six import string_classes
import collections.abc as container_abcs
from torch.utils.data._utils.collate import default_convert

class PadImageToSize(object):

    def __init__(self, max_size, fill=0):
        self.max_size = max_size
        self.fill = fill

    def padding(self, img):
        w, h = img.size
        l = (self.max_size - w) // 2
        r = self.max_size - w - l
        t = (self.max_size - h) // 2
        b = self.max_size - h - t

        return [l, t, r, b]

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return TF.pad(img, padding=self.padding(img), fill=self.fill, padding_mode='constant')
    
    def __repr__(self):
        return self.__class__.__name__


def collate_based_on_len(batch):
    r"""Reimplement default_collate function to have different behaviour for data of same and varying length.
        If data field are of equal length, puts each data field into a tensor with outer dimension batch size.
        If data field is of varying length, simply convert it to tensor and keep the original structure without padding/truncation."""

    int_classes = int

    np_str_obj_array_pattern = re.compile(r'[SaUO]')

    default_collate_err_msg_format = (
        "default_collate: batch must contain tensors, numpy arrays, numbers, "
        "dicts or lists; found {}")

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        
        ## Start of reimplemented section. 
        # check if all elem in batch has same length
        if len(set([b.shape[0] for b in batch])) <= 1:
            _shape = [-1] + list(batch[0].shape)
            out = out.reshape(*_shape)
            return torch.stack(batch, 0, out=out)
        else:
            return default_convert(batch)
        ## End of reimplemented section.

    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return collate_based_on_len([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: collate_based_on_len([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate_based_on_len(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [collate_based_on_len(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))