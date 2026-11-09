"""Nikon physical penalty; field input is center ABS + edge DIFF, in waves."""
import torch

VERSION = 'nikon_fullfield_pseudohuber_v2'

def physical_per_sample(fields, threshold=.07):
    # Unlike ell/(1+ell), this penalty retains finite, nonvanishing pressure
    # for large edge errors. Existing equal mean/worst field weights remain.
    terms=torch.sqrt(1+(fields/threshold).square())-1
    return .5*terms.mean(1)+.5*terms.amax(1)
