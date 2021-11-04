import torch

def module_numel(m: torch.nn.Module, only_trainable: bool = False):
    """
    returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`

    Parameters:
    -----------
    m : torch.nn.Module
        The module to check.
    only_trainable : bool
        If `True`, only trainable parameters are counted.

    Returns:
    --------
    numel : int
        The number of parameters in `m`.

    References:
    -----------
    Taken from : https://stackoverflow.com/a/62764464
    """
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)
