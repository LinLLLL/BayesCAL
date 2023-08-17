"""The main module with the loss_landscape_anim API.

Conceptual steps to produce the animation:
1. Load data
2. Create a pytorch lightning model
3. Record the parameters during training
4. Use PCA's top 2 PCs (or any 2 directions) to project the parameters to 2D
5. Collect the values in 2D:
    a. A list of 2D values as the trajectory obtained by projecting the
       parameters down to the 2D space.
    b. A 2D slice of the loss landscape (loss grid) that capture (a) with some
       adjustments for visual aesthetics.
"""
import pathlib

import pytorch_lightning as pl
import torch

from .loss_landscape import LossGrid, DimReduction
from ._plot import animate_contour


def loss_landscape_anim(
    path_info,
    train_samples_loader,
    model,
    parse_batch_train,
    TRAINER=None,
    seed=None,
    reduction_method="pca",  # "pca", "random", "custom" are supported
    custom_directions=None,
    output_to_file=True,
    output_filename="sample.gif",
    giffps=5,
    sampling=False,
    return_data=False,
):

    if seed:
        torch.manual_seed(seed)

    optim_path, loss_path, accu_path = path_info["optim_path_params"], path_info["loss"], path_info["acc"]

    # print(f"\n# sampled steps in optimization path: {len(optim_path)}")

    """Dimensionality reduction and Loss Grid"""
    print(f"Dimensionality reduction method specified: {reduction_method}")
    dim_reduction = DimReduction(
        params_path=optim_path,
        reduction_method=reduction_method,
        custom_directions=custom_directions,
        seed=seed,
    )
    reduced_dict = dim_reduction.reduce()
    path_2d = reduced_dict["path_2d"]
    directions = reduced_dict["reduced_dirs"]
    pcvariances = reduced_dict.get("pcvariances")

    loss_grid = LossGrid(
        optim_path=optim_path,
        model=model,
        data=train_samples_loader,
        loader=parse_batch_train,
        path_2d=path_2d,
        directions=directions,
    )

    animate_contour(
        param_steps=path_2d.tolist(),
        loss_steps=loss_path,
        acc_steps=accu_path,
        loss_grid=loss_grid.loss_values_2d,
        coords=loss_grid.coords,
        true_optim_point=loss_grid.true_optim_point,
        true_optim_loss=loss_grid.loss_min,
        pcvariances=pcvariances,
        giffps=giffps,
        TRAINER=TRAINER,
        SEED=seed,
        sampling=sampling,
        output_to_file=output_to_file,
        filename=output_filename,
    )
    if return_data:
        return list(optim_path), list(loss_path), list(accu_path)
