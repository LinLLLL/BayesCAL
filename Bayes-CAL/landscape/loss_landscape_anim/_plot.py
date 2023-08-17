# pylint: disable = no-member, unused-variable
import warnings

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
from matplotlib.animation import FuncAnimation


def _plot_multiclass_decision_boundary(model, data, ax=None):
    X, y = data
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    X_test = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()
    y_pred = model(X_test, apply_softmax=True)
    _, y_pred = y_pred.max(dim=1)
    y_pred = y_pred.reshape(xx.shape)
    if not ax:
        plt.contourf(xx, yy, y_pred, cmap=plt.cm.Spectral, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
    else:
        ax.contourf(xx, yy, y_pred, cmap=plt.cm.Spectral, alpha=0.8)
        ax.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
        ax.xlim(xx.min(), xx.max())
        ax.ylim(yy.min(), yy.max())


def _animate_decision_area(
    model, X_train, y_train, steps, giffps, write2gif=False, file="nn_decision"
):
    # (W, loss, acc) in steps
    print(f"frames: {len(steps)}")
    weight_steps = [step[0] for step in steps]
    loss_steps = [step[1] for step in steps]
    acc_steps = [step[2] for step in steps]

    fig, ax = plt.subplots(figsize=(9, 6))
    W = weight_steps[0]
    model.init_from_flat_params(W)
    _plot_multiclass_decision_boundary(model, X_train, y_train)

    ax.set_title("DECISION BOUNDARIES")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

    # animation function. This is called sequentially
    def animate(i):
        W = weight_steps[i]
        model.init_from_flat_params(W)
        ax.clear()
        # This line is key!!
        ax.collections = []

        X = X_train
        y = y_train
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101)
        )

        X_test = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()
        y_pred = model(X_test, apply_softmax=True)
        _, y_pred = y_pred.max(dim=1)
        y_pred = y_pred.reshape(xx.shape)
        ax.contourf(xx, yy, y_pred, cmap=plt.cm.Spectral, alpha=0.8)
        ax.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)

        ax.set_title("DECISION BOUNDARIES")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        step_text = ax.text(
            0.05, 0.9, "", fontsize=10, ha="left", va="center", transform=ax.transAxes
        )
        value_text = ax.text(
            0.05, 0.8, "", fontsize=10, ha="left", va="center", transform=ax.transAxes
        )
        step_text.set_text(f"epoch: {i}")
        value_text.set_text(f"loss: {loss_steps[i]: .3f}\nacc: {acc_steps[i]: .3f}")

    # call the animator.  blit=True means only
    # re-draw the parts that have changed.
    # NOTE this anim must be global to work
    global anim
    anim = FuncAnimation(fig, animate, frames=len(steps), interval=200, blit=False)
    plt.ioff()

    # Write to gif
    if write2gif:
        anim.save(f"./{file}.gif", writer="imagemagick", fps=giffps)

    plt.show()


def _static_contour(steps, loss_grid, coords, pcvariances, filename="test.png"):
    _, ax = plt.subplots(figsize=(6, 4))
    coords_x, coords_y = coords
    ax.contourf(coords_x, coords_y, loss_grid, levels=35, alpha=0.9, cmap="YlGnBu")
    w1s = [step[0] for step in steps]
    w2s = [step[1] for step in steps]
    (pathline,) = ax.plot(w1s, w2s, color="r", lw=1)

    ax.set_title("MLP")
    ax.set_xlabel(f"principal component 0, {pcvariances[0]:.1%}")
    ax.set_ylabel(f"principal component 1, {pcvariances[1]:.1%}")
    plt.savefig(filename)
    print(f"{filename} created.")


def animate_contour(
    param_steps,
    loss_steps,
    acc_steps,
    loss_grid,
    coords,
    true_optim_point,
    true_optim_loss,
    pcvariances,
    giffps,
    TRAINER,
    SEED,
    sampling=False,
    max_frames=300,
    figsize=(9, 6),
    output_to_file=True,
    filename="test.gif",
):
    """Draw the frames of the animation.

    Args:
        param_steps: The list of full-dimensional flattened model parameters.
        loss_steps: The list of loss values during training.
        acc_steps: The list of accuracy values during training.
        loss_grid: The 2D slice of loss landscape.
        coords: The coordinates of the 2D slice.
        true_optim_point: The coordinates of the minimum point in the loss grid.
        true_optim_loss: The loss value of the minimum point.
        pcvariances: Variances explained by the principal components.
        giffps: Frames per second in the output.
        sampling (optional): Whether to sample from the steps. Defaults to False.
        max_frames (optional): Max number of frames to sample. Defaults to 300.
        figsize (optional): Figure size. Defaults to (9, 6).
        output_to_file (optional): Whether to write to file. Defaults to True.
        filename (optional): Defaults to "test.gif".
    """
    if sampling:
        print(f"\nSampling {max_frames} from {len(param_steps)} input frames.")
        param_steps = sample_frames(param_steps, max_frames)
        loss_steps = sample_frames(loss_steps, max_frames)
        acc_steps = sample_frames(acc_steps, max_frames)

    n_frames = len(param_steps)
    print(f"\nTotal frames to process: {n_frames}, result frames per second: {giffps}")

    fig, ax = plt.subplots(figsize=figsize)
    coords_x, coords_y = coords
    # norm = matplotlib.colors.Normalize(vmin=0, vmax=3.5)
    cs = ax.contourf(coords_x, coords_y, loss_grid, levels=35, alpha=0.9, cmap="YlGnBu")
    cmarker = ax.contour(coords_x, coords_y, loss_grid, [0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0],
                         colors='blue', linestyles='dashed')
    plt.clabel(cmarker, fontsize=10, colors='r', fmt='%.2f')
    cbar = plt.colorbar(cs)

    ax.set_title("Optimization in Loss Landscape")
    xlabel_text = "direction 0"
    ylabel_text = "direction 1"
    if pcvariances is not None:
        xlabel_text = f"principal component 0, {pcvariances[0]:.1%}"
        ylabel_text = f"principal component 1, {pcvariances[1]:.1%}"

    ax.set_xlabel(xlabel_text)
    ax.set_ylabel(ylabel_text)

    # W0 = param_steps[0]
    # w1s = [W0[0]]
    # w2s = [W0[1]]
    # (pathline,) = ax.plot(w1s, w2s, color="r", lw=1)
    w1s, w2s = [], []
    for i in range(len(param_steps)):
        W0 = param_steps[i]
        w1s.append(W0[0])
        w2s.append(W0[1])
    (optim_point,) = ax.plot(
        true_optim_point[0], true_optim_point[1], "bx", label="target local minimum", markersize=15)
    (pathline,) = ax.plot(w1s, w2s, color="r", lw=1)
    (point,) = ax.plot(w1s, w2s, "ro")
    plt.legend(loc="upper right")

    value_text = ax.text(
        0.05, 0.75, "", fontsize=20, ha="left", va="center", transform=ax.transAxes
    )
    step_text = ax.text(
        0.05, 0.9, "", fontsize=20, ha="left", va="center", transform=ax.transAxes
    )
    value_text_animate = ax.text(
        0.05, 0.75, "", fontsize=20, ha="left", va="center", transform=ax.transAxes
    )

    value_text.set_text(
        f"step: {1}\n"
        f"loss: {loss_steps[0]: .3f}\n"
        f"acc: {acc_steps[0]: .3f}\n\n"
        f"step: {20}\n"
        f"loss: {loss_steps[19]: .3f}\n"
        f"acc: {acc_steps[19]: .3f}\n\n"
        f"step: {50}\n"
        f"loss: {loss_steps[-1]: .3f}\n"
        f"acc: {acc_steps[-1]: .3f}\n\n"
        f"target coords: {true_optim_point[0]: .3f}, {true_optim_point[1]: .3f}\n"
        f"target loss: {true_optim_loss: .3f}"
    )
    plt.savefig('/home/zl/Dassl/CoOp/Figures/landscape_{}_seed{}.png'.format(TRAINER, SEED), dpi=600, bbox_inches='tight')

    def animate(i):
        W = param_steps[i]
        w1s.append(W[0])
        w2s.append(W[1])
        pathline.set_data(w1s, w2s)
        point.set_data(W[0], W[1])
        step_text.set_text(f"step: {i}")
        value_text_animate.set_text(
            f"step: {i}\n\n"
            f"loss: {loss_steps[i]: .3f}\n\n"
            f"acc: {acc_steps[i]: .3f}\n\n"
            f"target coords: {true_optim_point[0]: .3f}, {true_optim_point[1]: .3f}\n"
            f"target loss: {true_optim_loss: .3f}"
        )

    # Call the animator. blit=True means only re-draw the parts that have changed.
    # NOTE: anim must be global for the animation to work
    global anim
    anim = FuncAnimation(
        fig, animate, frames=len(param_steps), interval=100, blit=False, repeat=False
    )
    """
    if output_to_file:
        print(f"Writing {filename}.")
        anim.save(
            f"./{filename}",
            writer="imagemagick",
            fps=giffps,
            progress_callback=_animate_progress,
        )
        print(f"\n{filename} created successfully.")
    else:
        plt.ioff()
        plt.show()
    """


def _animate_progress(current_frame, total_frames):
    print("\r" + f"Processing {current_frame+1}/{total_frames} frames...", end="")
    if current_frame + 1 == total_frames:
        print("\nConverting to gif, this may take a while...")


def sample_frames(steps, max_frames):
    """Sample uniformly from given list of frames.

    Args:
        steps: The frames to sample from.
        max_frames: Maximum number of frames to sample.

    Returns:
        The list of sampled frames.
    """
    samples = []
    steps_len = len(steps)
    if max_frames > steps_len:
        warnings.warn(
            f"Less than {max_frames} frames provided, producing {steps_len} frames."
        )
        max_frames = steps_len
    interval = steps_len // max_frames
    counter = 0
    for i in range(steps_len - 1, -1, -1):  # Sample from the end
        if i % interval == 0 and counter < max_frames:
            samples.append(steps[i])
            counter += 1
    return list(reversed(samples))
