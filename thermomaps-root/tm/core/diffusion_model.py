import torch
import os
import numpy as np

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def temperature_density_rescaling(std_temp, ref_temp):
    """
    Calculate temperature density rescaling factor.

    Args:
        std_temp (float): The standard temperature.
        ref_temp (float): The reference temperature.

    Returns:
        float: The temperature density rescaling factor.
    """
    return (std_temp / ref_temp).pow(0.5)


def identity(t, *args, **kwargs):
    """
    Identity function.

    Args:
        t: Input tensor.

    Returns:
        t: Input tensor.
    """
    return t


RESCALE_FUNCS = {
    "density": temperature_density_rescaling,
    "no_rescale": identity,
}


class DiffusionModel:
    """
    Base class for diffusion models.
    """

    def __init__(
        self,
        diffusion_process,
        backbone,
        loader,
        prior,
        pred_type,
        rescale_func_name="no_rescale",
        RESCALE_FUNCS=RESCALE_FUNCS,
        **kwargs
    ):
        """
        Initialize a DiffusionModel.

        Args:
            diffusion_process: The diffusion process.
            backbone: The backbone model.
            loader: Data loader.
            pred_type: Type of prediction.
            prior: Prior distribution.
            control_ref (float): Control reference temperature.
            rescale_func_name (str): Name of the rescaling function.
            RESCALE_FUNCS (dict): Dictionary of rescaling functions.
        """
        self.loader = loader
        self.BB = backbone
        self.DP = diffusion_process
        self.pred_type = pred_type
        self.rescale_func = RESCALE_FUNCS[rescale_func_name]
        self.prior = prior

    def noise_batch(self, b_t, t, prior, **prior_kwargs):
        """
        Wrapper which calls applies the (marginal) transition kernel
        of the forward noising process.
        """
        return self.DP.forward_kernel(b_t, t, prior, **prior_kwargs)

    def denoise_batch(self, b_t, t):
        """
        Wrapper which calls applies the (marginal) transition kernel
        of the reverse noising process.
        """
        return self.DP.reverse_kernel(b_t, t, self.BB, self.pred_type)

    def denoise_step(self, b_t, t, t_next, eta, **prior_kwargs):
        """
        Wrapper which calls applies the (marginal) transition kernel
        of the reverse noising process.
        """
        b_t_next = self.DP.reverse_step(b_t, t, t_next, self.BB, self.pred_type, eta, self.prior, **prior_kwargs)
        return b_t_next

    def sample_times(self, num_times):
        """
        Randomly sample times from the time-discretization of the
        diffusion process.
        """
        return torch.randint(
            low=0, high=self.DP.num_diffusion_timesteps, size=(num_times,)
        ).long()

    @staticmethod
    def get_adjacent_times(times):
        """
        Pairs t with t+1 for all times in the time-discretization
        of the diffusion process.
        """
        times_next = torch.cat((torch.Tensor([0]).long(), times[:-1]))
        return list(zip(reversed(times), reversed(times_next)))


class DiffusionTrainer(DiffusionModel):
    """
    Subclass of a DiffusionModel: A trainer defines a loss function and
    performs backprop + optimizes model outputs.
    """

    def __init__(
        self,
        diffusion_process,
        backbone,
        train_loader,
        prior,
        pred_type='noise',
        model_dir=None,
        test_loader = None,
        optim=None,
        scheduler=None,
        rescale_func_name="density",
        RESCALE_FUNCS=RESCALE_FUNCS,
        device=0,
        identifier="model"
    ):
        """
        Initialize a DiffusionTrainer.

        Args:
            diffusion_process: The diffusion process.
            backbone: The backbone model.
            loader: Data loader.
            pred_type: Type of prediction.
            prior: Prior distribution.
            optim: Optimizer.
            scheduler: Learning rate scheduler.
            rescale_func_name (str): Name of the rescaling function.
            RESCALE_FUNCS (dict): Dictionary of rescaling functions.
        """
        super().__init__(
            diffusion_process,
            backbone,
            train_loader,
            prior,
            pred_type,
            rescale_func_name,
            RESCALE_FUNCS,
        )

        self.model_dir = model_dir
        if self.model_dir:
            os.makedirs(self.model_dir, exist_ok=True)
        self.identifier = identifier
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.train_losses = []
        self.test_losses = []

    def loss_function(self, e, e_pred, weight, loss_type="l2"):
        """
        Loss function can be the l1-norm, l2-norm, or the VLB (weighted l2-norm).

        Args:
            e: Actual data.
            e_pred: Predicted data.
            weight: Weight factor.
            loss_type (str): Type of loss function.

        Returns:
            float: The loss value.
        """
        sum_indices = tuple(list(range(1, self.loader.num_dims)))

        def l1_loss(e, e_pred, weight):
            return (e - e_pred).abs().sum(sum_indices)
        
        def smooth_l1_loss(e, e_pred, weight):
            return torch.nn.functional.smooth_l1_loss(e, e_pred, reduction='mean')

        def l2_loss(e, e_pred, weight):
            return (e - e_pred).pow(2).sum((1, 2, 3)).pow(0.5).mean()

        def VLB_loss(e, e_pred, weight):
            return (weight * ((e - e_pred).pow(2).sum(sum_indices)).pow(0.5)).mean()

        def smooth_l1_loss(e, e_pred, weight):
            return torch.nn.functional.smooth_l1_loss(e, e_pred)

        loss_dict = {"l1": l1_loss, "l2": l2_loss, "VLB": VLB_loss, "smooth_l1": smooth_l1_loss}

        return loss_dict[loss_type](e, e_pred, weight)

    def train(
        self,
        num_epochs,
        grad_accumulation_steps=1,
        print_freq=None,
        batch_size=128,
        loss_type="l2",
    ):
        """
        Trains a diffusion model.

        Args:
            num_epochs (int): Number of training epochs.
            grad_accumulation_steps (int): Number of gradient accumulation steps.
            print_freq (int): Frequency of printing training progress.
            batch_size (int): Batch size.
            loss_type (str): Type of loss function.
        """

        train_loader = torch.utils.data.DataLoader(
            self.train_loader,
            batch_size=batch_size,
            shuffle=True,
        )
        if self.test_loader:
            test_loader = torch.utils.data.DataLoader(
                self.test_loader,
                batch_size=batch_size,
                shuffle=True,
            )

        for epoch in range(num_epochs):
            epoch_train_loss = []
            epoch += self.BB.start_epoch
            for i, (temperatures, b) in enumerate(train_loader, 0):
                t = self.sample_times(b.size(0))
                t_prev = t - 1
                t_prev[t_prev == -1] = 0
                weight = self.DP.compute_SNR(t_prev) - self.DP.compute_SNR(t)
                # logging.debug(f"{b.shape=}")
                target, output = self.train_step(b, t, self.prior, 
                    batch_size=len(b), temperatures=temperatures, sample_type="from_data") # prior kwargs

                loss = (self.loss_function(target, output, weight, loss_type=loss_type) / grad_accumulation_steps)

                if i % grad_accumulation_steps == 0:
                    self.BB.optim.zero_grad()
                    epoch_train_loss.append(loss.detach().cpu().numpy())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.BB.model.parameters(), 1.)
                    self.BB.optim.step()

                if print_freq:
                    if i % print_freq == 0:
                        print(f"step: {i}, loss {loss.detach():.3f}")

            if self.test_loader:
                with torch.no_grad():
                    epoch_test_loss = []
                    for i, (temperatures, b) in enumerate(test_loader, 0):
                        t = self.sample_times(b.size(0))
                        t_prev = t - 1
                        t_prev[t_prev == -1] = 0
                        weight = self.DP.compute_SNR(t_prev) - self.DP.compute_SNR(t)
                        target, output = self.train_step(b, t, self.prior, 
                            batch_size=len(b), temperatures=temperatures, sample_type="from_data")
                        loss = self.loss_function(target, output, weight, loss_type=loss_type)
                        epoch_test_loss.append(loss.detach().cpu().numpy())


            self.train_losses.append(np.mean(epoch_train_loss))
            if self.test_loader:
                self.test_losses.append(np.mean(epoch_test_loss))
                print(f"epoch: {epoch} | train loss: {self.train_losses[-1]:.3f} | test loss: {self.test_losses[-1]:.3f}")
            else:
                print(f"epoch: {epoch} | train loss: {self.train_losses[-1]:.3f}")

            # if self.BB.scheduler:
                # self.BB.scheduler.step()
            if self.model_dir:
                self.BB.save_state(self.model_dir, epoch, identifier=self.identifier)

    def train_step(self, b, t, prior, **kwargs):
        """
        Training step.

        Args:
            b: Input batch.
            t: Sampled times.
            prior: Prior distribution.
            kwargs: Additional keyword arguments.

        Returns:
            Tuple: (noise, noise_pred)
        """
        b_t, noise = self.noise_batch(b, t, prior, **kwargs)
        #print(b_t.shape, noise.shape)
        b_0, noise_pred = self.denoise_batch(b_t, t)
        if self.pred_type == "noise":
            return noise, noise_pred
        elif self.pred_type == "x0":
            return b, b_0


class DiffusionSampler(DiffusionModel):
    """
    Subclass of a DiffusionModel: A sampler generates samples from random noise.
    """

    def __init__(
        self,
        diffusion_process,
        backbone,
        loader,
        prior,
        pred_type='noise',
        sample_dir=None,
        rescale_func_name="density",
        RESCALE_FUNCS=RESCALE_FUNCS,
        **kwargs
    ):
        """
        Initialize a DiffusionSampler.

        Args:
            diffusion_process: The diffusion process.
            backbone: The backbone model.
            loader: Data loader.
            pred_type: Type of prediction.
            prior: Prior distribution.
            rescale_func_name (str): Name of the rescaling function.
            RESCALE_FUNCS (dict): Dictionary of rescaling functions.
        """
        super().__init__(
            diffusion_process,
            backbone,
            loader,
            prior,
            pred_type,
            rescale_func_name,
            RESCALE_FUNCS,
            **kwargs
        )
        self.sample_dir = sample_dir
        if self.sample_dir:
            os.makedirs(self.sample_dir, exist_ok=True)

    def sample_batch(self, **prior_kwargs):
        """
        Sample a batch of data.

        Args:
            **prior_kwargs: Keyword arguments for sampling.

        Returns:
            Tensor: Sampled batch.
        """
        xt = self.prior.sample(**prior_kwargs)
        # stds = self.prior.fit_prior(**prior_kwargs)
        time_pairs = self.get_adjacent_times(self.DP.times)

        for t, t_next in time_pairs:
            t = torch.Tensor.repeat(t, prior_kwargs['batch_size'])
            t_next = torch.Tensor.repeat(t_next, prior_kwargs['batch_size'])
            xt_next = self.denoise_step(xt, t, t_next, control=prior_kwargs['temperature'])
            xt = xt_next
        return xt

    def save_batch(self, batch, save_prefix, temperature, save_idx):
        """
        Save a batch of samples.

        Args:
            batch: Batch of samples.
            save_prefix (str): Prefix for saving.
            temperature: Temperature for saving.
            save_idx (int): Index for saving.
        """
        save_path = os.path.join(self.sample_dir, f"{temperature}K")
        os.makedirs(save_path, exist_ok=True)
        np.savez_compressed(
            os.path.join(save_path, f"{save_prefix}_idx={save_idx}.npz"), traj=batch
        )

    def sample_loop(self, num_samples, batch_size, temperature, gamma=1, eta=1, save_prefix=None):
        """
        Sampling loop.

        Args:
            num_samples (int): Number of samples to generate.
            batch_size (int): Batch size.
            save_prefix (str): Prefix for saving.
            temperature: Temperature for saving.
            n_ch: Number of channels.
        """
        n_runs = max(num_samples // batch_size, 1)
        if num_samples <= batch_size:
            batch_size = num_samples
        with torch.no_grad():
            for save_idx in range(n_runs):
                batch = self.sample_batch(eta=eta, gamma=gamma, batch_size=batch_size, temperature=temperature, sample_type="from_fit")
                if self.sample_dir and save_prefix:
                    self.save_batch(batch, save_prefix, temperature, save_idx)
                if save_idx == 0:
                    x = batch
                else:
                    x = torch.cat((x, batch), 0)
        return x



class SteeredDiffusionSampler(DiffusionSampler):
    """
    A DiffusionModel consists of instances of a DiffusionProcess, Backbone,
    and Loader objects.
    """

    def __init__(
        self,
        diffusion_process,
        backbone,
        loader,
        prior,
        pred_type,
        sample_dir=None,
        rescale_func_name="no_rescale",
        RESCALE_FUNCS=RESCALE_FUNCS,
        **kwargs,
    ):
        """
        Initialize a SteeredDiffusionSampler.

        Args:
            diffusion_process: The diffusion process.
            backbone: The backbone model.
            loader: Data loader.
            pred_type: Type of prediction.
            prior: Prior distribution.
            rescale_func_name (str): Name of the rescaling function.
            RESCALE_FUNCS (dict): Dictionary of rescaling functions.
            kwargs: Additional keyword arguments.
        """
        super().__init__(
            diffusion_process,
            backbone,
            loader,
            prior,
            pred_type,
            sample_dir,
            rescale_func_name,
            RESCALE_FUNCS,
            **kwargs,
        )

        self.kwargs = kwargs

    def denoise_step(self, b_t, t, t_next, eta, gamma, control_dict, **prior_kwargs):
        """
        Wrapper which calls applies the (marginal) transition kernel
        of the reverse noising process.

        Wrapper to allow the alphas to be sampled and reshaped.
        """
        # b_t_next = self.DP.reverse_step(b_t, t, t_next, self.BB, self.pred_type)
        b_t_next = self.DP.reverse_step(b_t, t, t_next, self.BB, self.pred_type, eta, self.prior, **prior_kwargs)
        for channel, channel_control in control_dict.items():
            logger.debug(f"Setting channel {channel} to {channel_control}")
            b_t_next[:, channel] = (1 - gamma) * b_t_next[:, channel] + gamma * channel_control
        return b_t_next
    
    @staticmethod
    def build_channel_dict(batch_size, prior, temperature):
        """
        Build a dictionary of the conditional values for each channel.

        Args:
            batch_size: The size of the batch.
            prior: The prior object.
            temperature: The temperature, can be a scalar or a vector.

        Returns:
            Dict: Dictionary of the conditional values for each channel.
        """

        fluct_channels = prior.channels_info["fluctuation"]
        num_fluct_channels = len(fluct_channels)
        channel_slice = [batch_size] + [1] + list(prior.shape[1:])  # each channel is treated individually
        channel_dict = {}
        temperatures = torch.full((num_fluct_channels,), temperature)

        for channel, temp in zip(fluct_channels, temperatures):
            channel_dict[channel] = temp
        # logging.debug(f"{channel_dict=}")
        return channel_dict

    def sample_batch(self, eta=1, gamma=0, batch_size=1000, temperature=1, **kwargs):
        """
        Sample a batch of data.

        Args:
            **prior_kwargs: Keyword arguments for sampling.

        Returns:
            Tensor: Sampled batch.
        """
        coord_channels = self.prior.channels_info["coordinate"]
        num_coord_channels = len(coord_channels)
        prior_formatted_temps = torch.Tensor([[temperature]*num_coord_channels]*batch_size)
        logger.debug(f"{prior_formatted_temps}")

        xt = self.prior.sample(batch_size=batch_size, temperatures=prior_formatted_temps)
        # stds = self.prior.fit_prior(**prior_kwargs)

        channel_control_dict = self.build_channel_dict(batch_size, self.prior, temperature)
        time_pairs = self.get_adjacent_times(self.DP.times)

        for t, t_next in time_pairs:
            t = torch.Tensor.repeat(t, batch_size)
            t_next = torch.Tensor.repeat(t_next, batch_size)
            xt_next = self.denoise_step(xt, t, t_next, eta=eta, gamma=gamma, control_dict=channel_control_dict,
                                        batch_size=batch_size, temperatures=prior_formatted_temps)
            xt = xt_next
        return xt
