from flax import linen as nn
import jax.numpy as jnp

googlenet_kernel_init = nn.initializers.kaiming_normal()


class InceptionBlock(nn.Module):
    c_red: dict  # Dictionary of reduced dimensionalities with keys "1x1", "3x3", "5x5", and "max"
    c_out: dict  # Dictionary of output feature sizes with keys "1x1", "3x3", "5x5", and "max"
    act_fn: callable # Activation function

    @nn.compact
    def __call__(self, x, train=True):
        # 1x1 convolution branch
        x_1x1 = nn.Conv(
            self.c_out["1x1"],
            kernel_size=(1, 1),
            kernel_init=googlenet_kernel_init,
            use_bias=False,
        )(x)
        x_1x1 = nn.BatchNorm()(x_1x1, use_running_average=not train)
        x_1x1 = self.act_fn(x_1x1)

        # 3x3 convolution branch
        x_3x3 = nn.Conv(
            self.c_red["3x3"],
            kernel_size=(1, 1),
            kernel_init=googlenet_kernel_init,
            use_bias=False,
        )(x)
        x_3x3 = nn.BatchNorm()(x_3x3, use_running_average=not train)
        x_3x3 = self.act_fn(x_3x3)
        x_3x3 = nn.Conv(
            self.c_out["3x3"],
            kernel_size=(3, 3),
            kernel_init=googlenet_kernel_init,
            use_bias=False,
        )(x_3x3)
        x_3x3 = nn.BatchNorm()(x_3x3, use_running_average=not train)
        x_3x3 = self.act_fn(x_3x3)

        # 5x5 convolution branch
        x_5x5 = nn.Conv(
            self.c_red["5x5"],
            kernel_size=(1, 1),
            kernel_init=googlenet_kernel_init,
            use_bias=False,
        )(x)
        x_5x5 = nn.BatchNorm()(x_5x5, use_running_average=not train)
        x_5x5 = self.act_fn(x_5x5)
        x_5x5 = nn.Conv(
            self.c_out["5x5"],
            kernel_size=(5, 5),
            kernel_init=googlenet_kernel_init,
            use_bias=False,
        )(x_5x5)
        x_5x5 = nn.BatchNorm()(x_5x5, use_running_average=not train)
        x_5x5 = self.act_fn(x_5x5)

        # Max-pool branch
        x_max = nn.max_pool(x, (3, 3), strides=(2, 2))
        x_max = nn.Conv(
            self.c_out["max"],
            kernel_size=(1, 1),
            kernel_init=googlenet_kernel_init,
            use_bias=False,
        )(x)
        x_max = nn.BatchNorm()(x_max, use_running_average=not train)
        x_max = self.act_fn(x_max)

        x_out = jnp.concatenate([x_1x1, x_3x3, x_5x5, x_max], axis=-1)
        return x_out
