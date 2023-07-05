import tensorflow as tf
from keras import optimizers
from keras.optimizers.optimizer_v2 import optimizer_v2


# https://github.com/tensorflow/addons/issues/2260
@tf.keras.utils.register_keras_serializable(package="GradientAccumulation")
class GradientAccumulator(optimizer_v2.OptimizerV2):
    """Optimizer wrapper for gradient accumulation."""

    def __init__(
        self,
        optimizer: optimizers.Optimizer,
        accum_steps: int = 4,
        name: str = "GradientAccumulator",
        **kwargs,
    ) -> None:
        r"""Construct a new GradientAccumulator optimizer.

        Args:
            optimizer: str or `tf.keras.optimizers.Optimizer` that will be
                used to compute and apply gradients.
            accum_steps: int > 0. Update gradient in every accumulation steps.
            name: Optional name for the operations created when applying
                gradients. Defaults to "GradientAccumulator".
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
                `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by
                norm; `clipvalue` is clip gradients by value, `decay` is
                included for backward compatibility to allow time inverse
                decay of learning rate. `lr` is included for backward
                compatibility, recommended to use `learning_rate` instead.
        """
        super().__init__(name, **kwargs)
        self._optimizer = optimizers.get(optimizer)
        self._gradients = []  # type: ignore
        self._accum_steps = accum_steps

    def _create_slots(self, var_list):
        self._optimizer._create_slots(var_list=var_list)
        for var in var_list:
            self.add_slot(var, "ga")

        self._gradients = [self.get_slot(var, "ga") for var in var_list]

    @property
    def gradients(self):
        """The accumulated gradients on the current replica."""
        if not self._gradients:
            msg = "The accumulator should be called first to initialize the gradients"
            raise ValueError(
                msg,
            )
        return [
            gradient.read_value() if gradient is not None else gradient
            for gradient in self._gradients
        ]

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        self._optimizer._iterations = self.iterations
        return super().apply_gradients(grads_and_vars, name, **kwargs)

    def _transform_loss(self, loss):
        return loss / self._accum_steps

    def _resource_apply_dense(self, grad, var, apply_state=None):
        accum_gradient = self.get_slot(var, "ga")
        if accum_gradient is not None and grad is not None:
            accum_gradient.assign_add(
                grad,
                use_locking=self._use_locking,
                read_value=False,
            )

        def _apply():
            if "apply_state" in self._optimizer._dense_apply_args:
                train_op = self._optimizer._resource_apply_dense(
                    accum_gradient.read_value(),
                    var,
                    apply_state=apply_state,
                )
            else:
                train_op = self._optimizer._resource_apply_dense(
                    accum_gradient.read_value(),
                    var,
                )
            reset_op = accum_gradient.assign(
                tf.zeros_like(accum_gradient),
                use_locking=self._use_locking,
                read_value=False,
            )
            return tf.group(train_op, reset_op)

        return tf.cond(
            (self.iterations + 1) % self._accum_steps == 0,
            _apply,
            lambda: tf.no_op(),
        )

    def _resource_apply_sparse(self, grad, var, indices, apply_state):
        accum_gradient = self.get_slot(var, "ga")
        if accum_gradient is not None and grad is not None:
            self._resource_scatter_add(accum_gradient, indices, grad)

        def _apply():
            if "apply_state" in self._optimizer._sparse_apply_args:
                train_op = self._optimizer._resource_apply_sparse(
                    accum_gradient.sparse_read(indices),
                    var,
                    indices,
                    apply_state=apply_state,
                )
            else:
                train_op = self._optimizer._resource_apply_sparse(
                    accum_gradient.sparse_read(indices),
                    var,
                    indices,
                )
            reset_op = accum_gradient.assign(
                tf.zeros_like(accum_gradient),
                use_locking=self._use_locking,
                read_value=False,
            )
            return tf.group(train_op, reset_op)

        return tf.cond(
            (self.iterations + 1) % self._accum_steps == 0,
            _apply,
            lambda: tf.no_op(),
        )

    def reset(self):
        """Resets the accumulated gradients on the current replica."""
        assign_ops = []
        if not self._gradients:
            return assign_ops

        for gradient in self._gradients:
            if gradient is not None:
                assign_ops.append(
                    gradient.assign(
                        tf.zeros_like(gradient),
                        use_locking=self._use_locking,
                        read_value=False,
                    ),
                )

        return tf.group(assign_ops)

    @property
    def lr(self):
        return self._optimizer._get_hyper("learning_rate")

    @lr.setter
    def lr(self, lr):
        self._optimizer._set_hyper("learning_rate", lr)

    @property
    def learning_rate(self):
        return self._optimizer._get_hyper("learning_rate")

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._optimizer._set_hyper("learning_rate", learning_rate)

    def get_config(self):
        config = {
            "optimizer": optimizers.serialize(self._optimizer),
            "accum_steps": self._accum_steps,
        }
        base_config = super().get_config()
        return {**base_config, **config}
