"""
quantized_linear for QKeras 0.9.0

This quantizer is not included in the qkeras 0.9.0 PyPI release.
patch_qkeras() injects it into the qkeras.quantizers module at runtime
so it can be referenced by string in QKeras layer configs.

Called automatically by script06_utils.py — no manual step required.
"""

import numpy as np
import six
import tensorflow as tf
import tensorflow.keras.backend as K


def patch_qkeras():
    """Inject quantized_linear into qkeras.quantizers if not already present."""
    import qkeras.quantizers as _q
    if hasattr(_q, "quantized_linear"):
        return

    # Internal helpers available in qkeras 0.9.0
    BaseQuantizer = _q.BaseQuantizer
    _round_through = _q._round_through
    _get_scaling_axis = _q._get_scaling_axis
    _get_scale = _q._get_scale  # equivalent of _get_least_squares_scale

    class quantized_linear(BaseQuantizer):
        """Linear quantization with fixed number of bits (backport for QKeras 0.9.0).

        Maps inputs to the nearest evenly-spaced quantized value.
        Use instead of quantized_bits when explicit scale control is needed.

        Args:
            bits (int): Total number of bits. Default: 8.
            integer (int): Integer bits (determines data_type_scale). Default: 0.
            symmetric (bool): Symmetric clip range. Default: True.
            alpha: Scale control. None, "auto", "auto_po2", or a Tensor.
            keep_negative (bool): Include negative values. Default: True.
            use_stochastic_rounding (bool): Default: False.
            scale_axis (int or None): Axis for per-channel scaling. Default: None.
            qnoise_factor (float): Blend between float and quantized [0,1]. Default: 1.0.
            use_variables (bool): Store scale as tf.Variable. Default: False.
            var_name (str or None): Prefix for variable names. Default: None.
        """

        ALPHA_STRING_OPTIONS = ("auto", "auto_po2")

        def __init__(
            self,
            bits=8,
            integer=0,
            symmetric=1,
            keep_negative=True,
            alpha=None,
            use_stochastic_rounding=False,
            scale_axis=None,
            qnoise_factor=1.0,
            var_name=None,
            use_variables=False,
        ):
            super().__init__()
            self.var_name = var_name
            self._check_bits(bits)
            self._check_alpha(alpha)
            self._bits = bits
            self._integer = integer
            self._keep_negative = keep_negative
            self._use_stochastic_rounding = use_stochastic_rounding
            self._scale_axis = scale_axis
            self._use_variables = use_variables
            self.alpha = alpha
            self.qnoise_factor = qnoise_factor
            self.symmetric = symmetric
            self.quantization_scale = self.default_quantization_scale

        def _check_bits(self, bits):
            if bits <= 0:
                raise ValueError(f"Bit count {bits} must be positive")

        def _check_alpha(self, alpha):
            if isinstance(alpha, six.string_types):
                if alpha not in self.ALPHA_STRING_OPTIONS:
                    raise ValueError(
                        f"Invalid alpha '{alpha}'. Must be one of {self.ALPHA_STRING_OPTIONS}")
            elif alpha is not None:
                try:
                    np.array(alpha)
                except TypeError:
                    raise TypeError(
                        f"alpha must be a string, an array, or None, not {type(alpha)}")

        @property
        def bits(self):
            return self._bits

        @property
        def integer(self):
            return self._integer

        @property
        def keep_negative(self):
            return self._keep_negative

        @property
        def use_stochastic_rounding(self):
            return self._use_stochastic_rounding

        @property
        def scale_axis(self):
            return self._scale_axis

        @property
        def use_variables(self):
            return self._use_variables

        @property
        def scale(self):
            return self.quantization_scale / self.data_type_scale

        @property
        def data_type_scale(self):
            integer = tf.cast(self.integer, tf.float32)
            return K.pow(2.0, integer - self.bits + self.keep_negative)

        @property
        def auto_alpha(self):
            return isinstance(self.alpha, six.string_types)

        @property
        def use_sign_function(self):
            return (self.bits == 1.0) and self.keep_negative

        @property
        def default_quantization_scale(self):
            quantization_scale = self.data_type_scale
            if self.alpha is not None and not self.auto_alpha:
                quantization_scale = self.alpha * self.data_type_scale
            return quantization_scale

        def get_clip_bounds(self):
            if self.use_sign_function:
                clip_min = K.cast_to_floatx(-0.5)
                clip_max = K.cast_to_floatx(0.5)
            else:
                unsigned_bits_po2 = K.pow(2.0, self.bits - self.keep_negative)
                clip_min = self.keep_negative * (-unsigned_bits_po2 + self.symmetric)
                clip_max = unsigned_bits_po2 - K.cast_to_floatx(1.0)
            return clip_min, clip_max

        def __call__(self, x):
            self._build()
            x = K.cast_to_floatx(x)
            shape = x.shape
            if self.auto_alpha:
                quantization_scale = self._get_auto_quantization_scale(x)
            else:
                quantization_scale = self.quantization_scale
            scaled_xq = self._scale_clip_and_round(x, quantization_scale)
            xq = scaled_xq * quantization_scale
            res = x + self.qnoise_factor * (xq - x)
            res.set_shape(shape)
            return res

        def _scale_clip_and_round(self, x, quantization_scale):
            shift = self.use_sign_function * 0.5
            clip_min, clip_max = self.get_clip_bounds()
            scaled_x = x / quantization_scale
            clipped_scaled_x = K.clip(scaled_x, clip_min, clip_max)
            scaled_xq = _round_through(
                clipped_scaled_x - shift,
                use_stochastic_rounding=self.use_stochastic_rounding,
                precision=1.0,
            )
            return scaled_xq + shift

        def _get_auto_quantization_scale(self, x):
            quantization_scale = self._get_quantization_scale_from_max_data(x)
            if self.alpha == "auto_po2":
                quantization_scale = self._po2_autoscale(x, quantization_scale)
            self.quantization_scale = tf.stop_gradient(quantization_scale)
            return self.quantization_scale

        def _get_quantization_scale_from_max_data(self, x):
            axis = _get_scaling_axis(self.scale_axis, tf.rank(x))
            clip_min, clip_max = self.get_clip_bounds()
            clip_range = clip_max - clip_min
            if self.keep_negative:
                data_max = K.max(tf.math.abs(x), axis=axis, keepdims=True)
                quantization_scale = (data_max * 2) / clip_range
            else:
                data_max = K.max(x, axis=axis, keepdims=True)
                quantization_scale = data_max / clip_range
            return tf.math.maximum(quantization_scale, K.epsilon())

        def _po2_autoscale(self, x, quantization_scale):
            quantization_scale = K.pow(
                2.0,
                tf.math.round(K.log(quantization_scale + K.epsilon()) / K.log(2.0)))

            def loop_body(_, qs):
                scaled_xq = self._scale_clip_and_round(x, qs)
                new_qs = _get_scale(
                    alpha="auto_po2", x=x, q=scaled_xq, scale_axis=self.scale_axis)
                return qs, new_qs

            def loop_cond(last_qs, qs):
                return tf.math.reduce_any(tf.not_equal(last_qs, qs))

            dummy = -tf.ones_like(quantization_scale)
            max_iter = 1 if self.use_sign_function else 5
            _, quantization_scale = tf.while_loop(
                loop_cond, loop_body, (dummy, quantization_scale),
                maximum_iterations=max_iter)
            return quantization_scale

        def _build(self):
            if not self.built:
                self.build(var_name=self.var_name, use_variables=self.use_variables)

        def max(self):
            _, clip_max = self.get_clip_bounds()
            return clip_max * self.quantization_scale

        def min(self):
            clip_min, _ = self.get_clip_bounds()
            return clip_min * self.quantization_scale

        def range(self):
            if self.use_sign_function:
                return K.cast_to_floatx([self.max(), self.min()])
            clip_min, clip_max = self.get_clip_bounds()
            clip_max = tf.cast(clip_max, tf.int32)
            clip_min = tf.cast(clip_min, tf.int32)
            pos_array = K.cast_to_floatx(tf.range(clip_max + 1))
            neg_array = K.cast_to_floatx(tf.range(clip_min, 0))
            return self.quantization_scale * tf.concat([pos_array, neg_array], axis=0)

        def __str__(self):
            flags = [str(int(self.bits)), str(int(self.integer)), str(int(self.symmetric))]
            if not self.keep_negative:
                flags.append("keep_negative=False")
            if self.auto_alpha:
                flags.append(f"alpha='{self.alpha}'")
            elif self.alpha is not None:
                flags.append(f"alpha={np.array(self.alpha)}")
            if self.use_stochastic_rounding:
                flags.append(f"use_stochastic_rounding={int(self.use_stochastic_rounding)}")
            return "quantized_linear(" + ",".join(flags) + ")"

        def _set_trainable_parameter(self):
            if self.alpha is None:
                self.alpha = "auto_po2"
                self.symmetric = True

        @classmethod
        def from_config(cls, config):
            return cls(**config)

        def get_config(self):
            return {
                "bits": self.bits,
                "integer": self.integer,
                "symmetric": self.symmetric,
                "alpha": self.alpha,
                "keep_negative": self.keep_negative,
                "use_stochastic_rounding": self.use_stochastic_rounding,
                "qnoise_factor": self.qnoise_factor,
            }

    # Inject into qkeras.quantizers globals so string resolution works
    # (qkeras resolves quantizer strings via safe_eval(identifier, globals()))
    _q.quantized_linear = quantized_linear
