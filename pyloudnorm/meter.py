import warnings
import numpy as np
from . import util
from .iirfilter import IIRfilter


class Meter:
    """Meter object which defines how the meter operates

    Defaults to the algorithm defined in ITU-R BS.1770-4.

    Parameters
    ----------
    rate : float
        Sampling rate in Hz.
    filter_class : str
        Class of weighting filter used.
        - 'K-weighting'
        - 'Fenton/Lee 1'
        - 'Fenton/Lee 2'
        - 'Dash et al.'
        - 'DeMan'
    block_size : float
        Gating block size in seconds.
    """

    def __init__(
        self,
        rate,
        filter_class="K-weighting",
        lufs_i_window_time=0.400,
        lufs_s_window_time=3.0,
        lufs_m_window_time=0.4,
        en_lufs_i=True,
        en_lufs_s=True,
    ):
        self.rate = rate
        self.filter_class = filter_class
        self.lufs_i_window_time = lufs_i_window_time
        self.ch_gains = [1.0, 1.0, 1.0, 1.41, 1.41]  # 5 channel gains
        self.lufs_i_overlap = 0.75  # overlap of 75% of the block duration
        self.lufs_step = 1.0 - self.lufs_i_overlap  # step size by percentage
        # integrated loudness
        self.lufs_i_gamma_abs = -70.0  # -70 LKFS = absolute loudness threshold
        self.lufs_i_buffer = None
        self.lufs_i_blocks = None
        self.lufs_i_z = None
        # short loudness
        self.lufs_s_num_samples = int(lufs_s_window_time * self.rate)
        self.lufs_s_buffer = None
        # momentary loudness
        self.lufs_m_num_samples = int(lufs_m_window_time * self.rate)
        self.lufs_m_buffer = None
        # enable flags
        self.en_lufs_i = en_lufs_i
        self.en_lufs_s = en_lufs_s

    def reset(self):
        """
        Reset the meter to its initial state.
        """
        self.lufs_i_buffer = None
        self.lufs_i_blocks = None
        self.lufs_i_z = None
        self.lufs_s_buffer = None
        self.lufs_m_buffer = None
        # reset filter states
        for filter_ch in self._filters:
            for _, filter_stage in filter_ch.items():
                filter_stage.reset()

    def linear2lufs(self, val_lin):
        """
        Convert linear value to LUFS value.
        """
        return -0.691 + 10.0 * np.log10(val_lin)

    def calc_z_one_block(self, data):
        """
        Calculate the mean square of the data signal as a single block, on each channel.
        """
        num_channels = data.shape[1]
        # Instantiate array. Shape to be consistent with calc_z
        z = np.zeros(shape=(num_channels, 1))
        for i in range(num_channels):
            # calculate mean square of the filtered for each block (see eq. 1)
            z[i, :] = np.mean((np.square(data[:, i])))
        return z

    def calc_z(self, data):
        """
        Calculate the mean square of the data signal for each sliding block, on each channel.
        """
        num_channels = data.shape[1]
        t_data = data.shape[0] / self.rate  # length of the input in seconds
        # Total number of gated blocks (see end of eq. 3)
        num_blocks = int(
            np.round(
                (
                    (t_data - self.lufs_i_window_time)
                    / (self.lufs_i_window_time * self.lufs_step)
                )
            )
            + 1
        )
        j_range = np.arange(0, num_blocks)  # indexed list of total blocks
        z = np.zeros(shape=(num_channels, num_blocks))  # instantiate array

        for i in range(num_channels):  # iterate over input channels
            for j in j_range:  # iterate over total frames
                # lower and upper bounds of integration (in samples)
                lower_ind = int(
                    self.lufs_i_window_time * (j * self.lufs_step) * self.rate
                )
                upper_ind = int(
                    self.lufs_i_window_time * (j * self.lufs_step + 1) * self.rate
                )
                # calculate mean square of the filtered for each block (see eq. 1)
                z[i, j] = np.mean(np.square(data[lower_ind:upper_ind, i]))
        return z

    def calc_lufs_blocks(self, z):
        """
        Calculate the loudness for each block (see eq. 4).
        """
        lufs_blocks = None
        with warnings.catch_warnings():
            (num_ch, num_blocks) = z.shape
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # loudness for each jth block (see eq. 4)
            lufs_blocks = np.array(
                [
                    self.linear2lufs(
                        np.sum([self.ch_gains[i] * z[i, j] for i in range(num_ch)])
                    )
                    for j in range(num_blocks)
                ]
            )
        return lufs_blocks

    def gate_loudness(self, z, lufs_blocks):
        """
        Gate the loudness of a signal (only valid for the integrated loudness)
        :param z: ndarray of shape (ch, blocks) with the mean square of the filtered
                  signal for each block.
        :param lufs_blocks: ndarray of shape (blocks,) with the loudness of each block.
        :return: LUFS: float, integrated gated loudness of the input measured in dB LUFS.
        """
        lufs_i = self.lufs_i_gamma_abs
        num_channels = z.shape[0]
        # find gating block indices above absolute threshold
        j_gated_abs = np.argwhere(lufs_blocks >= self.lufs_i_gamma_abs).flatten()
        if len(j_gated_abs) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # calculate the average of z[i,j] as show in eq. 5
                z_avg_gated_abs = np.mean(z[:, j_gated_abs], axis=1)

            # calculate the relative threshold value (see eq. 6)
            gamma_rel = (
                self.linear2lufs(
                    np.sum(
                        [
                            self.ch_gains[i] * z_avg_gated_abs[i]
                            for i in range(num_channels)
                        ]
                    )
                )
                - 10.0
            )
            # find gating block indices above relative and absolute thresholds  (end of eq. 7)
            j_gated = np.argwhere(
                (lufs_blocks > gamma_rel) & (lufs_blocks > self.lufs_i_gamma_abs)
            ).flatten()
            if len(j_gated) > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    # calculate the average of z[i,j] as show in eq. 7
                    # with blocks above both thresholds
                    z_avg_gated = np.mean(z[:, j_gated], axis=1)
                # calculate final loudness gated loudness (see eq. 7)
                with np.errstate(divide="ignore"):
                    lufs_i = self.linear2lufs(
                        np.sum(
                            [
                                self.ch_gains[i] * z_avg_gated[i]
                                for i in range(num_channels)
                            ]
                        )
                    )
        return lufs_i

    def filter_input(self, data):
        """
        Apply frequency weighting filters to the input signal.
        :param data: ndarray of shape (samples, ch) or (samples,) for mono audio.
        :return: ndarray of shape (samples, ch) or (samples,) for mono audio.
        """
        input_data = data.copy()
        if input_data.ndim == 1:
            input_data = np.reshape(input_data, (input_data.shape[0], 1))

        num_ch = input_data.shape[1]
        # Apply frequency weighting filters to the data signal
        # Accounts for the acoustic response of the head and auditory system
        for ch in range(num_ch):
            for _, filter_stage in self._filters[ch].items():
                input_data[:, ch] = filter_stage.apply_filter(input_data[:, ch])

        return input_data

    def calc_moving_average_lufs(self, buffer, data, num_samples):
        """
        Calculate the moving average of "num_samples" of the given buffer, 
        after appending the new "data" to it.
        :param buffer: ndarray of shape (samples, ch). Buffer to calculate the moving average from.
        :param data: ndarray of shape (samples, ch). Incoming data, to be appended to the buffer.
        :param num_samples: int, number of samples to calculate the moving average.
        :return: (LUFS: float, buffer: ndarray). Respectively the averaged loudness of the buffer, 
                 measured in dB LUFS and the buffer with the new data appended
        """
        # TODO: this can be optimized
        buffer = data if buffer is None else np.append(buffer, data, axis=0)
        if len(buffer) >= num_samples:
            buffer = buffer[-num_samples:, :]
            lufs_buffer = buffer
        else:
            # Add trailing zeros to the input data to make it a full 3s block
            trailing_zeros = np.zeros(
                shape=(
                    int(num_samples - buffer.shape[0]),
                    buffer.shape[1],
                )
            )
            lufs_buffer = np.append(trailing_zeros, buffer, axis=0)
        # calculate the mean square of the filtered signal as a single block
        z = self.calc_z_one_block(lufs_buffer)
        lufs_blocks_s = self.calc_lufs_blocks(z)
        lufs = np.mean(lufs_blocks_s)

        return (lufs, buffer)

    def step_loudness(self, data):
        """
        Measure the integrated gated loudness of a signal, block by block, in RT.
        :param data: ndarray of shape (samples, ch) or (samples,) for mono audio.
        :return: LUFS: float, integrated gated loudness of the input measured in dB LUFS.
        """
        input_data = self.filter_input(data)

        # Initialize integrated loudness to its minimum value
        lufs_i = self.lufs_i_gamma_abs
        # calculation for lufs_i
        if self.en_lufs_i:
            self.lufs_i_buffer = (
                input_data
                if self.lufs_i_buffer is None
                else np.append(self.lufs_i_buffer, input_data, axis=0)
            )
            # length of the lufs_i buffer in seconds
            t_lufs_i_buffer = self.lufs_i_buffer.shape[0] / self.rate

            if t_lufs_i_buffer >= self.lufs_i_window_time:
                # calculate the mean square of the filtered signal for each block
                z = self.calc_z(self.lufs_i_buffer)
                (_, num_blocks) = z.shape
                # calculate the loudness for each block (see eq. 4)
                lufs_blocks = self.calc_lufs_blocks(z)
                # advance buffer
                self.lufs_i_buffer = self.lufs_i_buffer[
                    int(
                        num_blocks
                        * self.lufs_i_window_time
                        * self.lufs_step
                        * self.rate
                    ) :,
                    :,
                ]
                self.lufs_i_z = (
                    z if self.lufs_i_z is None else np.append(self.lufs_i_z, z, axis=1)
                )
                self.lufs_i_blocks = (
                    lufs_blocks
                    if self.lufs_i_blocks is None
                    else np.append(self.lufs_i_blocks, lufs_blocks)
                )
                # TODO: I can't think of any way to do this without it increasing in
                # memory and CPU for every block appended
                lufs_i = self.gate_loudness(self.lufs_i_z, self.lufs_i_blocks)
            else:
                # Add trailing zeros to the input data to make it a full 0.4s block
                trailing_zeros = np.zeros(
                    shape=(
                        int(self.lufs_i_window_time * self.rate)
                        - self.lufs_i_buffer.shape[0],
                        self.lufs_i_buffer.shape[1],
                    )
                )
                lufs_buffer = np.append(trailing_zeros, self.lufs_i_buffer, axis=0)
                # calculate the mean square of the filtered signal for each block
                z = self.calc_z(lufs_buffer)
                (_, num_blocks) = z.shape
                # calculate the loudness for each block (see eq. 4)
                lufs_blocks = self.calc_lufs_blocks(z)

        # Momentary loudness
        (lufs_m, self.lufs_m_buffer) = self.calc_moving_average_lufs(
            self.lufs_m_buffer, input_data, self.lufs_m_num_samples
        )

        # Short-term loudness
        if self.en_lufs_s:
            # We use the last 3 seconds of audio to calculate the short-term loudness
            (lufs_s, self.lufs_s_buffer) = self.calc_moving_average_lufs(
                self.lufs_s_buffer, input_data, self.lufs_s_num_samples
            )
        else:
            lufs_s = -np.inf

        return (lufs_i, lufs_m, lufs_s)

    def integrated_loudness(self, data):
        """Measure the integrated gated loudness of a signal.

        Uses the weighting filters and block size defined by the meter
        the integrated loudness is measured based upon the gating algorithm
        defined in the ITU-R BS.1770-4 specification.

        Input data must have shape (samples, ch) or (samples,) for mono audio.
        Supports up to 5 channels and follows the channel ordering:
        [Left, Right, Center, Left surround, Right surround]

        Params
        -------
        data : ndarray
            Input multichannel audio data.

        Returns
        -------
        LUFS : float
            Integrated gated loudness of the input measured in dB LUFS.
        """
        util.valid_audio(data, self.rate, self.lufs_i_window_time)

        # integrated_loudness() in performed in the entire data array all at once,
        # hence we should reset all internal states before processing, in case we already
        # have called it before
        self.reset()
        input_data = self.filter_input(data)
        # calculate the mean square of the filtered signal for each block
        z = self.calc_z(input_data)
        # calculate the loudness for each block (see eq. 4)
        lufs_blocks = self.calc_lufs_blocks(z)
        return self.gate_loudness(z, lufs_blocks)

    @property
    def filter_class(self):
        return self._filter_class

    @filter_class.setter
    def filter_class(self, value):
        max_num_channels = 5
        self._filters = [
            {} for _ in range(max_num_channels)
        ]  # reset (clear) filters for each channel
        self._filter_class = value
        if self._filter_class == "K-weighting":
            for i in range(max_num_channels):
                self._filters[i]["high_shelf"] = IIRfilter(
                    4.0, 1 / np.sqrt(2), 1500.0, self.rate, "high_shelf"
                )
                self._filters[i]["high_pass"] = IIRfilter(
                    0.0, 0.5, 38.0, self.rate, "high_pass"
                )
        elif self._filter_class == "Fenton/Lee 1":
            for i in range(max_num_channels):
                self._filters[i]["high_shelf"] = IIRfilter(
                    5.0, 1 / np.sqrt(2), 1500.0, self.rate, "high_shelf"
                )
                self._filters[i]["high_pass"] = IIRfilter(
                    0.0, 0.5, 130.0, self.rate, "high_pass"
                )
                self._filters[i]["peaking"] = IIRfilter(
                    0.0, 1 / np.sqrt(2), 500.0, self.rate, "peaking"
                )
        elif self._filter_class == "Fenton/Lee 2":  # not yet implemented
            for i in range(max_num_channels):
                self._filters[i]["high_shelf"] = IIRfilter(
                    4.0, 1 / np.sqrt(2), 1500.0, self.rate, "high_shelf"
                )
                self._filters[i]["high_pass"] = IIRfilter(
                    0.0, 0.5, 38.0, self.rate, "high_pass"
                )
                self._filters[i]["peaking"] = IIRfilter(
                    0.0, 1 / np.sqrt(2), 500.0, self.rate, "peaking"
                )
        elif self._filter_class == "Dash et al.":
            for i in range(max_num_channels):
                self._filters[i]["high_shelf"] = IIRfilter(
                    4.0, 1 / np.sqrt(2), 1500.0, self.rate, "high_shelf"
                )
                self._filters[i]["high_pass"] = IIRfilter(
                    0.0, 0.5, 38.0, self.rate, "high_pass"
                )
                self._filters[i]["peaking"] = IIRfilter(
                    -2.0, 1 / np.sqrt(2), 500.0, self.rate, "peaking"
                )
        elif self._filter_class == "DeMan":
            for i in range(max_num_channels):
                self._filters[i]["high_shelf"] = IIRfilter(
                    3.99984385397,
                    0.7071752369554193,
                    1681.9744509555319,
                    self.rate,
                    "high_shelf",
                )
                self._filters[i]["high_pass"] = IIRfilter(
                    0.0, 0.5003270373253953, 38.13547087613982, self.rate, "high_pass"
                )
        elif self._filter_class == "custom":
            pass
        else:
            raise ValueError("Invalid filter class:", self._filter_class)
