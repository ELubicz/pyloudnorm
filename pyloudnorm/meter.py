import warnings
import numpy as np
from . import util
from .iirfilter import IIRfilter

class Meter(object):
    """ Meter object which defines how the meter operates

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

    def __init__(self, rate, filter_class="K-weighting", block_size=0.400):
        self.rate = rate
        self.filter_class = filter_class
        self.block_size = block_size
        self.ch_gains = [1.0, 1.0, 1.0, 1.41, 1.41] # 5 channel gains
        self.gamma_abs = -70.0 # -70 LKFS = absolute loudness threshold
        self.overlap = 0.75 # overlap of 75% of the block duration
        self.step = 1.0 - self.overlap # step size by percentage
        self.reset()

    def reset(self):
        """ 
        Reset the meter to its initial state.
        """
        self.buffer_data = None
        self.z_sum_gated_abs = None
        self.z_sum_gated = None
        self.lufs_integrated = self.gamma_abs
        self.lufs_num_gated_blocks_abs = 0
        self.lufs_num_gated_blocks = 0
        for filter_ch in self._filters:
            for (_, filter_stage) in filter_ch.items():
                filter_stage.reset()

    def linear2lufs(self, val_lin):
        """
        Convert linear value to LUFS value.
        """
        return -0.691 + 10.0 * np.log10(val_lin)

    def step_loudness(self, data):
        """
        Measure the integrated gated loudness of a signal, block by block, in RT.
        :param data: ndarray of shape (samples, ch) or (samples,) for mono audio.
        :return: LUFS: float, integrated gated loudness of the input measured in dB LUFS.
        """
        input_data = data.copy()

        if input_data.ndim == 1:
            input_data = np.reshape(input_data, (input_data.shape[0], 1))

        num_channels = input_data.shape[1]

        # Apply frequency weighting filters - account for the acoustic response of the head and auditory system
        for ch in range(num_channels):
            for (_, filter_stage) in self._filters[ch].items():
                input_data[:,ch] = filter_stage.apply_filter(input_data[:,ch])

        if self.buffer_data is None:
            self.buffer_data = input_data
        else:
            self.buffer_data = np.append(self.buffer_data, input_data, axis=0)

        t_block = self.block_size # 400 ms gating block standard
        t_data = self.buffer_data.shape[0] / self.rate # length of the input in seconds

        if t_data >= t_block:
            ch_gains = self.ch_gains
            gamma_abs = self.gamma_abs
            step = self.step

            num_blocks = int(np.floor(((t_data - t_block) / (t_block * step)))+1) # total number of gated blocks (see end of eq. 3)
            j_range = np.arange(0, num_blocks) # indexed list of total blocks
            z = np.zeros(shape=(num_channels,num_blocks)) # instantiate array - tresponse of input

            for i in range(num_channels): # iterate over input channels
                for j in j_range: # iterate over total frames
                    lower_ind = int(t_block * (j * step    ) * self.rate) # lower bound of integration (in samples)
                    upper_ind = int(t_block * (j * step + 1) * self.rate) # upper bound of integration (in samples)
                    # calculate mean square of the filtered for each block (see eq. 1)
                    z[i,j] = (1.0 / (t_block * self.rate)) * np.sum(np.square(self.buffer_data[lower_ind:upper_ind,i]))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # loudness for each jth block (see eq. 4)
                lufs_blocks = np.array([self.linear2lufs(np.sum([ch_gains[i] * z[i,j] for i in range(num_channels)])) for j in j_range])

            # find gating block indices above absolute threshold
            j_gated_abs = np.argwhere(lufs_blocks >= gamma_abs).flatten()
            if len(j_gated_abs) > 0:
                self.lufs_num_gated_blocks_abs += len(j_gated_abs)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    # calculate the average of z[i,j] as show in eq. 5
                    if self.z_sum_gated_abs is None:
                        self.z_sum_gated_abs = np.sum(z[:,j_gated_abs], axis=1)
                    else:
                        self.z_sum_gated_abs += np.sum(z[:,j_gated_abs], axis=1)
                    z_avg_gated_abs = self.z_sum_gated_abs / self.lufs_num_gated_blocks_abs
                # calculate the relative threshold value (see eq. 6)
                gamma_rel = self.linear2lufs(np.sum([ch_gains[i] * z_avg_gated_abs[i] for i in range(num_channels)])) - 10.0

                # find gating block indices above relative and absolute thresholds  (end of eq. 7)
                j_gated = np.argwhere((lufs_blocks > gamma_rel) & (lufs_blocks > gamma_abs)).flatten()

                if len(j_gated) > 0:
                    self.lufs_num_gated_blocks += len(j_gated)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        # calculate the average of z[i,j] as show in eq. 7 with blocks above both thresholds
                        if self.z_sum_gated is None:
                            self.z_sum_gated = np.sum(z[:,j_gated], axis=1)
                        else:
                            self.z_sum_gated += np.sum(z[:,j_gated], axis=1)
                        z_avg_gated = self.z_sum_gated / self.lufs_num_gated_blocks

                    # calculate final loudness gated loudness (see eq. 7)
                    with np.errstate(divide='ignore'):
                        self.lufs_integrated = self.linear2lufs(np.sum([ch_gains[i] * z_avg_gated[i] for i in range(num_channels)]))

            # advance buffer
            self.buffer_data = self.buffer_data[int(num_blocks * t_block * step * self.rate):,:]
        return self.lufs_integrated

    def integrated_loudness(self, data):
        """ Measure the integrated gated loudness of a signal.

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
        # integrated_loudness() in performed in the entire data array all at once,
        # hence we should reset all internal states before processing, in case we already
        # have called it before
        self.reset()
        input_data = data.copy()
        util.valid_audio(input_data, self.rate, self.block_size)

        if input_data.ndim == 1:
            input_data = np.reshape(input_data, (input_data.shape[0], 1))

        num_channels = input_data.shape[1]
        num_samples  = input_data.shape[0]

        # Apply frequency weighting filters - account for the acoustic response of the head and auditory system
        for ch in range(num_channels):
            for (_, filter_stage) in self._filters[ch].items():
                input_data[:,ch] = filter_stage.apply_filter(input_data[:,ch])

        ch_gains = self.ch_gains
        t_block = self.block_size # 400 ms gating block standard
        gamma_abs = self.gamma_abs
        step = self.step

        t_data = num_samples / self.rate # length of the input in seconds
        num_blocks = int(np.round(((t_data - t_block) / (t_block * step)))+1) # total number of gated blocks (see end of eq. 3)
        j_range = np.arange(0, num_blocks) # indexed list of total blocks
        z = np.zeros(shape=(num_channels,num_blocks)) # instantiate array - tresponse of input

        for i in range(num_channels): # iterate over input channels
            for j in j_range: # iterate over total frames
                lower_ind = int(t_block * (j * step    ) * self.rate) # lower bound of integration (in samples)
                upper_ind = int(t_block * (j * step + 1) * self.rate) # upper bound of integration (in samples)
                # calculate mean square of the filtered for each block (see eq. 1)
                z[i,j] = (1.0 / (t_block * self.rate)) * np.sum(np.square(input_data[lower_ind:upper_ind,i]))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # loudness for each jth block (see eq. 4)
            lufs_blocks = np.array([self.linear2lufs(np.sum([ch_gains[i] * z[i,j] for i in range(num_channels)])) for j in j_range])

        # find gating block indices above absolute threshold
        j_gated_abs = np.argwhere(lufs_blocks >= gamma_abs).flatten()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # calculate the average of z[i,j] as show in eq. 5
            z_avg_gated_abs = np.mean(z[:,j_gated_abs], axis=1)
        # calculate the relative threshold value (see eq. 6)
        gamma_rel = self.linear2lufs(np.sum([ch_gains[i] * z_avg_gated_abs[i] for i in range(num_channels)])) - 10.0

        # find gating block indices above relative and absolute thresholds  (end of eq. 7)
        j_gated = np.argwhere((lufs_blocks > gamma_rel) & (lufs_blocks > gamma_abs)).flatten()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # calculate the average of z[i,j] as show in eq. 7 with blocks above both thresholds
            z_avg_gated = np.nan_to_num(np.mean(z[:,j_gated], axis=1))

        # calculate final loudness gated loudness (see eq. 7)
        with np.errstate(divide='ignore'):
            LUFS = self.linear2lufs(np.sum([ch_gains[i] * z_avg_gated[i] for i in range(num_channels)]))

        return LUFS

    @property
    def filter_class(self):
        return self._filter_class

    @filter_class.setter
    def filter_class(self, value):
        max_num_channels = 5
        self._filters = [{} for _ in range(max_num_channels)] # reset (clear) filters for each channel
        self._filter_class = value
        if self._filter_class == "K-weighting":
            for i in range(max_num_channels):
                self._filters[i]['high_shelf'] = IIRfilter(4.0, 1/np.sqrt(2), 1500.0, self.rate, 'high_shelf')
                self._filters[i]['high_pass'] = IIRfilter(0.0, 0.5, 38.0, self.rate, 'high_pass')
        elif self._filter_class == "Fenton/Lee 1":
            for i in range(max_num_channels):
                self._filters[i]['high_shelf'] = IIRfilter(5.0, 1/np.sqrt(2), 1500.0, self.rate, 'high_shelf')
                self._filters[i]['high_pass'] = IIRfilter(0.0, 0.5, 130.0, self.rate, 'high_pass')
                self._filters[i]['peaking'] = IIRfilter(0.0, 1/np.sqrt(2), 500.0, self.rate, 'peaking')
        elif self._filter_class == "Fenton/Lee 2": # not yet implemented
            for i in range(max_num_channels):
                self._filters[i]['high_shelf'] = IIRfilter(4.0, 1/np.sqrt(2), 1500.0, self.rate, 'high_shelf')
                self._filters[i]['high_pass'] = IIRfilter(0.0, 0.5, 38.0, self.rate, 'high_pass')
                self._filters[i]['peaking'] = IIRfilter(0.0, 1/np.sqrt(2), 500.0, self.rate, 'peaking')
        elif self._filter_class == "Dash et al.":
            for i in range(max_num_channels):
                self._filters[i]['high_shelf'] = IIRfilter(4.0, 1/np.sqrt(2), 1500.0, self.rate, 'high_shelf')
                self._filters[i]['high_pass'] = IIRfilter(0.0, 0.5, 38.0, self.rate, 'high_pass')
                self._filters[i]['peaking'] = IIRfilter(-2.0, 1/np.sqrt(2), 500.0, self.rate, 'peaking')
        elif self._filter_class == "DeMan":
            for i in range(max_num_channels):
                self._filters[i]['high_shelf'] = IIRfilter(3.99984385397, 0.7071752369554193, 1681.9744509555319, self.rate, 'high_shelf')
                self._filters[i]['high_pass'] = IIRfilter(0.0, 0.5003270373253953, 38.13547087613982, self.rate, 'high_pass')
        elif self._filter_class == "custom":
            pass
        else:
            raise ValueError("Invalid filter class:", self._filter_class)

if __name__ == "__main__":
    import argparse
    import soundfile as sf

    from pathlib import Path

    parser = argparse.ArgumentParser(description="Measure the integrated loudness of a signal.")
    parser.add_argument("-i","--input_file", help="Path to input file.", required=True)
    parser.add_argument("-f", "--filter_class", help="Frequency weighting filter class.", default="K-weighting")
    parser.add_argument("-b", "--block_size", help="Gating block size in seconds.", default=0.400, type=float)
    args = parser.parse_args()

    if Path(args.input_file).is_absolute():
        input_file = Path(args.input_file)
    else:
        curr_dir = Path(__file__).parent
        input_file = curr_dir / args.input_file

    data, rate = sf.read(input_file)
    meter = Meter(rate, args.filter_class, args.block_size)
    LUFS = meter.integrated_loudness(data)
    print("Integrated loudness:", LUFS, "LUFS")
