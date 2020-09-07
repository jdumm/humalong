import wave

import numpy as np
from pyaudio import PyAudio, paInt16
from pydub import AudioSegment
from pydub.playback import play


def record_audio(outfile_name=None, record_seconds=1, chunk=1024, channels=1, rate=22050):
    """ Opens microphone (OS may ask for permission) and records <record_seconds> if audio.
        If an outfile_name is provided, recording is saved into the file.
        returns: A normalized numpy array of the waveform.
    """

    format_ = paInt16
    p = PyAudio()

    stream = p.open(format=format_,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)
    print("Recording Audio")

    frames_for_file = []
    frames_for_np = []

    mic_fudge = 1  # in chunks, delay start of recording to avoid dead time
    for i in range(0, int(rate / chunk * record_seconds) + mic_fudge + 1):
        if i < mic_fudge:  # discard first chunk(s)
            stream.read(chunk)
            continue
        data = stream.read(chunk)
        frames_for_file.append(data)
        frames_for_np.append(np.frombuffer(data, dtype=np.int16))

    print("* Done Recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    if outfile_name is not None:
        with wave.open(outfile_name, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(format_))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames_for_file))
            wf.close()

    frames_for_np = np.array(frames_for_np).flatten()
    return frames_for_np / np.max(frames_for_np)
