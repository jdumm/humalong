import sys
import time
from random import choice

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from pydub.playback import play

from humutils import record_audio, create_pure_tone_numpy, note2freq


def main():
    """ Execute pitch match game.
    """
    # Start by generating a pure tone with a sine wave
    duration = 1.0  # seconds per segment
    sr = 22050  # sample rate
    num_segs = 1  # segments
    y_segs = []
    for _ in range(num_segs):
        freq = choice(list(note2freq.values()))
        y_segs.append(create_pure_tone_numpy(freq, duration, sr))
    y = np.array(y_segs).flatten()
    y_audio_seg = AudioSegment(y.tobytes(), frame_rate=sr, sample_width=y.dtype.itemsize, channels=1)

    # Play the pure tone
    print("Now playing the pitch to match...")
    play(y_audio_seg)

    # Now wait for user input to start recording their humming of the tone, or load test audio
    test_mode = False  # Set to True to use a default recording - for testing
    if not test_mode:
        input("Press enter to record your purest tone!")
        time.sleep(0.01)
        outfile_name = 'temp_user_recording.wav'
        x = record_audio(outfile_name=outfile_name, record_seconds=duration*num_segs, chunk=1024, channels=1, rate=sr)
    else:
        audio_path = "default_user_recording.wav"
        x, _ = librosa.load(audio_path, sr=sr)
        x = x / x.max()
        x = np.tile(x, num_segs)
    x = x[0:len(y)]  # trim to be the same length as the pure tone
    y = y[0:len(x)]  # and just in case the other direction is ever needed

    # Calculate constant-q transform for y (target) and x (user recording)
    cqt_y = np.abs(librosa.cqt(y, sr=sr))
    cqt_x = np.abs(librosa.cqt(x, sr=sr))

    # Calculate user and baseline scores
    score = np.sum(np.multiply(cqt_y, cqt_x))
    baseline_score = np.sum(np.multiply(cqt_y, cqt_y))

    print(f"Your Pure Tone Score is {score / baseline_score * 100:0.1f}!")

    menu(x, y, sr)


def menu(x, y, sr):
    """ Wait for user input to decide what to do next.
    """
    response = input(
        "r) retry\nq) quit\ny) listen again to the pure tone\nx) listen to your recording\ns) listen together"
        "\ng) see graphics\n")
    if response == 'r':
        print('-- Reloading')
        main()
    elif response == 'q':
        print('-- Quiting...')
        sys.exit()
    elif response == 'y':
        print('-- Playing the pure tone')
        y_audio_seg = AudioSegment(y.astype('float32').tobytes(), frame_rate=sr,
                                   sample_width=y.astype('float32').dtype.itemsize, channels=1)
        play(y_audio_seg)
    elif response == 'x':
        print('-- Playing the user tone')
        x_audio_seg = AudioSegment(x.astype('float32').tobytes(), frame_rate=sr,
                                   sample_width=x.astype('float32').dtype.itemsize, channels=1)
        play(x_audio_seg)
    elif response == 's':
        print("-- Playing sum of waveforms")
        s = y + x
        s = s.astype('float32')  # downscale from float64 to enable playback with pydub
        s_audio_seg = AudioSegment(s.tobytes(), frame_rate=sr, sample_width=y.dtype.itemsize, channels=1)
        play(s_audio_seg)
    elif response == 'g':
        print('-- Plotting waveform and constant-q transform')
        # Display the waveforms
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 9))
        plt.tight_layout(pad=2.0)
        ax0.set_title("Full Audio")
        librosa.display.waveplot(y, sr=sr, ax=ax0, color='k', alpha=0.3, label='Pure')
        librosa.display.waveplot(x, sr=sr, ax=ax0, color='b', alpha=0.5, label='You')
        ax0.legend(fontsize=16)
        ax0.set_xlabel('')

        ax1.set_title("Sum of Waveforms")
        librosa.display.waveplot(y+x, sr=sr, ax=ax1, color='k', alpha=0.3, label='Sum')
        ax1.legend(fontsize=16)
        ax1.set_xlabel('')

        frac = 0.05
        offset = int(min(x.argmax(), len(x) - frac * len(x)))
        ax2.set_title(f"{frac * 100:0.0f}% Sample Near Max Volume")
        librosa.display.waveplot(y[offset:offset + int(frac * len(y))], sr=sr, ax=ax2, color='k', alpha=0.3,
                                 label='Pure')
        librosa.display.waveplot(x[offset:offset + int(frac * len(x))], sr=sr, ax=ax2, color='b', alpha=0.5,
                                 label='You')
        plt.draw()

        # Calculate constant-q transform for y (target) and x (user recording) and display them
        cqt_y = np.abs(librosa.cqt(y, sr=sr))
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 9))
        plt.tight_layout(pad=2.0)
        db_vs_time_y = librosa.amplitude_to_db(cqt_y, ref=np.min)
        img_y = librosa.display.specshow(db_vs_time_y,
                                         sr=sr, x_axis='time', y_axis='cqt_note', ax=ax0)
        ax0.set_title('Constant-Q power spectrum - Pure Tone')
        ax0.set_xlabel('')
        fig.colorbar(img_y, ax=ax0, format="%+2.0f dB")

        cqt_x = np.abs(librosa.cqt(x, sr=sr))
        db_vs_time_x = librosa.amplitude_to_db(cqt_x, ref=np.min)
        img_x = librosa.display.specshow(db_vs_time_x, sr=sr, x_axis='time', y_axis='cqt_note', ax=ax1)
        ax1.set_title('Constant-Q power spectrum - Recorded Tone')
        ax1.set_xlabel('')
        fig.colorbar(img_x, ax=ax1, format="%+2.2f dB")

        img_x = librosa.display.specshow(cqt_y * cqt_x, sr=sr, x_axis='time', y_axis='cqt_note', ax=ax2)
        ax2.set_title('Product (Overlap) of Spectra')
        fig.colorbar(img_x, ax=ax2, format="%+2.0f dB")
        plt.draw()

        print(' * Close the graphs to proceed')
        plt.show()
    else:
        print('-- Input not understood')
    menu(x, y, sr)


if __name__ == '__main__':
    main()
