### Overview

A quick game to see how good you can match and maintain musical pitch while humming!

Listen to the pitch and hum it back to get your match score.

### Installation

`pip install -r requirements.txt`

You may need to run `brew install portaudio` in advance to get `pyaudio` to work.

### Running

`python humalong.py`

Listen to the pitch.  Once you are ready to record your version, press enter.  
You will be given a score based on how well your humming matches the pure tone.

Several options are then presented to you, 
including options to see graphs of your waveforms and the constant-q transform used for judging.

Or retry for a better score!

### Tips

The overall volume of your recording will not impact the score, as long as there is not a lot of 
background noise or if you have a bad microphone.  However, try to keep your volume constant during the
recording.

You can listen to the beat frequency in the options at the end to hear how close you are.