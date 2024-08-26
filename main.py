import os
import pyaudio
import wave
import numpy as np
import datetime
import time
import argparse
import joblib
import subprocess
from fit_model import extract_features, CustomFeatureExtractor
from config import PHONE_NUMBER
import logging
import mylib

logger = logging.getLogger(__name__)


# Parameters
CHUNK_SIZE = 512
FORMAT = pyaudio.paInt16
RATE = 44100
THRESHOLD = 1000
LENGTH_THRESHOLD = 20  # number of above-threshold frames you must here within RECORD_DELAY window to classify as train
RECORD_DELAY = 250  # number of sub-threshold frames to wait before saving clip

BIN_PATH = "/Users/joeschlessinger/signal-cli-2/bin/signal-cli"
logger.info(
    f"Initialized with length threshold {LENGTH_THRESHOLD} and record delay {RECORD_DELAY}"
)


def classify_audio(file_path, model):
    features = extract_features(
        file_path,
        n_bins=model.named_steps["feature_extractor"].n_bins,
        min_freq=model.named_steps["feature_extractor"].min_freq,
        max_freq=model.named_steps["feature_extractor"].max_freq,
    )
    features = model.named_steps["scaler"].transform([features])
    prediction = model.named_steps["classifier"].predict(features)
    return prediction[0]


def detect_train_horn(audio_data):
    # Basic threshold-based detection
    max_amplitude = np.max(np.abs(audio_data))
    return max_amplitude > THRESHOLD


def format_filename(filename):
    # 'train_horn_20240326_115843.wav' turn this into DTG
    dtg = filename[filename.find("train_horn_") + 11 : -4]
    dt = datetime.datetime.strptime(dtg, "%Y%m%d_%H%M%S")
    return dt.strftime("%a %d %b %Y, %I:%M%p")


def record_audio(classify_and_delete=False, model_file=None, verbose=False):
    model = None
    if classify_and_delete:
        model = joblib.load(model_file)
        logger.info("Classifying in real time, only saving trains")
        print("Classifying in real time and only saving trains")
    else:
        print("Recording all loud sustained noises.")
        logger.info("Recording all loud sustained noises.")
    p = pyaudio.PyAudio()

    device_index = p.get_default_input_device_info()["index"]

    stream = p.open(
        format=FORMAT,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
        input_device_index=device_index,
    )

    frames = []
    is_recording = False
    quiet_frames = 0
    loud_frames = 0
    while True:
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)

        if detect_train_horn(audio_data):
            logger.info("Possible train horn detected at:", datetime.datetime.now())
            if verbose:
                print("Possible train horn detected at:", datetime.datetime.now())
            loud_frames += 1
            frames.append(data)
            is_recording = True

        if is_recording and not detect_train_horn(audio_data):
            frames.append(data)
            quiet_frames += 1
            if quiet_frames > RECORD_DELAY:
                if loud_frames > LENGTH_THRESHOLD:
                    logger.info(
                        "Possible train horn ended at:", datetime.datetime.now()
                    )
                    logger.info("confirmed " + str(LENGTH_THRESHOLD) + " loud frames")

                    if verbose:
                        print("Possible train horn ended at:", datetime.datetime.now())
                        print("confirmed " + str(LENGTH_THRESHOLD) + " loud frames")
                    file_name = save_recorded_clip(frames, verbose=verbose)

                    if classify_and_delete and classify_audio(file_name, model) == 1:
                        print(f"Train detected at {format_filename(file_name)}")
                        logger.info(f"Train detected at {format_filename(file_name)}")
                        print(
                            f"Train detected at {format_filename(file_name)}",
                            file=open("./train.log", "w+"),
                        )
                        save_recorded_clip(
                            frames, save_to_special_dir=True, verbose=verbose
                        )
                        send_text(format_filename(file_name), PHONE_NUMBER)
                        os.remove(file_name)
                    elif classify_and_delete:
                        logger.info("Noise was not a train, deleting recording")
                        if verbose:
                            print("Not a train, deleting recording")
                        os.remove(file_name)

                else:
                    logger.info("Noise wasn't log enough to be a train")
                    if verbose:
                        print(
                            "not identified as a train, loud_frames was ", loud_frames
                        )

                frames = []
                is_recording = False
                quiet_frames = 0
                loud_frames = 0

    stream.stop_stream()
    stream.close()
    p.terminate()


def send_text(dtg, number):
    cmd = [BIN_PATH, "send", "-m", f"Train coming by your apartment: {dtg}", number]
    subprocess.Popen(cmd)


def save_recorded_clip(frames, save_to_special_dir=False, verbose=False):
    if save_to_special_dir:
        filename = f"./recordings/classified_trains/train_horn_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    else:
        filename = f"./recordings/train_horn_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    wf = wave.open(filename, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(2)  # 16-bit audio
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()
    logger.info("Saved recorded clip as:", filename)
    if verbose:
        print("Saved recorded clip as:", filename)
    return filename


if __name__ == "__main__":
    logging.basicConfig(filename="trainTrackER.log", level=logging.INFO)

    parser = argparse.ArgumentParser(description="Process optional arguments.")
    parser.add_argument(
        "--classify_and_delete",
        type=bool,
        default=False,
        help="If you want to real time process noises and only save trains",
    )
    parser.add_argument("--model", type=str, help="Path to model", default=None)
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="For more logging",
    )

    args = parser.parse_args()

    record_audio(args.classify_and_delete, args.model, args.verbose)
