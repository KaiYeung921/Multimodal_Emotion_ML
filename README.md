# Multimodal signals used for predicting emotion

This project trains on the MELD dataset that contains sentences from Friends (the show), and has them labeled with its emotion and sentiment. The goal of this project is to make and end-to-end system that can take audio and text and detect emotion on a rolling basis

Audio sourced from ajyy/MELD_audio (lossless FLAC re-encode of the official mp4s, 16kHz mono), joined against the canonical declare-lab annotations on (Dialogue_ID, Utterance_ID). 13,706 of 13,708 utterances usable; dia125_utt3 (train) and dia110_utt7 (dev) have no corresponding audio and are dropped — both neutral. 141 unlabelled audio files (largely final_videos_test duplicates) ignored. Verified by scripts/verify_data.py - made by Claude.
