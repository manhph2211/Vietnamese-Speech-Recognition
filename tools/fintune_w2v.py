import flash
from flash.audio import SpeechRecognition, SpeechRecognitionData
import torch
import sys
sys.path.append(".")

WAV2VEC_MODELS = ["facebook/wav2vec2-base-960h", "facebook/wav2vec2-large-960h-lv60", "nguyenvulebinh/wav2vec2-base-vietnamese-250h"]

# 1. Data
datamodule = SpeechRecognitionData.from_json(
    "file",
    "text",
    train_file="train.json",
    test_file="test.json",
    batch_size=128,
)

# 2. Build the task
model = SpeechRecognition(backbone="nguyenvulebinh/wav2vec2-base-vietnamese-250h", processor_backbone = "nguyenvulebinh/wav2vec2-base-vietnamese-250h")

# # 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=5, gpus=0)
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# # 4. Predict on audio files!
datamodule = SpeechRecognitionData.from_files(predict_files=["demo/assets/database_sa1_Jan08_Mar19_cleaned_utt_0000000005-1.wav"], batch_size=1)
predictions = trainer.predict(model, datamodule=datamodule)
print(predictions)

# 5. Save the model!
# trainer.save_checkpoint("checkpoints/speech_recognition_model.pt")