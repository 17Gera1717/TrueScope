from nemo.collections.asr.models import EncDecCTCModel, EncDecCTCModelBPE

# Download and load a pre-trained Russian ASR model
# Several options are available:
# - "stt_ru_conformer_ctc_large" (best quality)
# - "stt_ru_conformer_ctc_medium" (balanced)
# - "stt_ru_quartznet15x5" (faster, older model)

asr_model = EncDecCTCModelBPE.from_pretrained(model_name="stt_ru_conformer_ctc_large")

# If you want to save the model locally to avoid downloading it again
asr_model.save_to("russian_asr_model.nemo")

# To load from a local file later
# asr_model = EncDecCTCModel.restore_from("russian_asr_model.nemo")