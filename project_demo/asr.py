"""Optional real NeMo Speech transcription; not an N-best oracle generator."""
def transcribe(audio_paths, model_name="nvidia/parakeet-tdt-0.6b-v3"):
    from nemo.collections.asr.models import ASRModel
    model = ASRModel.from_pretrained(model_name=model_name)
    result = model.transcribe(audio=audio_paths)
    return [r.text if hasattr(r, "text") else str(r) for r in result]
