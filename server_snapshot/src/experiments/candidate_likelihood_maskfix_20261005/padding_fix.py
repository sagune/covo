"""Process-local Qwen3.5 batch-one padding fix. Never edits installed packages."""
import hashlib
import inspect
from pathlib import Path

EXPECTED_BACKEND_SHA256 = '159e92f43efef4d8a7ec57c2ec1906387ef9e8d6c5e12878de957aa802ccc751'


def apply_mask_to_padding_states(hidden_states, attention_mask):
    # Preserve the original sequence-length guard. Remove only the batch>1 guard.
    if attention_mask is not None and attention_mask.shape[1] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states


def install():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as backend
    source = Path(inspect.getfile(backend))
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if sha != EXPECTED_BACKEND_SHA256:
        raise RuntimeError('Backend source differs; refusing an unverified patch: ' + sha)
    old = backend.apply_mask_to_padding_states
    old_text = inspect.getsource(old)
    if 'attention_mask.shape[0] > 1' not in old_text:
        raise RuntimeError('Expected batch-one mask guard not found')
    backend.apply_mask_to_padding_states = apply_mask_to_padding_states
    return dict(backend_path=str(source), backend_sha256=sha,
        original_function_sha256=hashlib.sha256(old_text.encode()).hexdigest(),
        patched_function_sha256=hashlib.sha256(inspect.getsource(apply_mask_to_padding_states).encode()).hexdigest(),
        process_local=True, installed_package_modified=False,
        change='Remove batch>1 guard only; keep sequence-length guard and original dtype/mask operation')
