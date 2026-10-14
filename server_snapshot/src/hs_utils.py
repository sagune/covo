import torch


def quantize_hidden_states(hidden_states: torch.Tensor) -> dict:
    hidden_states = hidden_states.detach().cpu().clamp_(-1.0, 1.0)
    return {
        'quantized': True,
        'values': torch.round(hidden_states * 127.0).to(torch.int8)
    }


def dequantize_hidden_states(payload) -> torch.Tensor:
    if isinstance(payload, dict) and payload.get('quantized', False):
        values = payload['values']
        if torch.is_floating_point(values):
            return values.to(torch.float32)
        return values.to(torch.float32) / 127.0
    if isinstance(payload, torch.Tensor):
        return payload.detach().to(torch.float32)
    raise TypeError(f'unsupported hidden states payload type: {type(payload)}')


def load_hidden_states(file_obj, map_location=torch.device('cpu')) -> torch.Tensor:
    payload = torch.load(file_obj, map_location=map_location)
    return dequantize_hidden_states(payload).detach()
