"""Replay B on exact cached user messages, without regenerating legacy prompts.

This is a transmission regression test, not correction inference or CER evaluation.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from cbsensevoice_covo_bridge import supplement_missing_kws_hotwords, normalize_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.summary.exists():
        raise FileExistsError('Use new output paths; never overwrite prior artifacts')
    summary = dict(kind='B transmission-only cached-message replay; no model inference',
                   samples=0, changed_samples=0, added_unique_per_utterance=0,
                   recorded_unique_per_utterance=0, visible_before=0, visible_after=0,
                   total_added_characters=0, max_added_characters=0,
                   source=str(args.input), source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest())
    with args.input.open(encoding='utf-8') as source, args.output.open('x', encoding='utf-8') as target:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            messages = copy.deepcopy([m for m in record['messages'] if m['role'] in ('system', 'user')])
            users = [m for m in messages if m['role'] == 'user']
            assert len(users) == 1
            old = users[0]['content']
            lines = old.split('\n')
            extra, audit = supplement_missing_kws_hotwords(record['input'], lines)
            users[0]['content'] = '\n'.join(lines[:-1] + extra + lines[-1:]) if extra else old
            assert not audit['missing_after']
            for word in audit['added']:
                assert normalize_text(word) in normalize_text(users[0]['content'])
            added_chars = len(users[0]['content']) - len(old)
            summary['samples'] += 1
            summary['changed_samples'] += bool(extra)
            summary['added_unique_per_utterance'] += len(audit['added'])
            summary['recorded_unique_per_utterance'] += audit['recorded_unique']
            summary['visible_before'] += audit['visible_before']
            summary['visible_after'] += audit['visible_after']
            summary['total_added_characters'] += added_chars
            summary['max_added_characters'] = max(summary['max_added_characters'], added_chars)
            out = {k: record[k] for k in ('id', 'source', 'dataset', 'split', 'reference', 'input') if k in record}
            out.update(messages=messages, hotword_transmission=audit)
            # Old predictions and assistant gold completions are deliberately not copied.
            target.write(json.dumps(out, ensure_ascii=False) + '\n')
    summary['output_sha256'] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
