"""Fixed optimal-alignment diagnostics, not a changed CER scoring rule."""
from collections import Counter


def edit_counts(reference, prediction):
    a, b = reference, prediction
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j - 1] + (a[i - 1] != b[j - 1]), d[i - 1][j] + 1, d[i][j - 1] + 1)
    counts = Counter(S=0, D=0, I=0)
    i, j = len(a), len(b)
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            counts["S"] += int(a[i - 1] != b[j - 1]); i -= 1; j -= 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            counts["D"] += 1; i -= 1
        else:
            counts["I"] += 1; j -= 1
    return counts


def diagnose(rows, normalize):
    total = Counter()
    for row in rows:
        reference = normalize(row["reference"])
        original = normalize(row["input"]["asr_top1"])
        prediction = normalize(row["prediction"])
        counts = edit_counts(reference, prediction)
        total.update(counts)
        if reference == original:
            total["input_correct_sentences"] += 1
            total["correct_input_broken_sentences"] += int(prediction != reference)
            total["errors_on_correct_inputs"] += sum(counts.values())
    return dict(total, alignment_tie_break="diagonal, deletion, insertion; S/D/I may vary between optimal alignments")
