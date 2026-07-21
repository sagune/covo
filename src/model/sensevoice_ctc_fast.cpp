#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

constexpr double kNegInf = -std::numeric_limits<double>::infinity();

double logadd(double a, double b) {
    if (a == kNegInf) return b;
    if (b == kNegInf) return a;
    const double maximum = std::max(a, b);
    return maximum + std::log(std::exp(a - maximum) + std::exp(b - maximum));
}

struct VectorHash {
    std::size_t operator()(const std::vector<int>& values) const noexcept {
        std::size_t seed = values.size();
        for (const int value : values) {
            seed ^= static_cast<std::size_t>(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

struct BeamState {
    double search_blank = kNegInf;
    double search_nonblank = kNegInf;
    double acoustic_blank = kNegInf;
    double acoustic_nonblank = kNegInf;
};

int suffix_prefix_progress(const std::vector<int>& prefix, const std::vector<int>& hotword) {
    const int upper = std::min(prefix.size(), hotword.size());
    for (int size = upper; size > 0; --size) {
        bool matches = true;
        for (int index = 0; index < size; ++index) {
            if (prefix[prefix.size() - size + index] != hotword[index]) {
                matches = false;
                break;
            }
        }
        if (matches) return size;
    }
    return 0;
}

double extension_bonus(
    const std::vector<int>& prefix,
    int token,
    const std::vector<std::vector<int>>& hotwords,
    const std::vector<double>& confidences,
    double token_weight,
    double completion_weight
) {
    if (hotwords.empty()) return 0.0;
    std::vector<int> extended(prefix);
    extended.push_back(token);
    double best = 0.0;
    for (std::size_t index = 0; index < hotwords.size(); ++index) {
        const auto& hotword = hotwords[index];
        if (hotword.empty()) continue;
        const int old_progress = suffix_prefix_progress(prefix, hotword);
        const int new_progress = suffix_prefix_progress(extended, hotword);
        if (new_progress <= old_progress) continue;
        const double confidence = index < confidences.size() ? std::max(0.0, confidences[index]) : 0.0;
        double reward = token_weight * confidence / static_cast<double>(hotword.size());
        if (new_progress == static_cast<int>(hotword.size())) reward += completion_weight * confidence;
        best = std::max(best, reward);
    }
    return best;
}

void update_blank(BeamState& state, double search, double acoustic) {
    state.search_blank = logadd(state.search_blank, search);
    state.acoustic_blank = logadd(state.acoustic_blank, acoustic);
}

void update_nonblank(BeamState& state, double search, double acoustic) {
    state.search_nonblank = logadd(state.search_nonblank, search);
    state.acoustic_nonblank = logadd(state.acoustic_nonblank, acoustic);
}

}  // namespace

std::vector<std::tuple<std::vector<int>, double, double, double>> decode(
    const std::vector<std::vector<double>>& frame_values,
    const std::vector<std::vector<int>>& frame_indices,
    int beam_size,
    int blank_id,
    const std::vector<std::vector<int>>& hotwords,
    const std::vector<double>& confidences,
    double token_weight,
    double completion_weight,
    const std::vector<int>& excluded_tokens
) {
    beam_size = std::max(1, beam_size);
    const std::unordered_set<int> excluded(excluded_tokens.begin(), excluded_tokens.end());
    std::unordered_map<std::vector<int>, BeamState, VectorHash> beams;
    BeamState initial;
    initial.search_blank = 0.0;
    initial.acoustic_blank = 0.0;
    beams.emplace(std::vector<int>{}, initial);

    const std::size_t time_steps = std::min(frame_values.size(), frame_indices.size());
    for (std::size_t time = 0; time < time_steps; ++time) {
        std::unordered_map<int, double> frame;
        const std::size_t width = std::min(frame_values[time].size(), frame_indices[time].size());
        for (std::size_t column = 0; column < width; ++column) {
            frame[frame_indices[time][column]] = frame_values[time][column];
        }

        std::unordered_map<std::vector<int>, BeamState, VectorHash> next;
        next.reserve(static_cast<std::size_t>(beam_size) * std::max<std::size_t>(frame.size(), 1));
        for (const auto& beam : beams) {
            const auto& prefix = beam.first;
            const auto& state = beam.second;
            const double search_total = logadd(state.search_blank, state.search_nonblank);
            const double acoustic_total = logadd(state.acoustic_blank, state.acoustic_nonblank);
            for (const auto& token_score : frame) {
                const int token = token_score.first;
                const double token_log_prob = token_score.second;
                if (token == blank_id) {
                    auto& output = next[prefix];
                    update_blank(output, search_total + token_log_prob, acoustic_total + token_log_prob);
                    continue;
                }
                if (excluded.count(token) != 0) continue;

                const int last_token = prefix.empty() ? -1 : prefix.back();
                if (token == last_token) {
                    auto& same = next[prefix];
                    update_nonblank(
                        same,
                        state.search_nonblank + token_log_prob,
                        state.acoustic_nonblank + token_log_prob
                    );
                    std::vector<int> extended(prefix);
                    extended.push_back(token);
                    const double bonus = extension_bonus(
                        prefix, token, hotwords, confidences, token_weight, completion_weight
                    );
                    auto& output = next[extended];
                    update_nonblank(
                        output,
                        state.search_blank + token_log_prob + bonus,
                        state.acoustic_blank + token_log_prob
                    );
                    continue;
                }

                std::vector<int> extended(prefix);
                extended.push_back(token);
                const double bonus = extension_bonus(
                    prefix, token, hotwords, confidences, token_weight, completion_weight
                );
                auto& output = next[extended];
                update_nonblank(
                    output,
                    search_total + token_log_prob + bonus,
                    acoustic_total + token_log_prob
                );
            }
        }

        std::vector<std::pair<std::vector<int>, BeamState>> ranked;
        ranked.reserve(next.size());
        for (auto& item : next) ranked.emplace_back(std::move(item.first), item.second);
        const auto score_greater = [](const auto& left, const auto& right) {
            return logadd(left.second.search_blank, left.second.search_nonblank)
                > logadd(right.second.search_blank, right.second.search_nonblank);
        };
        if (ranked.size() > static_cast<std::size_t>(beam_size)) {
            std::partial_sort(ranked.begin(), ranked.begin() + beam_size, ranked.end(), score_greater);
            ranked.resize(beam_size);
        } else {
            std::sort(ranked.begin(), ranked.end(), score_greater);
        }
        beams.clear();
        beams.reserve(ranked.size());
        for (auto& item : ranked) beams.emplace(std::move(item.first), item.second);
    }

    std::vector<std::tuple<std::vector<int>, double, double, double>> output;
    output.reserve(beams.size());
    for (const auto& beam : beams) {
        const double search = logadd(beam.second.search_blank, beam.second.search_nonblank);
        const double acoustic = logadd(beam.second.acoustic_blank, beam.second.acoustic_nonblank);
        output.emplace_back(beam.first, acoustic, search, search - acoustic);
    }
    std::sort(output.begin(), output.end(), [](const auto& left, const auto& right) {
        return std::get<2>(left) > std::get<2>(right);
    });
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("decode", &decode, "Context-biased CTC prefix beam search");
}
