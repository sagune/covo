#!/usr/bin/env bash
set -euo pipefail

dataset="${1:-}"
if [[ -z "${dataset}" ]]; then
  echo "usage: $0 aishell|stcmds|aishell-targeted|stcmds-targeted [additional LightningCLI arguments...]" >&2
  exit 2
fi
shift
extra_model_args=()

case "${dataset}" in
  aishell)
    config="configs/cb-sensevoice-aishell.yaml"
    data_root="../datasets/aishell/data_aishell_sensevoice_full"
    oracle_detail="logs/oracle_nbest_detail_cb_sensevoice_aishell_full.csv"
    oracle_summary="logs/oracle_nbest_summary_cb_sensevoice_aishell_full.csv"
    ;;
  stcmds)
    config="configs/cb-sensevoice-stcmds.yaml"
    data_root="../datasets/stcmds/cb_sensevoice_heldout"
    oracle_detail="logs/oracle_nbest_detail_cb_sensevoice_stcmds_heldout.csv"
    oracle_summary="logs/oracle_nbest_summary_cb_sensevoice_stcmds_heldout.csv"
    ;;
  aishell-targeted)
    config="configs/cb-sensevoice-aishell.yaml"
    data_root="../datasets/aishell/data_aishell_sensevoice_full_error_targeted"
    oracle_detail="logs/oracle_nbest_detail_cb_sensevoice_aishell_full_error_targeted.csv"
    oracle_summary="logs/oracle_nbest_summary_cb_sensevoice_aishell_full_error_targeted.csv"
    extra_model_args+=(--model.init_args.kws_positive_threshold=0.95)
    ;;
  stcmds-targeted)
    config="configs/cb-sensevoice-stcmds.yaml"
    data_root="../datasets/stcmds/cb_sensevoice_heldout_error_targeted"
    oracle_detail="logs/oracle_nbest_detail_cb_sensevoice_stcmds_heldout_error_targeted.csv"
    oracle_summary="logs/oracle_nbest_summary_cb_sensevoice_stcmds_heldout_error_targeted.csv"
    extra_model_args+=(--model.init_args.kws_positive_threshold=0.95)
    ;;
  *)
    echo "unsupported dataset: ${dataset}" >&2
    exit 2
    ;;
esac

exec /root/autodl-tmp/great/bin/python run_CLI.py test \
  --config "${config}" \
  --data.init_args.test_info.root="${data_root}" \
  --data.init_args.num_workers=0 \
  --model.init_args.root="${data_root}/hotword" \
  --model.init_args.use_precomputed_kws_features=false \
  --model.init_args.kws_infer_chunk_size=64 \
  --model.init_args.kws_prefilter_per_group=32 \
  --model.init_args.kws_force_group_fallback=false \
  --model.init_args.oracle_nbest_detail_path="${oracle_detail}" \
  --model.init_args.oracle_nbest_summary_path="${oracle_summary}" \
  "${extra_model_args[@]}" \
  "$@"
