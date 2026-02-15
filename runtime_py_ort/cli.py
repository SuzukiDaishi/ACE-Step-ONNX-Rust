from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import List

from .audio_parity import _save_audio, decode_audio_from_npz, run_audio_parity
from .case_schema import CaseSpec
from .constrained_decoding import SamplingConfig
from .parity import compare_npz, write_report
from .pipeline_core import CorePipeline
from .pipeline_lm import LMPipeline


def _parse_prompt_ids(raw: str | None) -> List[int]:
    if not raw:
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_bool(raw: str) -> bool:
    v = str(raw).strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {raw}")


def cmd_generate(args) -> int:
    case = CaseSpec.from_path(args.case)
    pipe = CorePipeline(args.onnx_dir, provider=args.provider, qwen_tokenizer_path=args.qwen_tokenizer)
    out_npz = Path(args.out_npz)
    pipe.generate_from_precomputed(
        case,
        Path(args.inputs_npz),
        out_npz,
        use_online_qwen_embed=args.online_qwen_embed,
        prefer_precomputed_condition=args.prefer_precomputed_condition,
        text=args.text,
        lyrics=args.lyrics,
        text_max_tokens=args.text_max_tokens,
        lyric_max_tokens=args.lyric_max_tokens,
    )
    print(f"Generated latent artifact: {out_npz}")
    return 0


def cmd_parity(args) -> int:
    report = compare_npz(Path(args.baseline), Path(args.candidate))
    write_report(report, Path(args.out_json))
    print(f"Wrote report: {args.out_json}")
    return 0


def cmd_audio_parity(args) -> int:
    case = CaseSpec.from_path(args.case)
    case_id = case.case_id
    candidate_npz = Path(args.candidate_npz) if args.candidate_npz else Path(f"reports/parity_py_ort/{case_id}_pyort.npz")
    out_audio = Path(args.out_audio) if args.out_audio else Path(f"reports/parity_py_ort/{case_id}_pyort.wav")
    out_json = Path(args.out_json) if args.out_json else Path(f"reports/parity_py_ort/{case_id}_audio_report.json")
    run_audio_parity(
        case_id=case_id,
        onnx_dir=Path(args.onnx_dir),
        candidate_npz=candidate_npz,
        out_audio=out_audio,
        out_json=out_json,
        provider=args.provider,
    )
    print(f"Wrote audio report: {out_json}")
    return 0


def cmd_full_generate(args) -> int:
    case = CaseSpec.from_path(args.case)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    case_id = case.case_id

    lm_model = args.lm_model if args.lm_model != "auto" else case.lm_model_variant
    effective_text = args.text
    effective_lyrics = args.lyrics
    used_online_simple_mode = False

    if args.mode in {"simple_mode", "auto"} and case.mode == "simple_mode":
        prompt_ids = _parse_prompt_ids(args.prompt_ids)
        sampling = SamplingConfig(
            temperature=0.0 if args.deterministic else args.temperature,
            top_k=0 if args.deterministic else args.top_k,
            top_p=1.0 if args.deterministic else args.top_p,
        )
        lm = LMPipeline(args.onnx_dir, provider=args.provider, lm_model_variant=lm_model)
        if prompt_ids:
            lm_out = lm.generate_tokens(
                prompt_ids=prompt_ids,
                max_new_tokens=args.max_new_tokens,
                sampling=sampling,
                seed=case.seed,
                deterministic=args.deterministic,
            )
            lm_json = out_dir / f"{case_id}_lm_tokens.json"
            lm_json.write_text(json.dumps({"token_ids": lm_out.token_ids}, indent=2), encoding="utf-8")
            print(f"Wrote LM tokens: {lm_json}")
        else:
            instrumental = bool((case.metadata or {}).get("instrumental", False))
            vocal_language = str((case.metadata or {}).get("vocal_language", "unknown"))
            lm_sample = lm.generate_sample_from_query(
                query=case.simple_mode_query,
                instrumental=instrumental,
                vocal_language=vocal_language,
                max_new_tokens=args.max_new_tokens,
                sampling=sampling,
                seed=case.seed,
                deterministic=args.deterministic,
                use_constrained_decoding=args.lm_constrained,
                tokenizer_path=args.lm_tokenizer,
            )
            lm_json = out_dir / f"{case_id}_lm_sample.json"
            lm_json.write_text(
                json.dumps(
                    {
                        "token_ids": lm_sample.token_ids,
                        "raw_text": lm_sample.raw_text,
                        "caption": lm_sample.caption,
                        "lyrics": lm_sample.lyrics,
                        "metadata": lm_sample.metadata,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(f"Wrote LM sample: {lm_json}")

            merged_meta = dict(case.metadata or {})
            for key in ["bpm", "duration", "keyscale", "language", "timesignature", "genres", "instrumental"]:
                if key in lm_sample.metadata and lm_sample.metadata.get(key) not in [None, "", "N/A"]:
                    merged_meta[key] = lm_sample.metadata[key]
            if "language" in merged_meta and "vocal_language" not in merged_meta:
                merged_meta["vocal_language"] = merged_meta["language"]

            case = replace(case, caption=lm_sample.caption, lyrics=lm_sample.lyrics, metadata=merged_meta)
            effective_text = lm_sample.caption
            effective_lyrics = lm_sample.lyrics
            used_online_simple_mode = True

    candidate_npz = Path(args.out_npz) if args.out_npz else out_dir / f"{case_id}_pyonnx.npz"
    pipe = CorePipeline(args.onnx_dir, provider=args.provider, qwen_tokenizer_path=args.qwen_tokenizer)
    pipe.generate_from_precomputed(
        case,
        Path(args.inputs_npz),
        candidate_npz,
        use_online_qwen_embed=(args.online_qwen_embed or used_online_simple_mode),
        prefer_precomputed_condition=args.prefer_precomputed_condition,
        text=effective_text,
        lyrics=effective_lyrics,
        text_max_tokens=args.text_max_tokens,
        lyric_max_tokens=args.lyric_max_tokens,
    )

    if not args.skip_audio:
        audio_provider = args.audio_provider if args.audio_provider else args.provider
        candidate_audio = decode_audio_from_npz(Path(args.onnx_dir), candidate_npz, provider=audio_provider)[0]
        out_audio = Path(args.out_audio) if args.out_audio else out_dir / f"{case_id}_pyonnx.wav"
        _save_audio(out_audio, candidate_audio, sample_rate=args.sample_rate)
        print(f"Wrote candidate wav: {out_audio}")
    else:
        print("Skipped audio decode (skip-audio enabled)")

    print(f"Wrote candidate npz: {candidate_npz}")
    return 0


def cmd_full_parity(args) -> int:
    case = CaseSpec.from_path(args.case)
    case_id = case.case_id
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_npz = Path(args.baseline_npz) if args.baseline_npz else Path(f"fixtures/tensors/{case_id}.npz")
    candidate_npz = Path(args.candidate_npz) if args.candidate_npz else out_dir / f"{case_id}_pyonnx.npz"
    parity_json = Path(args.parity_json) if args.parity_json else out_dir / f"{case_id}_tensor_report.json"

    tensor_report = compare_npz(baseline_npz, candidate_npz)
    write_report(tensor_report, parity_json)

    audio_wav = Path(args.candidate_wav) if args.candidate_wav else out_dir / f"{case_id}_pyonnx.wav"
    audio_json = Path(args.audio_json) if args.audio_json else out_dir / f"{case_id}_audio_report.json"
    audio_provider = args.audio_provider if args.audio_provider else args.provider
    run_audio_parity(
        case_id=case_id,
        onnx_dir=Path(args.onnx_dir),
        candidate_npz=candidate_npz,
        out_audio=audio_wav,
        out_json=audio_json,
        provider=audio_provider,
    )

    print(f"Wrote tensor report: {parity_json}")
    print(f"Wrote audio report : {audio_json}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="runtime_py_ort CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="run core generation from precomputed tensors")
    g.add_argument("--case", type=Path, required=True)
    g.add_argument("--onnx-dir", type=Path, default=Path("artifacts/onnx_runtime"))
    g.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    g.add_argument("--inputs-npz", type=Path, required=True)
    g.add_argument("--out-npz", type=Path, required=True)
    g.add_argument("--online-qwen-embed", action="store_true")
    g.add_argument("--prefer-precomputed-condition", type=_parse_bool, default=True)
    g.add_argument("--qwen-tokenizer", type=Path, default=Path("checkpoints/Qwen3-Embedding-0.6B/tokenizer.json"))
    g.add_argument("--text", type=str)
    g.add_argument("--lyrics", type=str)
    g.add_argument("--text-max-tokens", type=int, default=256)
    g.add_argument("--lyric-max-tokens", type=int, default=2048)
    g.set_defaults(func=cmd_generate)

    c = sub.add_parser("parity", help="compare baseline and generated npz")
    c.add_argument("--baseline", type=Path, required=True)
    c.add_argument("--candidate", type=Path, required=True)
    c.add_argument("--out-json", type=Path, required=True)
    c.set_defaults(func=cmd_parity)

    a = sub.add_parser("audio-parity", help="decode audio via VAE and compare with baseline wav")
    a.add_argument("--case", type=Path, required=True)
    a.add_argument("--onnx-dir", type=Path, default=Path("artifacts/onnx_runtime"))
    a.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    a.add_argument("--audio-provider", choices=["cpu", "cuda"])
    a.add_argument("--candidate-npz", type=Path)
    a.add_argument("--out-audio", type=Path)
    a.add_argument("--out-json", type=Path)
    a.set_defaults(func=cmd_audio_parity)

    fg = sub.add_parser("full-generate", help="generate core latents + wav, optionally LM tokens")
    fg.add_argument("--case", type=Path, required=True)
    fg.add_argument("--inputs-npz", type=Path, required=True)
    fg.add_argument("--onnx-dir", type=Path, default=Path("artifacts/onnx_runtime"))
    fg.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    fg.add_argument("--audio-provider", choices=["cpu", "cuda"])
    fg.add_argument("--out-dir", type=Path, default=Path("reports/parity_py_ort"))
    fg.add_argument("--out-npz", type=Path)
    fg.add_argument("--out-audio", type=Path)
    fg.add_argument("--sample-rate", type=int, default=48000)
    fg.add_argument("--online-qwen-embed", action="store_true")
    fg.add_argument("--prefer-precomputed-condition", type=_parse_bool, default=True)
    fg.add_argument("--qwen-tokenizer", type=Path, default=Path("checkpoints/Qwen3-Embedding-0.6B/tokenizer.json"))
    fg.add_argument("--text", type=str)
    fg.add_argument("--lyrics", type=str)
    fg.add_argument("--text-max-tokens", type=int, default=256)
    fg.add_argument("--lyric-max-tokens", type=int, default=2048)
    fg.add_argument("--mode", choices=["auto", "simple_mode", "text2music"], default="auto")
    fg.add_argument("--lm-model", choices=["auto", "0.6B", "1.7B"], default="auto")
    fg.add_argument("--lm-tokenizer", type=Path)
    fg.add_argument("--lm-constrained", type=_parse_bool, default=True)
    fg.add_argument("--deterministic", type=_parse_bool, default=True)
    fg.add_argument("--prompt-ids", type=str, default="")
    fg.add_argument("--max-new-tokens", type=int, default=128)
    fg.add_argument("--temperature", type=float, default=1.0)
    fg.add_argument("--top-k", type=int, default=0)
    fg.add_argument("--top-p", type=float, default=1.0)
    fg.add_argument("--skip-audio", type=_parse_bool, default=False)
    fg.set_defaults(func=cmd_full_generate)

    fp = sub.add_parser("full-parity", help="run tensor + audio parity in one command")
    fp.add_argument("--case", type=Path, required=True)
    fp.add_argument("--onnx-dir", type=Path, default=Path("artifacts/onnx_runtime"))
    fp.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    fp.add_argument("--audio-provider", choices=["cpu", "cuda"])
    fp.add_argument("--out-dir", type=Path, default=Path("reports/parity_py_ort"))
    fp.add_argument("--baseline-npz", type=Path)
    fp.add_argument("--candidate-npz", type=Path)
    fp.add_argument("--candidate-wav", type=Path)
    fp.add_argument("--parity-json", type=Path)
    fp.add_argument("--audio-json", type=Path)
    fp.set_defaults(func=cmd_full_parity)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
