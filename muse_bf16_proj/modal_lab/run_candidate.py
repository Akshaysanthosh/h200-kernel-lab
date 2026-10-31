"""One bounded H200 experiment. Launch only after dashboard billing verification."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from run_lt_check import candidate_path, verify_champion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('candidate')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--seed', type=int, default=82731)
    parser.add_argument('--mode', choices=('benchmark','tune','hopper','other','packed','native','cutlass','decode','public','public_tune','public_compressed','public_refine','public_wide','pipeline','pipeline_debug','tinygemm','profile_compressed','tune_decode','validate_decode','zip_format','vector_decoder','exception_patch','padded_layout','dense_schedule','transposed_schedule','qkv_geometry','qkv_control','router_interleave','specialized_correction','sk2_refine'), default='benchmark')
    args = parser.parse_args()
    candidate = candidate_path(args.candidate)
    verify_champion()
    print(json.dumps({'candidate':args.candidate,'sha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),
                      'gpu':'H200','max_gpu_seconds':900,'execute':args.execute}))
    if not args.execute:
        return
    # This record is created only after directly verifying the account dashboard.
    gate = json.loads((ROOT / 'results/modal/billing-verified.json').read_text())
    checked = datetime.fromisoformat(gate['verified_at'])
    age = (datetime.now(timezone.utc)-checked).total_seconds()
    if not (0 <= age < 3600 and gate['spend_limit_usd'] == 0
            and gate['remaining_compute_credits_usd'] >= 3 and gate['workspace']
            and gate.get('cli_workspace_verified') and gate.get('cli_profile') == 'okc-free'):
        raise RuntimeError('Reverify workspace, $0 spend limit and >=$3 free credits before each run')
    if args.mode in ('public','public_tune','public_compressed','public_refine','public_wide','pipeline','pipeline_debug','tinygemm','profile_compressed','tune_decode','validate_decode','zip_format','vector_decoder','exception_patch','padded_layout','dense_schedule','transposed_schedule','qkv_geometry','qkv_control','router_interleave','specialized_correction','sk2_refine'):
        base_image = modal.Image.from_registry('lmsysorg/sglang:dev-muse-spark-open').pip_install('numpy','ninja')
    else:
        base_image = modal.Image.from_registry('nvidia/cuda:12.8.1-devel-ubuntu22.04',add_python='3.11').pip_install('torch==2.9.0','numpy','ninja')
    image = (base_image
             .env({'MAX_JOBS':'2','TORCH_CUDA_ARCH_LIST':'9.0'})
             .add_local_file(str(ROOT/'reference.py'),'/work/reference.py')
             .add_local_file(str(ROOT/'checks.py'),'/work/checks.py')
             .add_local_file(str(ROOT/'champions/2026-10-30_1.227412/submission.py'),'/work/champion.py')
             .add_local_file(str(candidate),'/work/candidate.py')
             .add_local_file(str(ROOT/'modal_lab/benchmark.py'),'/work/benchmark.py')
             .add_local_file(str(ROOT/'modal_lab/tune_algorithms.py'),'/work/tune_algorithms.py')
             .add_local_file(str(ROOT/'modal_lab/tune_hopper.py'),'/work/tune_hopper.py')
             .add_local_file(str(ROOT/'modal_lab/tune_other.py'),'/work/tune_other.py')
             .add_local_file(str(ROOT/'modal_lab/tune_packed.py'),'/work/tune_packed.py')
             .add_local_file(str(ROOT/'modal_lab/test_native.py'),'/work/test_native.py')
             .add_local_file(str(ROOT/'modal_lab/test_cutlass.py'),'/work/test_cutlass.py'))
    if args.mode in ('public','public_tune','public_compressed','public_refine','public_wide','pipeline','pipeline_debug','tinygemm','profile_compressed','tune_decode','validate_decode','zip_format','vector_decoder','exception_patch','padded_layout','dense_schedule','transposed_schedule','qkv_geometry','qkv_control','router_interleave','specialized_correction','sk2_refine'):
        for local, remote in (
            ('modal_lab/test_sk2_refine.py', 'test_sk2_refine.py'),
            ('modal_lab/test_specialized_correction.py', 'test_specialized_correction.py'),
            ('research/2026-10-31/vjeux/muse_bf16_proj.py', 'vjeux.py'),
            ('modal_lab/test_router_interleave.py', 'test_router_interleave.py'),
            ('modal_lab/router_interleave.py', 'router_interleave.py'),
            ('modal_lab/test_qkv_control.py', 'test_qkv_control.py'),
            ('modal_lab/qkv_retest_configs.json', 'qkv_retest_configs.json'),
            ('modal_lab/test_qkv_geometry.py', 'test_qkv_geometry.py'),
            ('modal_lab/test_transposed_schedule.py', 'test_transposed_schedule.py'),
            ('modal_lab/transposed_schedule.cu', 'transposed_schedule.cu'),
            ('modal_lab/test_dense_schedule.py', 'test_dense_schedule.py'),
            ('modal_lab/dense_schedule.cu', 'dense_schedule.cu'),
            ('modal_lab/test_padded_layout.py', 'test_padded_layout.py'),
            ('modal_lab/padded_layout.py', 'padded_layout.py'),
            ('modal_lab/test_exception_patch.py', 'test_exception_patch.py'),
            ('modal_lab/exception_patch.py', 'exception_patch.py'),
            ('modal_lab/test_vector_decoder.py', 'test_vector_decoder.py'),
            ('modal_lab/vector_decoder.py', 'vector_decoder.py'),
            ('modal_lab/test_zip_format.py', 'test_zip_format.py'),
            ('modal_lab/zip_format.py', 'zip_format.py'),
            ('modal_lab/validate_decode.py', 'validate_decode.py'),
            ('modal_lab/tune_decode.py', 'tune_decode.py'),
            ('modal_lab/decode_variants.py', 'decode_variants.py'),
            ('modal_lab/reuse_variant.py', 'reuse_variant.py'),
            ('modal_lab/profile_compressed.py', 'profile_compressed.py'),
            ('modal_lab/test_tinygemm.py', 'test_tinygemm.py'),
            ('modal_lab/tinygemm.cu', 'tinygemm.cu'),
            ('modal_lab/debug_pipeline.py', 'debug_pipeline.py'),
            ('modal_lab/pipeline_source.py', 'pipeline_source.py'),
            ('modal_lab/test_pipeline.py', 'test_pipeline.py'),
            ('modal_lab/tune_public_wide.py', 'tune_public_wide.py'),
            ('modal_lab/test_public_refine.py', 'test_public_refine.py'),
            ('candidates/public/nathen_v18_linear.py', 'parent.py'),
            ('modal_lab/tune_public_compressed.py', 'tune_public_compressed.py'),
            ('modal_lab/tune_public.py', 'tune_public.py'),
            ('modal_lab/test_public.py', 'test_public.py'),
            ('modal_lab/hybrid_comparison.py', 'hybrid.py'),
        ):
            image = image.add_local_file(str(ROOT/local), '/work/'+remote)
    if args.mode == 'decode':
        image=image.add_local_file(str(ROOT/'modal_lab/test_decode.py'),'/work/test_decode.py').add_local_dir(str(ROOT/'cutlass_decode/include'),'/opt/cutlass_decode/include')
    if args.mode in ('cutlass','decode','dense_schedule','transposed_schedule','qkv_geometry','qkv_control','router_interleave','specialized_correction','sk2_refine'):
        image=image.add_local_dir(str(ROOT/'vendor/cutlass/include'),'/opt/cutlass/include')
    app = modal.App('okc-free-h200-lab')
    from remote_task import evaluate
    remote_evaluate = app.function(image=image,gpu='H200',cpu=4,memory=16384,timeout=900,
                                  max_containers=1,retries=0,scaledown_window=2)(evaluate)
    outdir = ROOT/'results/modal'
    outdir.mkdir(parents=True,exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    with modal.enable_output(), app.run():
        result = json.loads(remote_evaluate.remote(args.candidate,args.mode,args.seed))
    result['source_sha256'] = hashlib.sha256(candidate.read_bytes()).hexdigest()
    path = outdir / f'{stamp}_{candidate.stem}_{args.mode}.json'
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'result_file':str(path),'passed':result['passed'],
                      'relative_to_champion':result.get('relative_to_champion')}))


if __name__ == '__main__':
    main()
