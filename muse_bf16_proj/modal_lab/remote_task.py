"""Importable Modal entry point; source loaded on remote Python, no code pickling."""
def evaluate(candidate_name,mode="benchmark",seed=82731):
    import sys
    sys.path.insert(0, '/work')
    if mode == "sk2_refine":
        from test_sk2_refine import run
    elif mode == "specialized_correction":
        from test_specialized_correction import run
    elif mode == "router_interleave":
        from test_router_interleave import run
    elif mode == "qkv_control":
        from test_qkv_control import run
    elif mode == "qkv_geometry":
        from test_qkv_geometry import run
    elif mode == "transposed_schedule":
        from test_transposed_schedule import run
    elif mode == "dense_schedule":
        from test_dense_schedule import run
    elif mode == "padded_layout":
        from test_padded_layout import run
    elif mode == "exception_patch":
        from test_exception_patch import run
    elif mode == "vector_decoder":
        from test_vector_decoder import run
    elif mode == "zip_format":
        from test_zip_format import run
    elif mode == "validate_decode":
        from validate_decode import run
    elif mode == "tune_decode":
        from tune_decode import run
    elif mode == "profile_compressed":
        from profile_compressed import run
    elif mode == "tinygemm":
        from test_tinygemm import run
    elif mode == "pipeline_debug":
        from debug_pipeline import run
    elif mode == "pipeline":
        from test_pipeline import run
    elif mode == "public_wide":
        from tune_public_wide import run
    elif mode == "public_refine":
        from test_public_refine import run
    elif mode == "public_compressed":
        from tune_public_compressed import run
    elif mode == "public_tune":
        from tune_public import run
    elif mode == "public":
        from test_public import run
    elif mode == "decode":
        from test_decode import run
    elif mode == "cutlass":
        from test_cutlass import run
    elif mode == "native":
        from test_native import run
    elif mode == "packed":
        from tune_packed import run
    elif mode == "other":
        from tune_other import run
    elif mode == "hopper":
        from tune_hopper import run
    elif mode == "tune":
        from tune_algorithms import run
    else:
        from benchmark import run
    import json
    result = run(candidate_name) if mode in ("tune","hopper","other","packed","native","cutlass","decode","public","public_tune","public_compressed","public_refine","public_wide","pipeline","pipeline_debug","tinygemm","profile_compressed","tune_decode","validate_decode","zip_format","vector_decoder","exception_patch","padded_layout","dense_schedule","transposed_schedule","qkv_geometry","qkv_control","router_interleave","specialized_correction","sk2_refine") else run(candidate_name,seed=seed)
    payload = json.dumps(result)
    print('OKC_MODAL_RESULT', payload, flush=True)
    return payload
