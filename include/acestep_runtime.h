#ifndef ACESTEP_RUNTIME_H
#define ACESTEP_RUNTIME_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AceContext AceContext;

AceContext* ace_create_context(const char* config_json);
void ace_free_context(AceContext* ctx);

void ace_string_free(char* ptr);
char* ace_last_error(void);

int32_t ace_prepare_step_inputs(
    AceContext* ctx,
    const char* state_json,
    const float* in_tensor_ptr,
    size_t in_tensor_len,
    char** out_json
);

int32_t ace_scheduler_step(
    AceContext* ctx,
    const float* xt_ptr,
    const float* vt_ptr,
    size_t len,
    float dt,
    float* out_xt_ptr
);

int32_t ace_apply_lm_constraints(
    AceContext* ctx,
    const float* logits_ptr,
    size_t vocab_size,
    float* out_masked_logits_ptr
);

int32_t ace_finalize_metadata(
    AceContext* ctx,
    const int64_t* token_ids_ptr,
    size_t len,
    char** out_json
);

#ifdef __cplusplus
}
#endif

#endif
