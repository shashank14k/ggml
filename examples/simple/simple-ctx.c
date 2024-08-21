#include "ggml.h"
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

typedef struct
{
    struct ggml_tensor *a;
    struct ggml_tensor *b;
    struct ggml_context *ctx;
} simple_model;

void load_model(simple_model *model, float *a, float *b, int rows_A, int cols_A, int rows_B, int cols_B)
{
    float ctx_size = 0;
    ctx_size += rows_A * cols_A * ggml_type_size(GGML_TYPE_F32);
    ctx_size += rows_B * cols_B * ggml_type_size(GGML_TYPE_F32);
    ctx_size += 2 * ggml_tensor_overhead();
    ctx_size += ggml_graph_overhead();
    ctx_size += 1024;
    struct ggml_init_params params = {
        .mem_size = (size_t)ctx_size,
        .mem_buffer = NULL,
        .no_alloc = false};
    model->ctx = ggml_init(params);
    model->a = ggml_new_tensor_2d(model->ctx, GGML_TYPE_F32, cols_A, rows_A);
    model->b = ggml_new_tensor_2d(model->ctx, GGML_TYPE_F32, cols_B, rows_B);

    memcpy(model->a->data, a, ggml_nbytes(model->a));
    memcpy(model->b->data, b, ggml_nbytes(model->b));
}

struct ggml_cgraph *build_graph(const simple_model *model)
{
    struct ggml_cgraph *gf = ggml_new_graph(model->ctx);
    struct ggml_tensor *result = ggml_mul_mat(model->ctx, model->a, model->b);
    ggml_build_forward_expand(gf, result);
    return gf;
}

struct ggml_tensor *compute(const simple_model *model)
{
    struct ggml_cgraph *gf = build_graph(model);
    int n_threads = 1;

    ggml_graph_compute_with_ctx(model->ctx, gf, n_threads);
    return gf->nodes[gf->n_nodes - 1];
}

int main()
{
    ggml_time_init();

    const int rows_A = 4, cols_A = 2;
    float matrix_A_data[rows_A * cols_A];
    matrix_A_data[0] = 2; matrix_A_data[1] = 8;
    matrix_A_data[2] = 5; matrix_A_data[3] = 1;
    matrix_A_data[4] = 4; matrix_A_data[5] = 2;
    matrix_A_data[6] = 8; matrix_A_data[7] = 6;
    
    const int rows_B = 3, cols_B = 2;
    float matrix_B_data[rows_B * cols_B];
    matrix_B_data[0] = 10; matrix_B_data[1] = 5;
    matrix_B_data[2] = 9; matrix_B_data[3] = 9;
    matrix_B_data[4] = 5; matrix_B_data[5] = 4;


    float *matrix_A = matrix_A_data;
    float *matrix_B = matrix_B_data;

    simple_model *model = (simple_model *)malloc(sizeof(simple_model));
    load_model(model, matrix_A, matrix_B, rows_A, cols_A, rows_B, cols_B);

    struct ggml_tensor *result = compute(model);

    int64_t n_elements = ggml_nelements(result);
    float *out_data = (float *)malloc(n_elements * sizeof(float));
    if (out_data == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    memcpy(out_data, result->data, ggml_nbytes(result));

    printf("mul mat (%d x %d) (transposed result):\n[", (int) result->ne[0], (int) result->ne[1]);
    for (int j = 0; j < result->ne[1] /* rows */; j++) {
        if (j > 0) {
            printf("\n");
        }

        for (int i = 0; i < result->ne[0] /* cols */; i++) {
            printf(" %.2f", out_data[j * result->ne[0] + i]);
        }
    }
    printf(" ]\n");

    // free memory
    ggml_free(model->ctx);
    free(model);
    free(out_data);
    return 0;
}