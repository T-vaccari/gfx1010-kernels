#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <tuple>
#include <vector>


std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
residual_layer_norm_forward_hip(
    const at::Tensor& x,
    const at::Tensor& branch,
    const at::Tensor& weight,
    const at::Tensor& bias,
    double dropout_p,
    double eps,
    bool training,
    int64_t threads);


std::vector<at::Tensor> residual_layer_norm_backward_hip(
    const at::Tensor& grad_updated,
    const at::Tensor& grad_normalized,
    const at::Tensor& updated,
    const at::Tensor& weight,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const at::Tensor& dropout_mask,
    double dropout_p,
    bool training,
    bool branch_is_half,
    int64_t threads,
    int64_t block_rows,
    int64_t parameter_threads,
    int64_t parameter_groups,
    bool compute_parameter_gradients);
