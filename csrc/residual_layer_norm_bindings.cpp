#include <torch/extension.h>
#include <torch/csrc/autograd/custom_function.h>

#include "residual_layer_norm.h"


namespace {


class ResidualLayerNormFunction
    : public torch::autograd::Function<ResidualLayerNormFunction> {
 public:
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::Variable x,
        torch::autograd::Variable branch,
        torch::autograd::Variable weight,
        torch::autograd::Variable bias,
        double dropout_p,
        double eps,
        bool training,
        int64_t threads,
        int64_t block_rows,
        int64_t parameter_threads,
        int64_t parameter_groups) {
        auto result = residual_layer_norm_forward_hip(
            x,
            branch,
            weight,
            bias,
            dropout_p,
            eps,
            training,
            threads);
        auto updated = std::get<0>(result);
        auto normalized = std::get<1>(result);
        ctx->save_for_backward(
            {
                updated,
                weight,
                std::get<2>(result),
                std::get<3>(result),
                std::get<4>(result),
            });
        ctx->set_materialize_grads(false);
        ctx->saved_data["dropout_p"] = dropout_p;
        ctx->saved_data["training"] = training;
        ctx->saved_data["branch_is_half"] =
            branch.scalar_type() == at::kHalf;
        ctx->saved_data["threads"] = threads;
        ctx->saved_data["block_rows"] = block_rows;
        ctx->saved_data["parameter_threads"] = parameter_threads;
        ctx->saved_data["parameter_groups"] = parameter_groups;
        return {updated, normalized};
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        const bool has_grad_normalized = grad_outputs[1].defined();
        auto grad_updated = grad_outputs[0].defined()
            ? grad_outputs[0].contiguous()
            : at::zeros_like(saved[0]);
        auto grad_normalized = has_grad_normalized
            ? grad_outputs[1].contiguous()
            : at::zeros_like(saved[0]);
        auto gradients = residual_layer_norm_backward_hip(
            grad_updated,
            grad_normalized,
            saved[0],
            saved[1],
            saved[2],
            saved[3],
            saved[4],
            ctx->saved_data["dropout_p"].toDouble(),
            ctx->saved_data["training"].toBool(),
            ctx->saved_data["branch_is_half"].toBool(),
            ctx->saved_data["threads"].toInt(),
            ctx->saved_data["block_rows"].toInt(),
            ctx->saved_data["parameter_threads"].toInt(),
            ctx->saved_data["parameter_groups"].toInt(),
            has_grad_normalized);
        return {
            gradients[0],
            gradients[1],
            has_grad_normalized ? gradients[2] : at::Tensor(),
            has_grad_normalized ? gradients[3] : at::Tensor(),
            at::Tensor(),
            at::Tensor(),
            at::Tensor(),
            at::Tensor(),
            at::Tensor(),
            at::Tensor(),
            at::Tensor(),
        };
    }
};


torch::autograd::variable_list residual_layer_norm_autograd(
    const at::Tensor& x,
    const at::Tensor& branch,
    const at::Tensor& weight,
    const at::Tensor& bias,
    double dropout_p,
    double eps,
    bool training,
    int64_t threads,
    int64_t block_rows,
    int64_t parameter_threads,
    int64_t parameter_groups) {
    return ResidualLayerNormFunction::apply(
        x,
        branch,
        weight,
        bias,
        dropout_p,
        eps,
        training,
        threads,
        block_rows,
        parameter_threads,
        parameter_groups);
}


}  // namespace


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "residual_layer_norm",
        &residual_layer_norm_autograd);
    module.def(
        "residual_layer_norm_forward",
        &residual_layer_norm_forward_hip);
    module.def(
        "residual_layer_norm_backward",
        &residual_layer_norm_backward_hip);
}
