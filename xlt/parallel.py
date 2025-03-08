import jax
from jax.experimental.mesh_utils import create_device_mesh as jax_create_device_mesh
from jax.sharding import Mesh, PartitionSpec
from transformers import FlaxLlamaForCausalLM, AutoTokenizer
import jax.numpy as jnp
import re
import pytest


def create_device_mesh(
    shape: tuple[int, ...], *, axes: tuple[str, ...], devices: list[jax.Device] = jax.devices()
) -> Mesh:
    return Mesh(jax_create_device_mesh(shape, devices=devices), axes)


def shard_params(params):
    def _lookup_partition_spec(param_path):
        partition_rules = [
            # Attention blocks: partition heads across model dimension
            (".*attention.*q_proj.*", PartitionSpec("model", None)),
            (".*attention.*k_proj.*", PartitionSpec("model", None)),
            (".*attention.*v_proj.*", PartitionSpec("model", None)),
            (".*attention.*o_proj.*", PartitionSpec(None, "model")),
            # MLP blocks: partition hidden dim across model dimension
            (".*mlp.*gate_proj.*", PartitionSpec("model", None)),
            (".*mlp.*up_proj.*", PartitionSpec("model", None)),
            (".*mlp.*down_proj.*", PartitionSpec(None, "model")),
        ]
        path_str = "/".join(str(x) for x in param_path)
        for pattern, spec in partition_rules:
            if re.match(pattern, path_str):
                return spec
        return PartitionSpec()  # Default to replication

    return jax.tree_util.tree_map_with_path(
        lambda param_path, param: jax.lax.with_sharding_constraint(
            param, _lookup_partition_spec(param_path)
        ),
        params,
    )


def no_gpu_tpu_available():
    return not any(d.platform in ("gpu", "tpu") for d in jax.devices())


@pytest.mark.skipif(no_gpu_tpu_available(), reason="Test requires GPU or TPU")
def test_shard_params():
    MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model = FlaxLlamaForCausalLM.from_pretrained(MODEL, dtype=jnp.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    inputs = tokenizer(
        [
            "### Human: What is the color of sea?{prompt}### Assistant:",
            "### Human: What is the color of sky?{prompt}### Assistant:",
        ],
        return_tensors="np",
    )

    if (n := jax.device_count()) >= 2:
        mesh = create_device_mesh((2, n // 2), axes=("data", "model"))  # 2-way DP
    else:
        mesh = create_device_mesh((1, 1), axes=("data", "model"))  # 1-way DP

    with mesh:
        model.params = shard_params(model.params)
        input_ids = jax.lax.with_sharding_constraint(inputs.input_ids, PartitionSpec("data", None))
        attention_mask = jax.lax.with_sharding_constraint(
            inputs.attention_mask, PartitionSpec("data", None)
        )

    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=8, params=model.params
    )
    generated_text = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
    assert generated_text == [
        "### Human: What is the color of sea?{prompt}### Assistant: The color of sea is blue. ###",
        "### Human: What is the color of sky?{prompt}### Assistant: The color of sky is blue. ###",
    ], "generated_text"
