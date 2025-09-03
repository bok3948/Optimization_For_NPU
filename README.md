# NPU에서 LLM을 효과적으로 활용하기 위한 최적화 방안

## NPU에서 LLM을 구동시키기 위해 무엇이 필요할까?

모바일 기기 등 엣지 디바이스에서 LLM을 구동하기 위한 핵심 하드웨어는 NPU입니다. NPU는 저전력으로 빠른 추론을 가능하게 하지만, 데스크톱 GPU와는 다른 제약사항을 가집니다. 이 글에서는 퀄컴의 모바일 NPU(HTP)를 중심으로, LLM을 NPU에 최적화할 때 마주하는 주요 문제점과 해결 방안을 정리합니다.

***

## 문제점 1: Integer 연산 전용 하드웨어와 LLM의 민감도

모바일 NPU는 칩 면적과 전력 효율을 위해 precision 지원을 최소화하며, **주로 8비트 Integer 연산만을 지원**하는 경우가 많습니다. 설령 FP16을 지원하더라도 매우 느릴 수 있습니다. 따라서 NPU의 성능을 최대로 활용하려면 모델 전체를 정수 연산으로 변환하는 **Quantization이 필수적**입니다.

> <img width="1023" height="218" alt="image" src="https://github.com/user-attachments/assets/31884f2e-289e-48e8-805e-d2a7e7980b9d" />
>
> *Qualcomm AI Engine Direct SDK는 HTP(NPU)를 활용하려면 Quantization이 필수적이라고 명시합니다.*

><img width="759" height="341" alt="image" src="https://github.com/user-attachments/assets/8d6c7ab0-f202-46dc-a327-c29c920e99c6" />

> *위 표에서는 mobile NPU가 fp16과 같은 연산에서 얼마나 비효율적인지 보여줍니다.*

### CV 모델의 성공 사례: Integer-only Arithmetic
컴퓨터 비전(CV) 모델에서는 이미 활발한 연구를 통해 모든 연산을 정수로 처리하는 기법이 자리 잡았습니다. 핵심은 Quantization을 통해 연산을 정수로 대체하고, scale 값을 곱하는 실수 연산마저 bit shift와 정수 곱셈으로 구현하여 **모든 실수 연산을 제거**하는 것입니다(Fixed-Point Arithmetic). 퀄컴의 AIMET과 같은 툴은 이러한 Quantization을 효과적으로 지원합니다.

> [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)

### 실제 QNN 커널 코드
```python
def qdense_compute(
    tensor_a,
    tensor_b,
    zero_a,
    scale_a,
    zero_b,
    scale_b,
    zero_out=None,
    scale_out=None,
    bias=None,
    q_dtype=None,
):
    """Hexagon's implementation of a sliced dense operator in Topi.
    Uses matmul.
    """
    if bias is not None:
        assert len(bias.shape) == 1
    if q_dtype is None:
        q_dtype = tensor_a.dtype

    batch, in_dim = tensor_a.shape
    out_dim, red_dim = tensor_b.shape

    assert int(in_dim) == int(red_dim)

    k = te.reduce_axis((0, in_dim), name="k")
    compute_lambda = lambda n, m: te.sum(
        scale_a
        * (tensor_a[n, k] - zero_a)
        * scale_b
        * (tensor_b[k, m] - zero_b),
        axis=k,
    )
    compute_name = "qmatmul_sliced"

    out = te.compute(
        (batch, out_dim),
        compute_lambda,
        name=compute_name,
        attrs={"layout_free_placeholders": [tensor_b]},
    )

    if bias is not None:
        out = te.compute(
            (batch, out_dim),
            lambda i, j: out[i, j] + bias[j],
            tag=tag.BROADCAST,
            name="bias",
        )

    if scale_out is not None:
        out = te.compute(
            (batch, out_dim),
            lambda *i: (out[i] / scale_out + zero_out).astype(q_dtype),
            name="requantize",
        )

    return out
```

> 이 코드를 보면 현재 최신 버전 QNN은 scale 자체는 float 연산을 하는 것으로 보입니다.


<img width="423" height="212" alt="image" src="https://github.com/user-attachments/assets/f0c067e2-8fed-4dd4-b36f-8f7e59c4cb09" />


> *가장 최신 스냅드래곤 NPU는 float16 연산을 부분적으로 지원하는 것으로 보입니다. (하지만 느릴것으로 추정됨.)

## AIMET을 통한 모든 연산자의 quantization의 문제점. 

LLM에 그대로 적용하기는 어렵습니다. LLM은 CV 모델과 다른 특성을 가지기 때문입니다.

원인 1: CV에 맞춰진 Quantization Schema: LLM의 Activation 값에는 드물게 매우 큰 값(outlier)이 나타나는 경향이 있습니다. 이 때문에 전체 텐서에 단 하나의 스케일 값을 적용하는 Per-Tensor Quantization을 사용하면 대부분의 값이 표현 범위를 제대로 활용하지 못해 정보 손실이 극심해집니다. AIMET의 경우 activation은 per-tensor, weight는 per-channel, per-tensor만 지원해서 많은 SOTA 논문에서 활용하는 per-group quantization, per-token qunatization이 불가능합니다.

원인 2: 특정 연산자의 높은 민감도: Normalization, Softmax, Non-linear function, Attention의 BMM과 같은 특정 연산자들은 Quantization 오차에 매우 민감하여, 정수로 변환 시 모델의 정확도가 크게 하락합니다.

## 해결법

<img width="441" height="130" alt="image" src="https://github.com/user-attachments/assets/55d94993-f3e5-4532-8128-3159b60c3bda" />

> *실제로 다수의 연구에서는 BMM과 같은 민감한 연산자는 NPU에서 처리하지 않고, GPU/CPU를 활용해 FP16 또는 INT16으로 연산하는 Mixed-Precision 전략을 사용합니다.

<img width="1348" height="377" alt="image" src="https://github.com/user-attachments/assets/98109af5-7c20-4bc3-84f7-c383dc9e92ac" />

> * 제가 실제로 MobileQuant 저자에게서 받은 내용입니다. 옛날 NPU라면 CPU와 GPU을 활용하라고 조언해주셨고, error가 심한 layer는 fp16을 활용하는 방식을 활용했다고 합니다. 

mobilequant 코드에서는 FP16 사용
``` python
elif isinstance(module, QMatMul):
    if 'qk_bmm' in name and args.use_16bit_softmax_input:
        model._modules[name].output_quantizer.qcfg.bitwidth = 16
    if 'pv_bmm' in name and args.use_16bit_softmax_output:
        model._modules[name].input_quantizer.qcfg.bitwidth = 16
```

또한, non-linear operator들은 exponential 특징 때문에 FP16으로 처리했습니다. 작은 입력 변화에도 출력값이 크게 변하는 경우, linear quantization은 심각한 오차를 유발할 수 있습니다. 예를 들어 0과 0.1에 대한 non-linear 함수 출력이 각각 0과 100일 때, 이를 양자화하면 모두 0으로 출력될 수 있습니다.

따라서 MobileQuant에서는 non-linear operator와 attention 연산을 양자화 대상에서 제외했습니다.


## 문제점 2: NPU의 static graph 문제
NPU는 추론 성능을 극대화하기 위해 계산 그래프를 미리 컴파일하여 고정된 형태로 사용합니다. 하지만 LLM의 추론 과정, 특히 KV Cache를 사용하는 생성 단계는 동적인 측면이 있습니다.

Prefill 단계에서는 많은 토큰이 한 번에 입력되고, decode 단계에서는 하나의 토큰만 입력되는 등 입력 크기가 계속 변하는 것이 문제입니다. MobileQuant 저자는 이 문제를 Prefill과 Decode를 위한 두 개의 분리된 그래프를 생성하는 방식으로 해결했습니다.

## 문제점 3: ONNX 그래프 편집의 어려움.
다수의 NPU 제조사들은 오랫동안 표준으로 사용되어 온 ONNX를 핵심 입력 플랫폼으로 지원해왔습니다. 이는 하드웨어 제조사들이 ONNX에 맞춰 드라이버와 컴파일러를 최적화해 온 깊은 레거시를 가지고 있음을 의미합니다.

하지만 ONNX는 PyTorch와 달리 개발자가 그래프를 직접 제어하거나 편집하기 어렵다는 단점이 있습니다. 이 문제를 해결하기 위한 접근법은 다음과 같습니다.

ONNX Runtime Quantization Tool: ONNX 모델을 직접 양자화하는 툴킷을 사용합니다. (AMD에서 활용)

AIMET: ONNX 모델과 양자화 파라미터를 별도의 파일로 분리하여 유연성을 제공합니다. (퀄컴에서 활용)

대안: ExecuTorch (EXIR)
최근에는 PyTorch에서 직접 출시한 **ExecuTorch(EXIR)**가 새로운 대안으로 떠오르고 있습니다. EXIR은 PyTorch 네이티브 포맷이므로 다음과 같은 장점이 있습니다.

PyTorch 모델을 ONNX로 변환할 필요가 없음

하드웨어 특성을 반영한 그래프(Hardware-Aware Graph) 생성이 용이

그래프 편집 및 Fake Quantization 적용이 상대적으로 쉬움

대안: ONNX 편집 (AMD)

``` python
class VitisDPUQDQQuantizer(QDQQuantizer):
    def __init__(
        self,
        model,
        per_channel,
        reduce_range,
        mode,
        static,
        weight_qType,
        input_qType,
        tensors_range,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize,
        calibrate_method,
        need_layer_fusing=False,
        extra_options=None,
    ):
        QDQQuantizer.__init__(
            self,
            model,
            per_channel,
            reduce_range,
            mode,
            static,
            weight_qType,
            input_qType,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            extra_options,
        )
        self.tensors_to_quantize = {}
        self.calibrate_method = calibrate_method
        self.need_layer_fusing = need_layer_fusing

        if per_channel:
            logging.error("per_channel is not supported in PowerOfTwoMethod calibrate_method.")

        # In PowerOfTwoMethod calibrate_method, QDQ should always appear as a pair.
        # Therefore, we need to add qdq pair to weight.
        if "AddQDQPairToWeight" in self.extra_options and not self.extra_options["AddQDQPairToWeight"]:
            logging.error("AddQDQPairToWeight should be True in PowerOfTwoMethod calibrate_method.")
        self.add_qdq_pair_to_weight = True

        # In PowerOfTwoMethod calibrate_method, QDQ should always set WeightSymmetric as True.
        if "WeightSymmetric" in self.extra_options and not self.extra_options["WeightSymmetric"]:
            logging.error("WeightSymmetric should be True in PowerOfTwoMethod calibrate_method.")
        self.is_weight_symmetric = True

        # In PowerOfTwoMethod calibrate_method, QDQ should always always set ActivationSymmetric as True.
        if "ActivationSymmetric" in self.extra_options and not self.extra_options["ActivationSymmetric"]:
            logging.error("ActivationSymmetric should be True in PowerOfTwoMethod calibrate_method.")
        self.is_activation_symmetric = True

    def vitis_quantize_initializer(self, weight, bit_width=8, keep_float_weight=False):

        # Find if this input is already quantized
        if weight.name in self.quantized_value_map:
            quantized_value = self.quantized_value_map[weight.name]
            return (
                quantized_value.q_name,
                quantized_value.zp_name,
                quantized_value.scale_name,
            )

        q_weight_name = weight.name + "_quantized"
        zp_name = weight.name + "_zero_point"
        scale_name = weight.name + "_scale"

        # Update packed weight, zero point, and scale initializers
        weight_data = tensor_proto_to_array(weight)
        _, _, zero_point, scale, q_weight_data = vitis_quantize_data(
            weight_data.flatten(), bit_width, method=self.calibrate_method
        )
        scale_initializer = onnx.helper.make_tensor(scale_name, onnx_proto.TensorProto.FLOAT, [], [scale])
        zero_initializer = onnx.helper.make_tensor(zp_name, onnx_proto.TensorProto.INT8, [], [zero_point])
        self.model.initializer().extend([scale_initializer, zero_initializer])

        # Log entry for this quantized weight
        quantized_value = QuantizedValue(
            weight.name,
            q_weight_name,
            scale_name,
            zp_name,
            QuantizedValueType.Initializer,
            None,
        )
        self.quantized_value_map[weight.name] = quantized_value

        return q_weight_name, zp_name, scale_name

    def quantize_model(self):

        self.tensor_info = {}
        model = self.model.model
        annotate_output_name_list = get_annotate_output_name(model)
        relu_to_conv_output = get_relu_name(model, annotate_output_name_list)

        for node in self.model.nodes():
            if self.should_quantize_node(node):
                op_quantizer = CreateQDQQuantizer(self, node)
                op_quantizer.quantize()

                if self.dedicated_qdq_pair:
                    for tensor_name in node.input:
                        if tensor_name not in self.tensor_to_its_receiving_nodes:
                            self.tensor_to_its_receiving_nodes[tensor_name] = []
                        self.tensor_to_its_receiving_nodes[tensor_name].append(node)

        self._quantize_normal_tensors()

        self._quantize_sharing_param_tensors()
        dq_nodes_to_remove, q_nodes_to_remove = get_qdq_to_remove(
            model, relu_to_conv_output)
        convert_relu_input_to_annotate_output(model, relu_to_conv_output)
        if self.need_layer_fusing:
            model = remove_nodes(model, dq_nodes_to_remove)
            model = remove_nodes(model, q_nodes_to_remove)
        self._quantize_refine()
        self.remove_nodes()
        if not self.add_qdq_pair_to_weight:
            self.model.clean_initializers()

        model.producer_name = __producer__
        model.producer_version = __version__

        return model

    def _add_qdq_pair_for_initializer(self, weight_proto, tensor_type, axis=None):
        weight_name = weight_proto.name
        q_weight_name, zp_name, scale_name = self.vitis_quantize_initializer(
            weight_proto, self.weight_qType, keep_float_weight=True
        )

        weight_dequant_output = add_dequant_output_suffix(weight_name)
        self.model.replace_input_of_all_nodes(weight_name, weight_dequant_output)
        if self.add_qdq_pair_to_weight:
            weight_quant_output = add_quant_output_suffix(weight_name)

            self._create_qdq_nodes(
                weight_name,
                weight_quant_output,
                add_quant_suffix(weight_name),
                weight_quant_output,
                weight_dequant_output,
                add_dequant_suffix(weight_name),
                scale_name,
                zp_name,
                axis,
            )
        else:
            dequant_node = onnx.helper.make_node(
                DEQUANT_OP_NAME,
                [q_weight_name, scale_name, zp_name],
                [weight_dequant_output],
                add_dequant_suffix(weight_name),
                axis=axis,
            )

            self.model.add_node(dequant_node)

    def quantize_bias_tensor(self, bias_name, input_name, weight_name, beta=1.0):
        weight = find_by_name(bias_name, self.model.initializer())
        if weight is not None:
            if weight.data_type == onnx_proto.TensorProto.FLOAT:
                # Use int8 quantization for bias as well as weights.
                self.tensors_to_quantize[bias_name] = QDQTensorQuantInfo()
        else:
            logging.warning("Expected {} to be a weight".format(bias_name))

    def _quantize_refine(self):
        self.model = adjust_quantize_info(
            self.model,
            adjust_vitis_sigmoid=True,
            adjust_shift_cut=True,
            adjust_shift_bias=True,
            adjust_shift_read=True,
            adjust_shift_write=True,
            align_concat=True,
            align_pool=True,
        )

class VitisAIQDQQuantizer(VitisAIONNXQuantizer):
    def __init__(
        self,
        model,
        per_channel,
        reduce_range,
        mode,
        static,
        weight_qType,
        activation_qType,
        tensors_range,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize,
        extra_options=None,
    ):
        ONNXQuantizer.__init__(
            self,
            model,
            per_channel,
            reduce_range,
            mode,
            static,
            weight_qType,
            activation_qType,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            extra_options,
        )
        self.tensors_to_quantize = {}
        self.bias_to_quantize = []

        self.nodes_to_remove = []

        # Specific op types to exclude qdq quantization for their outputs.
        # In TRT, it's not recommended to quantize outputs for weighted ops such as Conv, Matmul, Gemm
        # because those ops may be followed by nodes that require high resolution inputs.
        # Adding QDQ for those ops' output may end up with worse accuracy.
        # So, we don't recommend to add QDQ to node's output under such condition.
        self.op_types_to_exclude_output_quantization = (
            []
            if "OpTypesToExcludeOutputQuantization" not in extra_options
            else extra_options["OpTypesToExcludeOutputQuantization"]
        )

        # We do quantization on Dequantizelinear's input to remove Quantizelinear for weight as an optimization.
        # In some cases, for example QDQ BERT model for TensorRT, QDQ should always appear as a pair.
        # Therefore, we need to disable this optimization and add qdq pair to weight.
        self.add_qdq_pair_to_weight = (
            False if "AddQDQPairToWeight" not in extra_options else extra_options["AddQDQPairToWeight"]
        )

        # The default behavior is that multiple nodes can share a QDQ pair as their inputs.
        # In TRT, QDQ pair can’t be shared between nodes, so it will create dedicated QDQ pairs for each node.
        self.dedicated_qdq_pair = (
            False if "DedicatedQDQPair" not in extra_options else extra_options["DedicatedQDQPair"]
        )
        if self.dedicated_qdq_pair:
            self.tensor_to_its_receiving_nodes = {}

        # Let user set channel axis for specific op type and it's effective only when per channel quantization is supported and per_channel is True.
        self.qdq_op_type_per_channel_support_to_axis = (
            {}
            if "QDQOpTypePerChannelSupportToAxis" not in extra_options
            else extra_options["QDQOpTypePerChannelSupportToAxis"]
        )
```
> * 이 코드는 실제로 AMD에서 AMD에서 NPU에 알맞은 형태로 ONNX 모델을 Quantization하는 방식이다. 






## 문제점 4: 제한적 연산자 지원
NPU는 모든 종류의 연산자를 하드웨어로 지원하지 않습니다. 이는 모든 엣지 디바이스가 가진 공통적인 문제입니다.

이 문제에 대한 표준적인 해결책은 Fallback입니다. ONNX Runtime, ExecuTorch와 같은 크로스-플랫폼은 NPU가 지원하지 않는 연산자를 만나면, 해당 연산만 CPU에서 수행하도록 자동으로 전환하는 기능을 제공합니다. 이를 통해 모델 전체의 호환성을 보장할 수 있습니다.
