Markdown

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

<details>
<summary><b>QNN Quantization Dense Layer Kernel Code (접기/펼치기)</b></summary>

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
        * (tensor_a[n, k].astype("float32") - zero_a)
        * scale_b
        * (tensor_b[k, m].astype("float32") - zero_b),
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
이 코드를 보면 현재 최신 버전 QNN은 scale 자체는 float 연산을 하는 것으로 보입니다.

</details>



<br>

<img width="427" height="216" alt="image" src="https://github.com/user-attachments/assets/2a83939e-1f44-46b4-b4c4-4bfe010f231c" />

현재 가장 최신의 스냅드래곤 NPU는 float16 연산을 부분적으로 지원하는 것으로 보입니다.

LLM 적용의 어려움
하지만 이 방식을 LLM에 그대로 적용하기는 어렵습니다. LLM은 CV 모델과 다른 특성을 가지기 때문입니다.

CV에 맞춰진 Quantization Schema: LLM의 Activation 값에는 드물게 매우 큰 값(outlier)이 나타나는 경향이 있습니다. 이 때문에 전체 텐서에 단 하나의 스케일 값을 적용하는 Per-Tensor Quantization을 사용하면 대부분의 값이 표현 범위를 제대로 활용하지 못해 정보 손실이 극심해집니다. AIMET의 경우 activation은 per-tensor, weight는 per-channel, per-tensor만 지원해서 많은 SOTA 논문에서 활용하는 per-group quantization, per-token qunatization이 불가능합니다.

특정 연산자의 높은 민감도: Normalization, Softmax, Non-linear function, Attention의 BMM과 같은 특정 연산자들은 Quantization 오차에 매우 민감하여, 정수로 변환 시 모델의 정확도가 크게 하락합니다.

<img width="447" height="127" alt="image" src="https://github.com/user-attachments/assets/d6916c7c-f71f-46d6-812e-b7f24dc88208" />

실제로 다수의 연구에서는 BMM과 같은 민감한 연산자는 NPU에서 처리하지 않고, GPU/CPU를 활용해 FP16 또는 INT16으로 연산하는 Mixed-Precision 전략을 사용합니다.

Python

# mobilequant 코드에서는 FP16 사용
elif isinstance(module, QMatMul):
    if 'qk_bmm' in name and args.use_16bit_softmax_input:
        model._modules[name].output_quantizer.qcfg.bitwidth = 16
    if 'pv_bmm' in name and args.use_16bit_softmax_output:
        model._modules[name].input_quantizer.qcfg.bitwidth = 16
<img width="1361" height="391" alt="image" src="https://github.com/user-attachments/assets/85d6ad3c-0b33-471c-a729-2b8456331005" />

위의 그림은 제가 직접 관련 연구하시는 분께 문의를 드려서 얻은 조언입니다. Older NPU에는 GPU/CPU를 활용해서 성능 저하를 막아야 한다고 하셨습니다. 하지만 최신 NPU는 fp16을 지원을 한다고 이야기 하셨습니다. 어쩌면 아직 Qualcomm AI Direct Engine Docs가 update가 안 된 걸 수도 있을 것 같습니다.

또한 non-linear operator들은 fp16으로 했는데 이는 exponential 특징 때문으로 보입니다. 작은 변화에도 크게 값이 변할 경우에는 linear quantization으로는 크게 오류가 생깁니다. 예를 들어 0, 0.1의 non-linear 함수의 출력이 0, 100이라고 하면 quantization을 해버리면 0, 0이 출력됩니다. 심각한 quantization error가 발생합니다.

따라서 mobilequant 에서는 non-linear operator는 quantization을 안 했습니다. 또한 논문에는 없었지만 코드를 보면 attention도 하지 않았습니다.

위의 표는 실제 여러 논문에서 Attention 연산을 실제로 integer을 통해 하는지 조사한 것입니다.

문제점 2: Static 그래프만 지원하는 NPU
NPU는 추론 성능을 극대화하기 위해 계산 그래프를 미리 컴파일하여 고정된 형태로 사용합니다. 하지만 LLM의 추론 과정, 특히 KV Cache를 사용하는 생성 단계는 동적인 측면이 있습니다.

Prefill 단계에서는 많은 토큰이 들어오고 decode 단계에서는 input token이 한 개인데, 문제는 input의 크기가 다양한 것입니다. MobileQuant 저자는 이 문제를 두 개의 그래프를 생성하는 방식으로 해결했습니다.

문제점 3: ONNX 편집의 어려움
다수의 NPU 제조사들은 오랫동안 표준으로 사용되어 온 ONNX를 핵심 입력 플랫폼으로 지원해왔습니다. 이는 하드웨어 제조사들이 ONNX에 맞춰 드라이버와 컴파일러를 최적화해 온 깊은 레거시를 가지고 있음을 의미합니다.

하지만 ONNX는 PyTorch와 달리 개발자가 그래프를 직접 제어하거나 편집하기 어렵다는 단점이 있습니다. 이 문제를 해결하기 위한 접근법은 다음과 같습니다.

ONNX Runtime Quantization Tool: ONNX 모델을 직접 양자화하는 툴킷을 사용합니다. (AMD에서 활용)

AIMET: ONNX 모델과 양자화 파라미터를 별도의 파일로 분리하여 유연성을 제공합니다. (퀄컴에서 활용)

대안: ExecuTorch (EXIR)
최근에는 PyTorch에서 직접 출시한 **ExecuTorch(EXIR)**가 새로운 대안으로 떠오르고 있습니다. EXIR은 PyTorch 네이티브 포맷이므로 다음과 같은 장점이 있습니다.

PyTorch 모델을 ONNX로 변환할 필요가 없음

하드웨어 특성을 반영한 그래프(Hardware-Aware Graph) 생성이 용이

그래프 편집 및 Fake Quantization 적용이 상대적으로 쉬움

문제점 4: 제한적인 연산자(Operator) 지원
NPU는 모든 종류의 연산자를 하드웨어로 지원하지 않습니다. 이는 모든 엣지 디바이스가 가진 공통적인 문제입니다.

이 문제에 대한 표준적인 해결책은 Fallback입니다. ONNX Runtime, ExecuTorch와 같은 크로스-플랫폼은 NPU가 지원하지 않는 연산자를 만나면, 해당 연산만 CPU에서 수행하도록 자동으로 전환하는 기능을 제공합니다. 이를 통해 모델 전체의 호환성을 보장할 수 있습니다.
