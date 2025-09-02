

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

    k = te.reduce_제
NPU는 추론 성능을 극대화하기 위해 계산 그래프를 미리 컴파일하여 고정된 형태로 사용합니다. 하지만 LLM의 추론 과정, 특히 KV Cache를 사용하는 생성 단계는 동적인 측면이 있습니다.

Prefill 단계에서는 많은 토큰이 한 번에 입력되고, decode 단계에서는 하나의 토큰만 입력되는 등 입력 크기가 계속 변하는 것이 문제입니다. MobileQuant 저자는 이 문제를 Prefill과 Decode를 위한 두 개의 분리된 그래프를 생성하는 방식으로 해결했습니다.

##문제점 3: ONNX 중심의 플랫폼과 편집의 어려움
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
