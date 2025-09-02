# NPU에서 LLM을 효과적으로 활용하기 위한 최적화 방안

## NPU에서 LLM을 구동시키기 위해 무엇이 필요할까?

모바일 기기 등 엣지 디바이스에서 LLM을 구동하기 위한 핵심 하드웨어는 NPU(Neural Processing Unit)입니다. NPU는 저전력으로 빠른 추론을 가능하게 하지만, 데스크톱 GPU와는 다른 제약사항을 가집니다. 이 글에서는 퀄컴의 모바일 NPU(HTP)를 중심으로, LLM을 NPU에 최적화할 때 마주하는 주요 문제점과 해결 방안을 정리합니다.

***

## 문제점 1: Integer 연산 전용 하드웨어와 LLM의 민감도

모바일 NPU는 칩 면적과 전력 효율을 위해 연산 정밀도(precision) 지원을 최소화하며, **주로 8비트 정수(Integer) 연산만을 지원**하는 경우가 많습니다. 설령 FP16을 지원하더라도 매우 느릴 수 있습니다. 따라서 NPU의 성능을 최대로 활용하려면 모델 전체를 정수 연산으로 변환하는 **양자화(Quantization)가 필수적**입니다.

> <img width="1023" alt="Qualcomm AI Engine Direct SDK documentation" src="https://github.com/user-attachments/assets/31884f2e-289e-48e8-805e-d2a7e7980b9d" />
>
> *Qualcomm AI Engine Direct SDK는 HTP(NPU)를 활용하려면 양자화가 필수적이라고 명시합니다.*

### CV 모델의 성공 사례: Fixed-Point 연산
컴퓨터 비전(CV) 모델에서는 이미 활발한 연구를 통해 모든 연산을 정수로 처리하는 기법이 자리 잡았습니다. 핵심은 양자화를 통해 연산을 정수로 대체하고, 스케일(scale) 값을 곱하는 실수 연산마저 비트 시프트(bit shift)와 정수 곱셈으로 구현하여 **모든 실수 연산을 제거**하는 것입니다(Fixed-Point Arithmetic). 퀄컴의 AIMET과 같은 툴은 이러한 양자화를 효과적으로 지원합니다.

### LLM 적용의 어려움
하지만 이 방식을 LLM에 그대로 적용하기는 어렵습니다. LLM은 CV 모델과 다른 특성을 가지기 때문입니다.

1.  **활성화(Activation)의 Outlier 문제**: LLM의 활성화 값에는 드물게 매우 큰 값(outlier)이 나타나는 경향이 있습니다. 이 때문에 전체 텐서에 단 하나의 스케일 값을 적용하는 **Per-Tensor 양자화**를 사용하면 대부분의 값이 표현 범위를 제대로 활용하지 못해 정보 손실이 극심해집니다.
2.  **특정 연산자의 높은 민감도**: **Softmax**나 **Attention의 BMM(Batch Matrix Multiplication)**과 같은 특정 연산자들은 양자화 오차에 매우 민감하여, 정수로 변환 시 모델의 정확도가 크게 하락합니다.

실제로 다수의 연구에서는 BMM과 같은 민감한 연산자는 NPU에서 처리하지 않고, GPU/CPU를 활용해 **FP16으로 연산하는 혼합 정밀도(Mixed-Precision) 전략**을 사용합니다. 이는 NPU 최적화 연구자들도 동의하는 현실적인 접근법입니다.

***

## 문제점 2: 정적인(Static) 그래프만 지원하는 NPU
NPU는 추론 성능을 극대화하기 위해 계산 그래프를 미리 컴파일하여 고정된 형태로 사용합니다. 하지만 LLM의 추론 과정, 특히 KV Cache를 사용하는 생성(generation) 단계는 동적인 측면이 있어, 이를 정적 그래프로 변환하는 과정에 추가적인 최적화 기법이 필요합니다.

***

## 문제점 3: ONNX 중심의 레거시 플랫폼
다수의 NPU 제조사들은 오랫동안 표준으로 사용되어 온 **ONNX**를 핵심 입력 플랫폼으로 지원해왔습니다. 이는 하드웨어 제조사들이 ONNX에 맞춰 드라이버와 컴파일러를 최적화해 온 깊은 레거시를 가지고 있음을 의미합니다.

하지만 ONNX는 PyTorch와 달리 개발자가 그래프를 직접 제어하거나 편집하기 어렵다는 단점이 있습니다. 이 문제를 해결하기 위한 접근법은 다음과 같습니다.

* **ONNX Runtime Quantization Tool**: ONNX 모델을 직접 양자화하는 툴킷을 사용합니다. (AMD에서 활용)
* **AIMET**: ONNX 모델과 양자화 파라미터를 별도의 파일로 분리하여 유연성을 제공합니다. (퀄컴에서 활용)

### 대안: ExecuTorch (EXIR)
최근에는 PyTorch에서 직접 출시한 **ExecuTorch(EXIR)**가 새로운 대안으로 떠오르고 있습니다. EXIR은 PyTorch 네이티브 포맷이므로 다음과 같은 장점이 있습니다.
* PyTorch 모델을 ONNX로 변환할 필요가 없음
* 하드웨어 특성을 반영한 그래프(Hardware-Aware Graph) 생성이 용이
* 그래프 편집 및 Fake Quantization 적용이 상대적으로 쉬움

***

## 문제점 4: 제한적인 연산자(Operator) 지원
NPU는 모든 종류의 연산자를 하드웨어로 지원하지 않습니다. 이는 모든 엣지 디바이스가 가진 공통적인 문제입니다.

이 문제에 대한 표준적인 해결책은 **폴백(Fallback)**입니다. ONNX Runtime, ExecuTorch와 같은 크로스-플랫폼은 NPU가 지원하지 않는 연산자를 만나면, 해당 연산만 **CPU에서 수행하도록 자동으로 전환**하는 기능을 제공합니다. 이를 통해 모델 전체의 호환성을 보장할 수 있습니다.



# Optimizatig_LLM_To_Leverage_NPU

# NPU에서 LLM을 구동시키기 위해 무엇이 필요할까??

# Insight From Qualcomn mobile NPU (HTP)
문제점 1. NPU가 integer 연산만을 지원한다. 
Mobile npu는 크기가 작다. 지원하는 연산자가 다양한 precision을 지원하지 않는다. 많은 precision을 지원하려면 그만큼의 개발과 많은 칩면적을 차지하게 된다.


<img width="1023" height="218" alt="image" src="https://github.com/user-attachments/assets/31884f2e-289e-48e8-805e-d2a7e7980b9d" />
<AI Engine Direct SDK documentation에서 발췌해온 설명, HTP(NPU)을 활용하려면 Quantization이 필수적이라고 말한다. >

또한, FP16을 지원하더라도 매우 느릴 수 있다. 




그럼integer만으로 AI을 구동시키는게 가능한가?? 

[paper integer arithmatic] 
이 논문은 CV model을 integer로만 구동시키는 방법을 소개한다. 

핵심은 quantization을 통해서 연산을 integer로 하고 scale 곱하는 연산 마저 integer와 bit shift로 구현하여 float 연산을 없앨 수 있다. (fixed point arthimatic method) 

아래의 그림은 AI qualcomn direct engine의 실제 Quant Add kernel 구현 방식이다. 보면 scale 을 float 으로 받지만 실제로는 bit shift 와 integer 연산으로만 바꾼다. 

결과적으로 모든 연산자를 quantization 해야한다. 그리고 이것은 CV model에서 어느정도 성능을 가지고 수행할 수 있다. 

그리고 이는 AIMET이라는 tool로써 실현 가능하다.
  [한정된 quantization과 integer arthmatic only 방식을 그대로 LLM에 적용하기 힘든 이유]

하지만 NPU가 결국 정해진 granularity의 quantization만 지원하고 (activation: per-tensor,  weight: per-channels, per-tensor) 모든 연산자를 quantization 해야한다면 LLM을 구동시키는 것에는 문제가 있다. 
  1. LLM activation은 outliar로 인해 per tensor quantization이 매우 어렵다.
  2. 특정 연산자는 매우 senstive 하다.   -> softmax , Attention의 BMM 연산.


이러한 문제를 다른 연구자들은 어떻게 해결했는지 paper을 찾아보고, 연구자들에게 여쭤보았다. 
실제로 대다수 paper 에서는 아예 fp16 으로 BMM 연산을 했다.  
또한 내가 여쭤보니 이쪽 관련 연구자도 몇몇 연산자는 fp 16으로(npu 안씀) 하고,  최근 mobile npu는 fp16도 지원을 한다고 이야기 해줬다. 











<내가 관련 NPU 최적화 연구를 하시는 분꼐 직접 매일로 여쭤보았다.,>









-> ineger arithmatic quantization (paper)
실제 integer 연산만으로 구동이 가능하다. 하지만 아래의 논문들은 전부  CV 기준에서 측정된 방식. 
중요한 것은 LLM은 마찬가지 방식으로 quantization 했을 떄 성능이 너무 떨어진다. 

-> paper like integer arithmatic quantization for LLM 이런 논문이 있을까??   현제로써는 가장 근사한 방식이 

문제점 2. Float 16,32 을 지원해도 여전히 있는  latency, power consumtion problem.


문제점 2. NPU가 static graph 만 지원한다. 
이 문제를 어떻게 해결 했을 까?? 


From paper ~~~ 




문제점 3. Input Platform ONNX만을 지원한다. 
현제 On-Device AI platform for mutilple device 은 정통적 강자 ONNX 그리고 2024년에 등장한 Executorch (EXIR) 이 있다. 

이부분은 NPU 제조사가 각각의 platform과 협업해서 해결해야 할 부분이다. 

ONNX 역사가 깊음.
여러 edge을 위한 hardware제조사들은 이미 onnx에 최적화 시켜왔다. 즉 레거시를 갖고 있다. 
과거의 npu에 최적화 하기 위해서 결국 ONNX 모델을 quantization 하든 fusion을 하든 folding을하든 해야한다. 
문제점은 ONNX는 pytorch 랑 다르게 개발자가 쉽게 컨트롤 하기 어렵다는 것이다. 

해결법 
ONNXruntime qunatization tool  : 실제로 amd 가 쓴방식
AIMET : onnx모델과 quantizaiton parameter을 분리해서 출력해줌 quanlcomn ai engine direct의 사용방식. 



EXIR은 애초에 pytorch에서 나온것이기에 hardware aware graph 을 생성하게 tutorial등이 이미 만들어져있다. 즉 그래프 편집 및 fake quantization 그래프 생성이 상대적으로 쉬우며, 많은 연구가 pytorch에서 시작되므로 onnx로 converting이 필요없음. 

ONNX랑 다르게 이미 표준화된 최적화 방식이 존재.     
여전히 개발중.


문제점 4. 연산자의 부족
이부분은 공통적으로 해결하는 방식이 같다. Cross platform을 써야하는 이유이다. 
ONNX, Executorch 와 같은 platcfrom 의 해결 방법은 fallback 이나 , NPU가 안되면 CPU로 넘기는 방식이다. 


# AMD desktop CPU 와 visit AI 에서 활용한 ONNX 그래프 편집. 






