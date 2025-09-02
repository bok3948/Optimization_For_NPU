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






