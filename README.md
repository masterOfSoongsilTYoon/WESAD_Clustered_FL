
### Stress Affect Detection At Wearable Devices Via Clustered Federated Learning
____

## Abstract(will be Updated)
- 웨어러블 디바이스에서는 사용자의 다양한 메타데이터를 수집할 수 있다. 그러나 이런 개인정보를 함유하고 있는 데이터를 수집하는 것은 사용자에게 개인정보침해 위협을 야기한다. 때문에 본 저자는 개인정보보호를 통한 웨어러블 디바이스 데이터활용방안으로 연합학습을 채택하였다. 다만 기존 연합학습에서도 해결해야할 문제점들이 있다. 우리는 그중에서도 데이터이질성(Data Heterogeneity) 보완을 위해 군집화(Clustering) 메소드를 활용하였다. 이를 통해 WESAD(Werable Stress Affect Detection)데이터에서 피실험자의 데이터 이질성이 존재하는 상황에서 기존 연합학습보다 Accuracy 점수가 00% 향상과 F1score 점수의 00% 향상됨을 보여주었다. 또한 기존의 코사인유사도 기반 클러스터링에서 파라미터중요도가 반영되지 않는다는 문제점을 해결하고자 데이터 수 기반 마하라노비스거리(Number of Samples Mahalanobis Distance) 클러스터링 방법을 제시하였다.

## Introduce WESAD Dataset
WESAD 데이터셋은 Wearable Stress Affect Detaction의 약자로, 특정 실험자들을 대상으로 실험을 통해 탐지한 스트레스 수치 및 메타데이터를 함유 하고 있다. 메타데이터 수집은 손목과 가슴에서 측정한 웨어러블 기기를 통해 수집하였다. 손목에서 측정한 메타데이터의 종류로는 (ACC, BVP, EDA, TEMP) 이고 가슴에서 측정한 메타데이터의 종류로는 (ACC, ECG, EMG, EDA, TEMP, RESP)가 있다[3]. 특정 실험자의 데이터셋을 기준으로 가슴에서 측정한 데이터의 분석을 시도하였다.

- __<그림 1>__
<image src="./Material/ACC Score.png" width= "800pt">
나머지 메타데이터도 <그림1>과 유사한 시계열적인 데이터 특징을 가지고 있다. 여기서 딥러닝을 위한 독립변수를 설정하기위해 종속변수인 스트레스 수치를 가지고 각 변수들의 상관지수를 분석하였다.

- __<표 1>__
<table border='0'>
  <tr>
    <td>독립변수</td>
    <td>상관지수(correlation)</td>
  </tr>
  <tr>
    <th> ACC </th>
    <th>-24.731%</th>
  </tr>
  <tr>
    <td>ECG</td>
    <td>0.006%</td>
  </tr>
  <tr>
    <td>EMG</td>
    <td>0.557%</td>
  </tr>
  <tr>
    <th>EDA</th>
    <th>-31.643%</th>
  </tr>
  <tr>
    <th>Temp</th>
    <th>37.716%</th>
  </tr>
  <tr>
    <td>Resp</td>
    <td>0.233%</td>
  </tr>
</table>

<표 1>을 통해 스트레스 탐지에 주된 유의미한 독립변수 3개 __(ACC, EDA, Temp)__ 를 선정하였다. 3개의 데이터의 시계열 데이터의 특징은 다음 그래프와 같다.
- __<그림 2>__ (ACC, EDA, Temp)
<image src="./Material/output.png" width=800pt>

- __<그림 3>__ Stress 
<image src="./Material/stress.png" width=800p>
<그림 2,3>에서 ACC, EDA의 증가는 스트레스의 감소에 영향을 Temp 수치의 증가는 스트레스의 증가에 영향을 끼치는 양상임을 볼 수 있다.

## Command
__Centralized Learning__
    
    python train.py -e 10 -w ./WESAD -v CentralNet -p ./Models/CentralNet/net.pt
-e: epoch, -w WESAD 폴더(train, valid, test 폴더 함유해야함), -v Model version or Project version, -p Pretrained model path, -s Seed number(default 2024)

__Federated Learning__
  
FedAvg Server
    
    python FedAvgServer.py -v FedAvg -w ./WESAD -r 10 -p ./Models/FedAvg/net.pt
-v Server version, -w WESAD 폴더(train, valid, test 폴더 함유해야함), -r Round number, -p Pretrained model path, -s Seed number(default 2024)

Clustered FedAvg Server
    
    python ClusteredFedAvgServer.py -v FedAvg -w ./WESAD -r 10 -p ./Models/FedAvg/net.pt

Client Command

    python client.py -v FedAvg -w ./WESAD -i 1 -e 3

## Performance Table
 - __Stress vs Non-Stress__ (will be updated)
<table>
  <tr>
    <td>CL or FL</td>
    <td>Accuracy</td>
    <td>F1-score</td>
    
  </tr>
  <tr>
    <td>CL(Centralized Learning)</td>
    <td>76.49%</td>
    <td>43.33%</td>
    
  </tr>
  <tr>
    <td>Fed-Avg</td>
    <td>00.00%</td>
    <td>00.00%</td>
    
  </tr>
  <tr>
    <td>Clustered Fed-Avg</td>
    <td>00.00%</td>
    <td>00.00%</td>
  </tr>
  
</table>