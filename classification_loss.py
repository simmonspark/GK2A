import torch
import torch.nn as nn
'''
https://www.kma.go.kr/kma/biz/forecast05.jsp

약한 비 : 1이상  ~ 3미만  mm/h
비 : 3이상 ~ 15미만  mm/h
강한 : 15이상 ~ 30미만  mm/h
매우 강한 비 : 30이상  mm/h

비의 종류는 4개입니다. label = mask1(class1) mask2(class2) mask3(class3) mask4(class4)

질문 1.  모델의 출력이 달라져야 하는지... classification이면 batch_size, width, height, depth(class 갯수) 
질문 2. 그럼 각 채널별로 softmax 해야함? 아니면 레이블 인코딩 해야하는지 
질문 3. 아니면 모델의 강우량 출력에 그냥 마스크를 씌워서 각 각 구하는 방식으로 하는지... 

'''