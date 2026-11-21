import torch

w = torch.Tensor([0.2]*5) # 최초 가중치 초기화
print("초기 가중치:".center(10), w)

    #공부한 시간에 따른 합격 불합격 학습
    # 데이터 번호, 공부시간, 실제 정답(합격: 1, 불합격: -1)
datas = torch.Tensor([[1, 1, -1],
                    [2, 2, -1],
                    [3, 3, 1],
                    [4, 4, -1], # 불합격 (특이 케이스)
                    [5, 5, 1]])

"""
    Decision stamp
    이 데이터의 학습데이터인 공부시간이 연속형인 데이터임으로 
    각 구간별 평균으로 데이터들을 나누어보고 지니불순도가 
    가장 낮은 것을 앙상블의 처음으로 사용한다.
"""

#데이터 나누는 기준점찾기
Standards = torch.Tensor([i for i in [(datas[x, 1] + datas[x + 1, 1]) / 2 for x in range(0, 4, 1)]])
print("기준점들: ".center(10), Standards, "\n")



def select_standard(weight : list, datas : list, Standards : list):
    class divided:
        def __init__(self):
            self.standard = 0
            self.passed = []
            self.failed = []
            self.gini = 1
            self.Error = 1
            self.Reliability = 0
            self.correct_index = []
            self.incorrect_index = []
            self.polarity = 1

        def print_data(self):
            print("선택된 기준:".center(10), self.standard)
            print("합격으로 예측:".center(10), self.passed)
            print("불학격으로 예측:".center(10), self.failed)
            print("맞춘 인덱스".center(10), self.correct_index)
            print("틀린 인덱스".center(10), self.incorrect_index)
            print("지니불순도:".center(10), self.gini)
            print("오류률:".center(10), self.Error)
            print("신뢰도:".center(10), self.Reliability,"\n")
    print("weight: ".center(10), weight)
    divided = divided()
    for standard in Standards:
        passed = []
        failed = []
        gini = 0
        Error = 0
        Reliability = 0
        correct_index = []
        incorrect_index = []
        polarity = 1
        # 공부시간이 기준보다 높으면 합격이라고 예측
        for data in datas:
            study_time = data[1]
            if study_time > standard:
                passed.append(data[0])
            else :
                failed.append(data[0])

        #TN: 거짓을 거짓으로 예측, TP: 참을 참으로 예측, FN: 참을 거짓으로 예측, FP 거짓을 참으로 예측
        # FN * w + FP * w = Error
        TN, TP, FN, FP = 0, 0, 0, 0
        for index in passed:
            index = int(index-1)
            if datas[index, 2] == 1:
                TP +=1
                correct_index.append(index+1)
            elif datas[index, 2] == -1:
                FP +=1
                Error += weight[index]
                incorrect_index.append(index+1)

        for index in failed:
            index = int(index-1)
            if datas[index, 2] == -1:
                TN += 1
                correct_index.append(index+1)
            elif datas[index, 2] == 1:
                FN += 1
                Error += weight[index]
                incorrect_index.append(index+1)

        if Error > 0.5:
            Error = 1.0 - Error
            polarity = -1
            correct_index, incorrect_index = incorrect_index, correct_index


        # 왼쪽을 합격이라 보고 오른쪽을 불합격으로 생각
        # TP/(TP+FP) 참으로 예측한것중에 참인것 P(TP|passed), FP/(TP+FP) 참으로 예측한 것 중에 거짓인것 P(FP|passed)
        left_gini = 1 - (TP/(TP+FP))**2 - (FP/(TP+FP))**2

        # TN/(TN+FN) 거짓으로 예측한것중에 거짓인것 P(TN|failed), FN/(TN+FN) 거짓으로 예측한 것 중에 참인것 P(FN|failed)
        right_gini = 1 - (TN/(TN+FN))**2 - (FN/(TN+FN))**2

        #len(passed)/(len(passed)+len(failed)) 총 개수중 합격으로 예측한 것 들의 비율
        passed_rate = len(passed)/(len(passed)+len(failed))

        #len(failed)/(len(passed)+len(failed)) 총 개수중 불합격으로 예측한 것 들의 비율
        failed_rate = len(failed)/(len(passed)+len(failed))

        #평균 지니불순도 계산
        gini = left_gini * passed_rate + right_gini * failed_rate

        if Error < divided.Error:
            divided.standard = standard
            divided.passed = passed
            divided.failed = failed
            divided.gini = gini
            divided.Error = Error
            divided.Reliability = (1/2)*torch.log((1-Error)/Error)
            divided.correct_index = sorted(correct_index)
            divided.incorrect_index = sorted(incorrect_index)
            divided.polarity = polarity

    divided.print_data()
    return divided

def calculate_weight(current_w : list, correct_index : list, incorrect_index : list, Reliability : float):
    next_w = torch.Tensor([0] * len(current_w))
    for index in correct_index:
        next_w[index-1] = current_w[index-1] * torch.exp((-1)*Reliability)
    for index in incorrect_index:
        next_w[index-1] = current_w[index-1] * torch.exp(Reliability)

    next_w /= sum(next_w)
    return next_w

def AdaBoost(weights : list, datas : list, Standards : list, n : int, models = None):
    if models == None:
        models = []

    result = select_standard(weights,datas,Standards)
    models.append(result)

    next_w = calculate_weight(weights,result.correct_index,result.incorrect_index,result.Reliability)
    if n > 0:
        n -= 1
        AdaBoost(next_w,datas,Standards,n, models)
    return models


models = AdaBoost(w,datas,Standards, 2)

total_score = torch.zeros(len(datas))

for model in models:
    sign = torch.where(model.polarity * datas[:,1] > model.polarity * model.standard, 1, -1)
    total_score += model.Reliability * sign

print("====최종 앙상블 결과====")
print("합산 점수: ", total_score)
print("예측 결과: ",torch.sign(total_score))
print("실제 정답: ", datas[:,2])




