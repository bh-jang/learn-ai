# Data의 Scale을 고려해보는 경우 ( preprocessing )

import numpy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

kn = KNeighborsClassifier()


fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

# tuple로 넘겨야함 ( tuple은 immutable하기 때문에 list에 비해 memory 확보를 덜한다 )
fish_data = numpy.column_stack((fish_length, fish_weight))
fish_target = numpy.concatenate((numpy.ones(35), numpy.zeros(14)))

# stratify는 target의 비율을 보고 적절히 data/test 셋을 나누어준다.
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify=fish_target)


# 해당 Data Set은 Y축에 더 가중치가 있음 (즉 scale이 다름)
# k-nearest-neighbors 는 거리기반 알고리즘이라 특히 더 신경써야함
# 대표적인 방법은 standard score(표준점수 / z점수)가 있다.
mean = numpy.mean(train_input, axis=0)
std = numpy.std(train_input, axis=0)

# numpy 값끼리 브로캐스팅이되어 항목 하나하나가 적용됨
train_scaled = (train_input - mean) / std

# 애매한녀석도 표준점수로 변환한다음 처리
new = ([25, 150] - mean) / std

kn.fit(train_scaled, train_target)


## TEST SET 평가
test_scaled = (test_input - mean) / std
train_result = kn.score(test_scaled, test_target)
new_result = kn.predict([new])
print(train_result, new_result)

distances, indexs = kn.kneighbors([new])
plt.scatter(train_scaled[:, 0], train_scaled[:, 1])
plt.scatter(new[0], new[1], marker='^')
plt.scatter(train_scaled[indexs[:], 0], train_scaled[indexs[:], 1])
plt.show()
