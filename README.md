# DNN_Project

## feature_map_distance.ipynb

### 2022년 07월 05일 화요일

* ED_arrays, ED_array_i_j는 유클리드 거리 데이터에 대한 분석을 하기 위해서 필요한 변수임. 실제 거리를 계산하는 데에 있어서 필요한 변수임.
* normalization이나 min-max에서 필요함.

### 2022년 07월 06일 수요일

1. n,m -> N, M로 바꿈

    for i in range(K) -> for k in range(K)로 바꿈.

2. DIstance_arrays의 shape을 (n, m, k)에서 (k, n, m)로 바꾸어야 함.

    -> 바꾸긴 했는데 그리 차이는 없어보임. 그냥 없애는 것도 방법. 대신 Distance를 람다 함수로 만드는게 가독성이나 수정에 용이해보임

3. random.randint 대신 np.random.randint로 바꿈.

    -> 데이터는 정수가 아니라 실수임. 이거 고쳐야 함.

4. 테스트 데이터 생산할 때 np.append 때문에 시간이 많이 걸림. 하지만 np.random.randint로 한 번에 데이터를 생산해여 np.append를 전혀 쓰지 않아서 몇 초 안으로 실행이 끝남.

    실제로 훈련 데이터나 테스트 데이터가 있다면 .npy 파일로 저장한 다음 이를 로드하면 되는데 sql이나 pickle보다 빠르다고 함.

5. 연습 2부터 4까지의 내용 practice.ipynb로 옮김. 연습 1은 삭제함

6. 거리를 구할 때, 넘파이 배열에 저장하는 방법 말고 딕셔너리에 람다함수를 저장해서 사용할 람다 metric으로 거리를 구할 수 있게 함.

    예를 들어, Distance['Taxi'] = lambda x, y: abs(x-y)

7. 이 ipynb을 기준으로 class를 만들기 위해 FMD_class.ipynb을 만듦

---

#### Note

* Distance를 미리 저장해두면 빠르긴 하지만 Distance의 계산법을 바꾸었을 때 다시 저장하고 사용해야함. 또한, 소량이더라도 램을 잡아먹음.

- Distance를 저장하지 않고 람다 함수를 사용한다면 램을 잡아먹지 않지만 속도는 조금 느림. 하지만 distance의 계산법을 바꾸었을 때 바로 적용이 가능함.

- 데이터를 받을 때 평균을 어떻게 낼 거냐가 중요한 듯 만들어서 시뮬레이션 해봐야 할 듯

- 속도를 빠르게 하거나 저장 공간을 효율적으로 사용하는 방법을 찾아야 함.

- 데이터 입력 값이 어떻게 되었든 간에 거리 계산을 할 수 있어야 함.
    좀 애매해져서 거리를 어떻게 계산할지 생각해야함.
