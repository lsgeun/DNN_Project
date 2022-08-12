import numpy as np
import matplotlib.pyplot as plt
import math
import os
import pickle
import copy

class FMD():
    ''' dir_infos
    root_dir: 관련된 데이터가 모두 저장되어 있는 디렉토리 경로
    origin_dir: train, rvalid, wvalid가 있는 디렉토리
    eval_dir: 거리를 잴 데이터가 있는 디렉토리
    '''
    root_dir=""; origin_dir=""; eval_dir=""
    ''' data_infos
    (origin_, eval_)names: 데이터 타입에 대한 이름
    (origin_, eval_)K: 피처 맵의 개수, L: 레이어의 개수(0~L-1는 각 레이어의 인덱스)
    shape: 레이어의 넘파이 모양; FMP_count: 피처 맵 패키지에 저장된 넘파이 개수
    '''
    origin_names = ["train", "rvalid", "wvalid"]; origin_K={}; eval_names = []; eval_K={}
    L=0; shape=[]; FMP_count=0
    ''' FM_repres
    TFM_repre: 훈련 대표 피처 맵(베이스 피처 맵)
    RFM_repre: 정분류 대표 피처 맵
    WFM_repre: 오분류 대표 피처 맵
    '''
    TFM_repre=[]; RFM_repre=[]; WFM_repre=[]
    ''' AMs
    TAM: 훈련 활성화 피처 맵
    RAM: 정분류 활성화 피처 맵
    WAM: 오분류 활성화 피처 맵
    '''
    TAM=[]; RAM=[]; WAM=[]
    ''' alpha_infos
    alpha는 거리 계산을 위한 인덱스를 고르기 위해 필요한 변수이다.
    r_minus_w_max(= max(r-w))는 나중에 alpha 값에 의존하여 최대로 만들어진다.
    alpha_min, alpha_max는 각 레이어에서의 alpha가 가질 수 있는 최소값 최대값을 나타낸 것이다.
    [hyper parameter] alpha_slice은 alpha_min에서 alpha_max로 몇 번의의 간격으로 도착할지 알려주는 변수임.
    '''
    alpha=[]; alpha_slice=[]; r_minus_w_max=[]
    alpha_min=[]; alpha_max=[]
    ''' DAM_infos
    DAM_indexes는 나중에 거리 계산할 때 쓰이는 다차원 인덱스들의 집합이다. 각 원소는 피처 맵의 한 원소의 인덱스를 나타낸다.
    각 레이어마다 튜플들 세트가 있어야 함. np.array의 item 메소드를 사용할 것이기 때문
    DAM: 거리 활성화 맵, DAM는 거리 계산을 위한 인덱스만 활성된 맵이다.
    [hyper parameter] DAM_select는 DAM를 고르는 방법을 알려줌.
    '''
    DAM_indexes=[]; DAM=[]; DAM_select=[]
    ''' layer_infos
    [hyper parameter] W는 각 레이어의 피처 맵에 곱할 weight 중요도이다
    레이어 피처 맵에 대한 가시적인 출력을 위해 1차 이상의 배열을 2차원 배열로 바꿈
    square_NPs_side은 1차 이상의 배열을 2차원 배열로 바꾸기 위해 필요한 2차원 배열의 한 변의 길이임.
    [hyper parameter] lfmd_select는 각 레이어에 대한 피처 맵을 구하는 방법을 저장한다.
    norm_min, norm_max는 레이어 피처 맵 거리를 구하기 전에 정규화할 범위의 최소 최대를 나타낸다.
    '''
    W=[]; lfmd_select=[]; norm_min=0; norm_max=1
    ''' fmdc infos
    fmdc: 피처 맵 거리 기준으로 어떤 데이터가 나중에 오분류 될 거 같은지 판단함.
    rfmds: 정분류 피처 맵 거리들을 모아둔 것
    wfmds: 오분류 피처 맵 거리들을 모아둔 것
    '''
    fmdc=-1; rfmds=[]; wfmds=[]
    ''' eval_infos
    eval한 결과를 담거나 시각화 하기 위해 필요한 것
    eval 데이터 타입마다 eval_U, is_eval_FMD, fmds가 나온다.
    eval_U는 데이터의 정분류(True), 오분류(False) 유무를 True, False로 담는다
    is_eval_FMD는 데이터가 is_eval_FMD이면 True, ts_fmd가 아니면 False를 담는다.
    fmds는 데이터의 fmd를 담는다.
    TP, FN, TN, FP는 confusion matrix를 표현하기 위한 가장 기본적인 속성이다.
    TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR
    '''
    eval_U={}; is_eval_FMD={}; fmds={}
    '''CM_infos
    CM_infos: Confusion Matrix로 효과성을 따지기 위해서 필요한 변수들의 모임이다
    '''
    TP={}; FN={}; TN={}; FP={}; P={}; N={}
    TPR={}; TNR={}; PPV={}; NPV={}; FNR={}; FPR={}; FDR={}; FOR={}
    ''' square_NPs_infos
    square_NPs_figsize는 show_square_NPs에서 그려지는 그래프의 figsize를 조절한다.
    square_NPs_columns은 show_square_NPs에서 그려지는 그래프의 column을 조절한다.
    '''
    square_NPs_figsize=[3*20, 8*20]; square_NPs_column=10

    def __init__(self, root_dir_=""):

        # root 디렉토리 입력이 없다면 return
        if root_dir_=="":
            print("루트 디렉토리 경로를 설정해주세요.")
            return

        # * root 디렉토리
        self.root_dir = root_dir_
        # * origin 디렉토리
        self.origin_dir = f"{self.root_dir}/origin"
        # 훈련 피처 맵을 저장하는 디렉토리
        self.train_dir=f"{self.origin_dir}/{self.origin_names[0]}"
        # 정분류 피처 맵을 저장하는 디렉토리
        self.rvalid_dir=f"{self.origin_dir}/{self.origin_names[1]}"
        # 오분류 피처 맵을 저장하는 디렉토리
        self.wvalid_dir=f"{self.origin_dir}/{self.origin_names[2]}"
        # * eval 디렉토리
        self.eval_dir = f"{self.root_dir}/eval"

        # * 객체를 불러오거나 저장할 디렉토리 생성
        # os.paht.isdir는 '\ ' 대신, ' '을 써도 됨
        is_there_instances = os.path.isdir(f"{self.root_dir}/instances")
        # ' '을 '\ '로 바꿈
        root_dir = self.root_dir.replace(' ', '\ ')
        # instances 디렉토리가 없을 경우만 instances 생성
        if not is_there_instances:
            os.system(f"mkdir {root_dir}/instances")

    def fit(self):
        # 객체의 속성 초기화
        self.set_data_infos_and_related()
        self.set_FM_repres()
        self.set_AMs_and_related()
        self.set_fmds()
        self.set_fmdc()

    def set_root_dir(self, root_dir_):
        # * root 디렉토리
        self.root_dir = root_dir_

        # * origin 디렉토리
        self.origin_dir = f"{self.root_dir}/origin"
        # * 훈련을 저장하는 디렉토리
        self.train_dir=f"{self.origin_dir}/{self.origin_names[0]}"
        # * 정분류 테스트를 저장하는 디렉토리
        self.rvalid_dir=f"{self.origin_dir}/{self.origin_names[1]}"
        # * 오분류 테스트를 저장하는 디렉토리
        self.wvalid_dir=f"{self.origin_dir}/{self.origin_names[2]}"
        
        # * eval 디렉토리
        self.eval_dir = f"{self.root_dir}/eval"

    def set_data_infos_and_related(self):
        # * 새로운 데이터를 받을 수 있게끔 속성을 초기 상태로 만듦.
        self.shape = []; self.lfmd_select = []; self.W = []

        # * root dir의 data_infos.txt 열기
        data_infos = open(f"{self.root_dir}/data_infos.txt", 'r')
        data_infos_strs = data_infos.read()
        data_infos_str_list = data_infos_strs.split('\n')
        # * 0th: origin_K
        origin_K = list(map(int, data_infos_str_list[0].split()))
        origin_name_K_zip = zip(self.origin_names, origin_K) 
        for origin_name, origin_K in origin_name_K_zip:
            self.origin_K[origin_name] = origin_K
        # * 1th: eval_names
        self.eval_names = data_infos_str_list[1].split()
        # * 2th: eval_K
        eval_K = list(map(int, data_infos_str_list[2].split()))
        eval_name_K_zip = zip(self.eval_names, eval_K) 
        for eval_name, eval_K in eval_name_K_zip:
            self.eval_K[eval_name] = eval_K
        # * 3th: L
        self.L = int(data_infos_str_list[3])
        # * 4+0th ~ 4+(L-1)th: shape
        for l in range(self.L):
            shape_l = list(map(int,data_infos_str_list[4+l].split()))
            self.shape.append(shape_l)
        # * 4+Lth: FMP_count
        self.FMP_count = int(data_infos_str_list[4+self.L])
        # * root dir의 data_infos.txt 닫기
        data_infos.close()

        # * 레이어 피처 맵을 구하는 방식을 se_lfmd 모두 통일
        for l in range(self.L):
            self.lfmd_select.append("se_lfmd")
        
        # * eval한 결과를 담거나 시각화하기 위한 값들을 초기화
        for eval_name in self.eval_names:
            self.eval_U[eval_name] = np.load(f'{self.eval_dir}/{eval_name}/{eval_name}_eval_U.npy')
            self.is_eval_FMD[eval_name] = []
            self.fmds[eval_name] = []

            self.TP[eval_name]=-1; self.FN[eval_name]=-1; self.TN[eval_name]=-1; self.FP[eval_name]=-1
            self.N[eval_name]=-1; self.P[eval_name]=-1
            self.TPR[eval_name]=-1; self.TNR[eval_name]=-1; self.PPV[eval_name]=-1; self.NPV[eval_name]=-1
            self.FNR[eval_name]=-1; self.FPR[eval_name]=-1; self.FDR[eval_name]=-1; self.FOR[eval_name]=-1
        
        # ! W를 등차수열로 만듦
        # for l in range(self.L):
        #     self.W.append((l+1)*(2/(self.L*(self.L+1))))
        # * W를 균등하게 만듦
        for l in range(self.L):
            self.W.append(1/self.L)

    def set_FM_repres(self):
        # * 새로운 데이터를 받기 전에 빈 배열로 초기화 함.
        self.TFM_repre=[]; self.RFM_repre=[]; self.WFM_repre=[];

        # 인스턴스 속성을 변수로 포인터처럼 가르킴
        train = self.origin_names[0]; rvalid = self.origin_names[1]; wvalid = self.origin_names[2]
        L = self.L; shape = self.shape

        def set_FM_repre(origin):
            # 인스턴스 속성을 변수로 포인터처럼 가르킴
            origin_K = self.origin_K[origin]

            if origin == train:
                OFM_repre = self.TFM_repre; origin_dir = self.train_dir;
            elif origin == rvalid:
                OFM_repre = self.RFM_repre; origin_dir = self.rvalid_dir;
            elif origin == wvalid:
                OFM_repre = self.WFM_repre; origin_dir = self.wvalid_dir;
            else:
                print('잘못된 origin: ', origin, sep='')
                return

            # * 각 레이어의 피처 맵을 0으로 초기화하여 생성
            for l in range(L):
                OFM_repre_l = np.zeros(shape[l])
                OFM_repre.append(OFM_repre_l)

            # OFMP_k는 k번째 origin 데이터가 속한 FMP임
            # k번 째 데이터의 OFMPI는 OFMPI_k = k // self.FMP_count임
            # k번 째 데이터의 OFMPO은 OFMPO_k = k % self.FMP_count임
            OFMP_k = None; prev_OFMPI_k = None; cur_OFMPI_k = None

            k = 0
            # * 0번 째 데이터로 OFM_repre를 초기화한다.
            # 0번 째 데이터가 속한 OFMP_k를 불러들인 후
            prev_OFMPI_k = k // self.FMP_count
            OFMPO_k = k % self.FMP_count
            with open(f'{origin_dir}/{origin}_{prev_OFMPI_k}.pickle', 'rb') as f:
                OFMP_k = pickle.load(f)
            # 0번 째 데이터를 OFM_repre에 넣는다.
            for l in range(L):
                OFM_k_l = OFMP_k[OFMPO_k][l]
                OFM_repre[l] = OFM_repre[l] + OFM_k_l

            # k = 1 ~ K-1
            # * 1~K-1번 째 데이터로 OFM_repre을 구한다.
            for k in range(1, origin_K):
                # k번 째 데이터의 OFMPI, OFMPO 구함
                cur_OFMPI_k = k // self.FMP_count
                OFMPO_k = k % self.FMP_count
                
                # * OFMP_k가 이미 램에 있다면 가지고 오지 않고
                # * 램에 없다면 이전 OFMP를 램에서 지우고 현재 OFMP를 램으로 가지고 온다.
                # cur_OFMPI_k와 prev_OFMPI_k가 같다면
                if cur_OFMPI_k == prev_OFMPI_k:
                    pass # 아무 작업 하지 않고
                # cur_OFMPI_k와 prev_OFMPI_k가 다를 경우
                else:
                    # 이전 OFMP_k의 기억공간을 램에서 제거한 후
                    del OFMP_k
                    # cur_OFMPI_k를 현재 OFMP_k를 가지고 온다.
                    with open(f'{origin_dir}/{origin}_{cur_OFMPI_k}.pickle', 'rb') as f:
                        OFMP_k = pickle.load(f)

                # prev_OFMPI_k를 cur_OFMPI_k로 초기화 
                prev_OFMPI_k = cur_OFMPI_k

                for l in range(L):
                    OFM_k_l = OFMP_k[OFMPO_k][l]
                    OFM_repre[l] = (OFM_repre[l]*k + OFM_k_l)/(k+1)
            
            # OFM을 모두 순회한 후 OFMP_k의 기억공간을 램에서 제거
            del OFMP_k

        # * 훈련, 정분류 테스트, 오분류 테스트 데이터에 대한 FM_repre을 구함
        set_FM_repre(train); set_FM_repre(rvalid); set_FM_repre(wvalid)

    def set_AMs_and_related(self):
        # * 새로운 데이터를 받을 수 있게끔 속성을 초기 상태로 만듦.
        self.TAM = []; self.RAM = []; self.WAM = []; self.DAM = []
        self.alpha_slice = []; self.DAM_select = []; self.DAM_indexes = []

        # alpha, r_minus_w_max를 0으로 초기화
        self.alpha = np.zeros(self.L)
        self.r_minus_w_max = np.zeros(self.L)

        # [hyperparameter] alpha_slice를 일단 1000으로 함
        for l in range(self.L):
            self.alpha_slice.append(100)

        # alpha의 최소값과 최대값을 구함
        self.alpha_min = np.zeros(self.L)
        self.alpha_max = np.zeros(self.L)
        for l in range(self.L):
            self.alpha_min[l] = np.array([self.RFM_repre[l].min(),
                                         self.TFM_repre[l].min(),
                                         self.WFM_repre[l].min()]).min()
            self.alpha_max[l] = np.array([self.RFM_repre[l].max(),
                                         self.TFM_repre[l].max(),
                                         self.WFM_repre[l].max()]).max()

        # TAM, RAM, WAM을 0으로 초기화 함
        for l in range(self.L):
            TAM_l = np.zeros(self.shape[l])
            self.TAM.append(TAM_l)
        for l in range(self.L):
            RAM_l = np.zeros(self.shape[l])
            self.RAM.append(RAM_l)
        for l in range(self.L):
            WAM_l = np.zeros(self.shape[l])
            self.WAM.append(WAM_l)

        # r-w가 최대가 되는 alpha, TAM, RAM, WAM을 찾음
        for l in range(self.L):
            # range 부분 고칠 필요가 있음
            a_min_l = self.alpha_min[l]; a_max_l = self.alpha_max[l]
            a_slice_l = self.alpha_slice[l]
            interval_l = (a_max_l - a_min_l)/a_slice_l
            # range(a_slice_l+1) 해야 a_min_l 부터 a_max_l 까지 감
            for s, alpha_l in enumerate([a_min_l + interval_l*s for s in range(a_slice_l+1)]):
                TAM_l = np.array(self.TFM_repre[l] > alpha_l)
                RAM_l = np.array(self.RFM_repre[l] > alpha_l)
                WAM_l = np.array(self.WFM_repre[l] > alpha_l)
                
                TAM_l_xnor_RAM_l = np.logical_not(np.logical_xor(TAM_l, RAM_l))
                TAM_l_xnor_WAM_l = np.logical_not(np.logical_xor(TAM_l, WAM_l))
                # r_l은 TAM_l과 RAM_l이 얼마나 유사한지 보여준다.
                # 즉, TAM_l과 RAM_l이 유사할수록 r_l 값이 커진다.
                # w_l도 마찬가지이다.
                r_l = len(np.where(TAM_l_xnor_RAM_l == True)[0])
                w_l = len(np.where(TAM_l_xnor_WAM_l == True)[0])
                # 처음에는 alpha_offset 0으로 초기화하고
                if s == 0:
                    self.alpha[l] = alpha_l
                    self.r_minus_w_max[l] = r_l - w_l
                    self.TAM[l] = TAM_l
                    self.RAM[l] = RAM_l
                    self.WAM[l] = WAM_l
                elif s > 0 and r_l - w_l > self.r_minus_w_max[l]:
                    # r-w가 이전의 r-w보다 클 때
                    # 즉, TAM과 RAM이 더 유사해지거나 TAM과 WAM이 더 다를 때
                    # alpha, w-r, TAM, RAM, WAM를 최신화함.
                    self.alpha[l] = alpha_l
                    self.r_minus_w_max[l] = r_l - w_l
                    self.TAM[l] = TAM_l
                    self.RAM[l] = RAM_l
                    self.WAM[l] = WAM_l

        # TAM, RAM, WAM를 이용하여 DAM를 구함
        # DAM에 WAM를 deep copy 복사함
        for l in range(self.L):
            DAM_l = self.WAM[l].copy()
            self.DAM.append(DAM_l)

        # [hyperparameter] 일단 모든 레이어를 'and' 방식으로 함
        for l in range(self.L):
            self.DAM_select.append("and")

        # 선택하려는 방식('and', 'or', 등등)대로 DAM를 고름
        for l in range(self.L):
            # 'and' 방식보다 'or' 방식에서 대부분의 인덱스에서 오분류 데이터의 값이 상대적으로 더 커짐
            # 다만 'or' 방식은 'and' 방식보다 거리 계산을 위한 인덱스의 개수가 더 줄어듦
            # 'all' 방식은 모든 인덱스를 다 사용함.
            if self.DAM_select[l] == "and":
                TAM_l_and_RAM_l = np.logical_and(self.TAM[l], self.RAM[l])
                # WAM에서 (TAM 교집합 RAM)과 곂치는 부분을 False로 만듦
                np.place(self.DAM[l], TAM_l_and_RAM_l, False)
            elif self.DAM_select[l] == "or":
                TAM_l_or_RAM_l = np.logical_or(self.TAM[l], self.RAM[l])
                # WAM에서 (TAM 합집합 RAM)과 곂치는 부분을 False로 만듦)
                np.place(self.DAM[l], TAM_l_or_RAM_l, False)
            elif self.DAM_select[l] == "all":
                not_DAM_l = np.logical_not(self.DAM[l])
                # 모든 인덱스를 다 사용함
                np.place(self.DAM[l], not_DAM_l, True)

        # * 만약 DAM[l]의 원소가 모두 False라면 WAM[l]로 초기화함.
        for l in range(self.L):
            shape_l_size = 1
            for shaple_l_i in self.shape[l]:
                shape_l_size *= shaple_l_i

            if len(np.where(self.DAM[l] == False)[0]) == shape_l_size:
                DAM_l = self.WAM[l].copy()
                self.DAM[l] = DAM_l

        # DAM_indexes를 지정함
        for l in range(self.L):
            nonzero_DAM_l = np.nonzero(self.DAM[l])
            DAM_indexes_l = np.empty((1,len(nonzero_DAM_l[0])), dtype="int32")
            # l 레이어 차원의 수 만큼 각 차원에 대한 인덱스들을 DAM_indexes_l에 삽입
            for i in range(len(nonzero_DAM_l)):
                DAM_indexes_l = np.append(DAM_indexes_l, nonzero_DAM_l[i].reshape(1,-1), axis=0)
            # 처음 배열은 np.empty 메소드로 만들어진 쓰레기 값이라 버림
            # 가로 방향이라 세로 방향으로 길게 늘어지도록 바꿈
            DAM_indexes_l = list(DAM_indexes_l[1:].T)
            # DAM_indexes_l 각 원소가 리스트 형태인데 그것을 튜플로 바꿈
            # 튜플로 만드는 이유는 np.item() 메소드가 튜플을 인자로 받기 때문
            for i in range(len(DAM_indexes_l)):
                DAM_indexes_l[i] = tuple(DAM_indexes_l[i])

            self.DAM_indexes.append(DAM_indexes_l)
    
    def se_lfmd(self, FM_k_l, l, percent=50, sensitive=1):
        '''
        se_lfmd: shift exponential layer feature map distance
        일단 디폴트로 length_min length_max의 50(정중앙 값)에 해당하는 부분을 origin(원점)으로 이동
        그리고 민감도는 디폴트로 1로 설정함
        '''
        se_lfmd = 0
        
        norm_min = self.norm_min; norm_max = self.norm_max
        # self.TFM_repre[l], FM_k_l를 self.norm_min, self.norm_max으로 정규화
        TFM_repre_l_norm = self.normalize_layer(self.TFM_repre[l], norm_min, norm_max)
        FM_k_l_norm = self.normalize_layer(FM_k_l, norm_min, norm_max)

        # lengths: 인덱스 마다 TFM_repre_norm과 FM_k_l_norm 사이의 거리(절대값)를 구함
        lengths = abs(TFM_repre_l_norm - FM_k_l_norm)
        
        # 'shift_value'을 구함
        # norm_min, norm_max가 같은 두 값의 길이의 min(length_min)은 0이고
        # length_max는 norm_max - norm_min임
        length_max = norm_max - norm_min; length_min = 0
        # 
        length_interval_max = length_max - length_min
        length_interval_percent = length_interval_max * (percent/100)
        # value_to_be_origin는 나중에 원점이 될 값임
        value_to_be_origin = length_min + length_interval_percent
        # shift_value는 value_to_be_origin을 0으로 옮기기 위한 이동 값임
        shift_value = -value_to_be_origin

        exp_lengths_minus_shift_value = np.zeros(self.shape[l])
        # 각 원소를 shift value 만큼 이동시키고 'exponential'을 취함
        for index in self.DAM_indexes[l]:
            exp_lengths_minus_shift_value.itemset(index, np.exp(lengths.item(index) - shift_value)**sensitive)
        # se를 취한 값들을 모두 더한 것이 se_lfmd임
        se_lfmd = exp_lengths_minus_shift_value.sum()
        
        return se_lfmd
    
    def normalize_layer(self, layer, min, max):
        '''
        'layer의 min, max'으로 layer를 'min-max 정규화'를 한 후
        최소값이 min, 최대값이 max가 되도록 layer를 정규화 한다.
        layer(넘파이)에 스칼라 곱셈과 스칼라 덧셈을 적용하여 구현할 수 있다.
        '''
        # 'layer의 min, max'으로 layer를 'min-max 정규화'
        layer_min = layer.min(); layer_max = layer.max();
        normalized_layer = None
        # layer 값들이 하나라도 다르다면
        if layer_max - layer_min != 0:
            # layer에 layer_min, layer_max으로 min-max 적용
            layer = (layer - layer_min) / (layer_max - layer_min)
            
            scalar_multiplyer = max - min
            scalar_adder = min

            normalized_layer = scalar_multiplyer*layer + scalar_adder
        # layer 값들이 모두 같다면
        else:
            # 아마도 거리 계산할 때 거리 크기를 줄이기 위해
            # layer를 min, max의 중앙값으로 바꾸기
            layer = np.zeros(layer.shape)
            normalized_layer = layer  + min + (max - min)/2

        return normalized_layer
    
    def Ln_lfmd(self, FM_k_l, l, n=1):
        '''
        Ln_lfmd: Ln layer feature map distance
        레이어의 인덱스들 간의 절대값을 모두 더한다.
        '''
        Ln_lfmd = 0
        
        norm_min = self.norm_min; norm_max = self.norm_max
        # self.TFM_repre[l], FM_k_l를 self.norm_min, self.norm_max으로 정규화
        TFM_repre_l_norm = self.normalize_layer(self.TFM_repre[l], norm_min, norm_max)
        FM_k_l_norm = self.normalize_layer(FM_k_l, norm_min, norm_max)

        # lengths: 인덱스 마다 TFM_repre_norm과 FM_k_l_norm 사이의 거리(절대값)를 구함
        lengths = abs(TFM_repre_l_norm - FM_k_l_norm)
        # DAM에서 True인 부분만 가지고 옴.
        lengths = lengths[self.DAM[l]]
        # lengths의 각 원소에 지수 n을 취함
        lengths_pow_n = np.power(lengths, n)
        # lengths_pow_n을 모두 더한 후 1/n 지수를 취하면 Ln_lfmd임
        Ln_lfmd = np.power(lengths_pow_n.sum(), 1/n)
        
        return Ln_lfmd

    def lfmd(self, lfmd_select):
        if lfmd_select == "se_lfmd":
            return self.se_lfmd
        elif lfmd_select == "Ln_lfmd":
            return self.Ln_lfmd

    def fmd(self, FM_k):
        # 피처 맵 거리를 0으로 초기화
        fmd=0
        # lfmds: 레이어 피처 맵 거리를 담는 곳
        lfmds=[]
        # 각 레이어에 대한 레이어 피처 맵 거리 계산법으로 레이어 피처 맵 계산
        for l in range(self.L):
            FM_k_l = FM_k[l]
            lfmd_l = self.lfmd(self.lfmd_select[l])(FM_k_l, l)
            lfmds.append(lfmd_l)
        # 레이어 피처 맵마다 weight를 줌
        for l in range(self.L):
            fmd += self.W[l]*lfmds[l]

        return fmd
    
    def set_fmds(self):
        # self.rfmds, self.wfmds는 정분류 테스트 데이터, 오분류 테스트 데이터에 대한 fmd를 저장함
        # * 새로운 데이터를 받기 전에 빈 배열로 초기화 함.
        self.rfmds=[]
        self.wfmds=[]

        rvalid = self.origin_names[1]; wvalid = self.origin_names[2]

        def set_fmds(test_name):
            # RFM와 WFM을 부르기 위한 변수들을 선언함
            if test_name=="rvalid":
                test=self.origin_names[1]; test_dir=self.rvalid_dir; test_K=self.origin_K[test]
                FMP_k=None; prev_FMPI_k=None; cur_FMPI_k=None; fmds = self.rfmds
            elif test_name=="wvalid":
                test=self.origin_names[2]; test_dir=self.wvalid_dir; test_K=self.origin_K[test]
                FMP_k=None; prev_FMPI_k=None; cur_FMPI_k=None; fmds = self.wfmds

            for k in range(test_K):
                # k번 째 데이터의 FMPI, FMPO 구함
                cur_FMPI_k = k // self.FMP_count
                FMPO_k = k % self.FMP_count

                # FMP_k가 이미 있다면 가지고 오지 않고 없다면 이전 FMP를 지우고 현재 FMP를 가지고 온다.
                if cur_FMPI_k == prev_FMPI_k:
                    pass
                else:
                    del FMP_k
                    with open(f'{test_dir}/{test}_{cur_FMPI_k}.pickle', 'rb') as f:
                        FMP_k = pickle.load(f)

                prev_FMPI_k = cur_FMPI_k

                FM_k = FMP_k[FMPO_k]

                fmds.append(self.fmd(FM_k))

            # test를 모두 순회하고 난 후 FMP_k의 기억공간을 램에서 제거
            del FMP_k
        
        set_fmds(rvalid); self.rfmds = np.array(self.rfmds)
        set_fmds(wvalid); self.wfmds = np.array(self.wfmds)

    def set_fmdc(self):        
        # self.fmdc = wfmds.mean() # !
        # self.fmdc = wfmds[len(wfmds) // 2] # !
        # self.fmdc = wfmds[(len(wfmds) // 3) * 2] # !
        # self.fmdc = wfmds.max() # !
        # self.fmdc = (rfmds.max() + wfmds.min()) / 2 # !
        # print(f'rfmds.max(): {rfmds.max()}, wfmds.min(): {wfmds.min()}') # !
        # self.fmdc = rfmds.max() # !
        self.fmdc = self.wfmds.min()

    def eval(self):
        # * 새로운 데이터를 받을 수 있게끔 속성을 초기 상태로 만듦.
        for eval_name in self.eval_names:
            self.is_eval_FMD[eval_name] = []; self.fmds[eval_name] = []

        for eval_name in self.eval_names:
            # self.fmds[eval_name], self.is_eval_FMD[eval_name]는
            # self.set_data_infos_and_related()에서 []로 초기화됨
            
            # EFM을 부르기 위한 변수들을 선언함
            eval_dir=f'{self.eval_dir}/{eval_name}'; eval_K=self.eval_K[eval_name]
            EFMP_k=None; prev_EFMPI_k=None; cur_EFMPI_k=None;

            for k in range(eval_K):
                # k번 째 데이터의 EFMPI, EFMPO 구함
                cur_EFMPI_k = k // self.FMP_count
                EFMPO_k = k % self.FMP_count

                # EFMP_k가 이미 있다면 가지고 오지 않고
                #             없다면 이전 EFMP를 지우고 현재 EFMP를 가지고 온다.

                # cur_EFMPI_k와 prev_EFMPI_k가 같다면
                if cur_EFMPI_k == prev_EFMPI_k:
                    pass # 아무 작업 하지 않고

                # cur_EFMPI_k와 prev_EFMPI_k가 다를 경우
                else:
                    # 이전 EFMP_k의 기억공간을 램에서 제거한 후
                    del EFMP_k

                    # cur_EFMPI_k를 현재 EFMP_k를 가지고 온다.
                    with open(f'{eval_dir}/{eval_name}_{cur_EFMPI_k}.pickle', 'rb') as f:
                        EFMP_k = pickle.load(f)

                # prev_EFMPI_k를 cur_EFMPI_k로 초기화 
                prev_EFMPI_k = cur_EFMPI_k

                EFM_k = EFMP_k[EFMPO_k]
                
                fmd = self.fmd(EFM_k)
                
                self.is_eval_FMD[eval_name].append(fmd >= self.fmdc)
                self.fmds[eval_name].append(fmd)
            
            # eval_name을 모두 순회한 후 EFMP_k의 기억공간을 램에서 제거
            del EFMP_k
            
            self.is_eval_FMD[eval_name] = np.array(self.is_eval_FMD[eval_name])
            self.fmds[eval_name] = np.array(self.fmds[eval_name])
    
    def set_CM_infos(self):
        # * confusion matrix를 표현하기 위한 가장 기초적인 원소들을 초기화
        for eval_name in self.eval_names:
            # * self.TP, self.FN, self.TN, self.FP을 초기화하기 위한 변수 초기화
            eval_U_K = self.eval_K[eval_name]
            eval_U_r = len(np.nonzero(self.eval_U[eval_name])[0])
            eval_U_w = eval_U_K - eval_U_r
            eval_FMD_K = len(self.eval_U[eval_name][self.is_eval_FMD[eval_name]])
            eval_FMD_r = len(np.nonzero(self.eval_U[eval_name][self.is_eval_FMD[eval_name]])[0])
            eval_FMD_w = eval_FMD_K - eval_FMD_r

            # * self.TP, self.FN, self.TN, self.FP 초기화
            self.TP[eval_name] = (eval_U_r - eval_FMD_r) / eval_U_K
            self.FN[eval_name] = eval_FMD_r / eval_U_K
            self.TN[eval_name] = eval_FMD_w / eval_U_K
            self.FP[eval_name] = (eval_U_w - eval_FMD_w) / eval_U_K
            
            # * self.P, self.N 초기화
            self.P[eval_name] = self.TP[eval_name] + self.FN[eval_name]; self.N[eval_name] = self.TN[eval_name] + self.FP[eval_name]
            # * self.TPR, self.TNR, self.FNR, self.FPR 초기화
            self.TPR[eval_name] = self.TP[eval_name] / self.P[eval_name]; self.TNR[eval_name]= self.TN[eval_name] / self.N[eval_name]
            self.FNR[eval_name] = self.FN[eval_name] / self.P[eval_name]; self.FPR[eval_name] = self.FP[eval_name] / self.N[eval_name]
            # * self.PPV, self.NPV, self.FDR, self.FOR 초기화
            # self.TP[eval_name] + self.FP[eval_name]가 0이 되는 경우가 발생할 때 그것으로 나누지 않음
            # 초기화된 -1 값을 그대로 가지고 감
            if self.TP[eval_name] + self.FP[eval_name] > 0:
                self.PPV[eval_name] = self.TP[eval_name] / (self.TP[eval_name] + self.FP[eval_name])
                self.FDR[eval_name] = self.FP[eval_name] / (self.TP[eval_name] + self.FP[eval_name])
            if self.FN[eval_name] + self.TN[eval_name] > 0:
                self.NPV[eval_name] = self.TN[eval_name] / (self.FN[eval_name] + self.TN[eval_name])
                self.FOR[eval_name] = self.FN[eval_name] / (self.FN[eval_name] + self.TN[eval_name])
    
    def set_square_NPs_infos(self, figsize=None, column=None):
        if figsize!=None:
            self.square_NPs_figsize = figsize
        if column!=None:
            self.square_NPs_column = column

    def show_all(self, show_all_mask):
        if show_all_mask['show_dirs']:
            print('self.show_dirs()'); self.show_dirs(); print('-'*200)

        if show_all_mask['show_data_infos']:
            print('self.show_data_infos()'); self.show_data_infos(); print('-'*200)

        if show_all_mask['show_hyper_parameter']:
            # * FMD_old 에서는 hyper_parameter가 모두 있지 않음.
            # print('self.show_hyper_parameter()'); self.show_hyper_parameter(); print('-'*200)
            pass

        if show_all_mask['show_FM_repres']:
            self.set_square_NPs_infos(figsize=[60,60], column=7)
            print('self.show_FM_repres()'); self.show_FM_repres(); print('-'*200)

        if show_all_mask['show_AMs_and_related']:
            self.set_square_NPs_infos(figsize=[60,60], column=7)
            print('self.show_AMs_and_related()'); self.show_AMs_and_related(); print('-'*200)

        if show_all_mask['show_layer_infos']:
            print('self.show_layer_infos()'); self.show_layer_infos(); print('-'*200)

        if show_all_mask['show_fmds_box_plot']:
            print('self.show_fmds_box_plot()'); self.show_fmds_box_plot(); print('-'*200)

        if show_all_mask['show_fmdc']:
            print('self.show_fmdc()'); self.show_fmdc(); print('-'*200)

        if show_all_mask['show_eval_infos']:
            print('self.show_eval_infos()'); self.show_eval_infos(); print('-'*200)

        if show_all_mask['show_fmd_right_ratio_graph']:
            print('self.show_fmd_right_ratio_graph()'); self.show_fmd_right_ratio_graph(); print('-'*200)

        if show_all_mask['show_eval_venn_diagrams']:
            print('self.show_eval_venn_diagrams()'); self.show_eval_venn_diagrams(); print('-'*200)

        if show_all_mask['show_efficience_and_FMD_ratio']:
            print('self.show_efficience_and_FMD_ratio()'); self.show_efficience_and_FMD_ratio(); print('-'*200)
        
    def show_data_infos(self):
        print("self.origin_names"); print(self.origin_names)
        print("self.origin_K"); print(self.origin_K)
        print("self.eval_names"); print(self.eval_names)
        print("self.eval_K"); print(self.eval_K)
        print("self.L"); print(self.L)
        print("self.shape"); print(self.shape)

    def show_hyper_parameter(self):
        print("self.alpha_slice"); print(self.alpha_slice)
        print("self.DAM_types"); print(self.DAM_types)
        print("self.W_types"); print(self.W_types)
        print("self.lfmd_types"); print(self.lfmd_types)
        print("self.fmdc_types"); print(self.fmdc_types)

    def show_FM_repres(self):
        self.set_square_NPs_infos([60,60],column=7)
        print("self.TFM_repre"); self.show_square_NPs(self.TFM_repre.copy()) # 포인터가 아닌 복사본을 인자로 넘겨줌
        print("self.RFM_repre"); self.show_square_NPs(self.RFM_repre.copy()) # 포인터가 아닌 복사본을 인자로 넘겨줌
        print("self.WFM_repre"); self.show_square_NPs(self.WFM_repre.copy()) # 포인터가 아닌 복사본을 인자로 넘겨줌

    def show_AMs_and_related(self):
        # self.set_square_NPs_infos([60,60],column=7)
        # print("self.TAM"); self.show_square_NPs(self.TAM.copy()) # 포인터가 아닌 복사본을 인자로 넘겨줌
        # print("self.RAM"); self.show_square_NPs(self.RAM.copy()) # 포인터가 아닌 복사본을 인자로 넘겨줌
        # print("self.WAM"); self.show_square_NPs(self.WAM.copy()) # 포인터가 아닌 복사본을 인자로 넘겨줌

        print("self.alpha_slice"); print(self.alpha_slice)
        print("self.alpha_min"); print(self.alpha_min)
        print("self.alpha"); print(self.alpha)
        print("self.alpha_max"); print(self.alpha_max)
        print("self.r_minus_w_max"); print(self.r_minus_w_max)

        print("self.DAM_select"); print(self.DAM_select)
        # print("self.DAM"); self.show_square_NPs(self.DAM.copy()) # 포인터가 아닌 복사본을 인자로 넘겨줌

    def show_layer_infos(self):
        print("self.lfmd_select"); print(self.lfmd_select)
        print("self.W"); print(self.W)
        print("self.norm_min"); print(self.norm_min)
        print("self.norm_max"); print(self.norm_max)

    def show_fmds_box_plot(self):
        plt.boxplot([self.rfmds, self.wfmds],notch=True)
        plt.xticks([1, 2], ['rfmds', 'wfmds'])
        plt.show()

    def show_fmdc(self):
        print("self.fmdc"); print(self.fmdc)

    def show_dirs(self):
        print("self.root_dir"); print(self.root_dir)
        
        print("self.origin_dir"); print(self.origin_dir)
        print("self.train_dir"); print(self.train_dir)
        print("self.rvalid_dir"); print(self.rvalid_dir)
        print("self.wvalid_dir"); print(self.wvalid_dir)

        print("self.eval_dir"); print(self.eval_dir)
    
    def show_square_NPs(self, np_arr, color_map=True):
        if color_map==True:
            # np_arr에 np_arr의 최소의 값부터 최대 값까지 그리는 넘파이 추가
            # 이 넘파이 배열을 그리는 이유는 최소 최대에 대한 시각적인 표현을 할 수 있을 뿐만 아니라
            # color bar로 인해 좁게 그려지는 넘파이 배열을 없앨 수 있다.
            # np_arr_min 찾기
            np_arr_min = np_arr[0].min()
            for i in range(1, len(np_arr)):
                if np_arr_min > np_arr[i].min():
                    np_arr_min = np_arr[i].min()
            # np_arr_max 찾기
            np_arr_max = np_arr[0].max()
            for i in range(1, len(np_arr)):
                if np_arr_max > np_arr[i].max():
                    np_arr_max = np_arr[i].max()
            # bool 타입이라면 마지막에 넘파이에 True, False만 넣고 즉, (T,T), (T,F), (F,T), (F,F)만 넣고
            if str(np_arr[0].dtype) == 'bool':
                np_arr.append(np.array([True, False]))
            # 그게 아니라면 마지막에 넘파이에 np_arr_slice로 잘린 연속적인 값들을 넣음.
            else:
                # np_arr_slice: np_arr_min 부터 np_arr_max 까지 그리는데 몇 번에 걸쳐서 그릴지 정하기
                np_arr_slice = 1024
                np_arr_interval = (np_arr_max - np_arr_min) / np_arr_slice
                # np_arr에 np_arr의 최소의 값부터 최대 값까지 그리는 넘파이 생성
                np_min_to_max = np.array([np_arr_min + i*np_arr_interval for i in range(np_arr_slice+1)])
                np_arr.append(np_min_to_max)

        # 레이어 개수 만큼 레이어 원소 개수 계산
        element_count=[]
        for ith in range(len(np_arr)):
            ith_element_count = 1
            for ith_shape_ele in np_arr[ith].shape:
                ith_element_count *= ith_shape_ele
            
            element_count.append(ith_element_count)

        # 레이어 원소 개수로 정방 이차 배열의 한 변의 길이 구함
        square_NPs_side=[]
        for ith in range(len(np_arr)):
            square_NPs_side.append(math.ceil(math.sqrt((element_count[ith]))))

        # 레이어 피처 맵을 평탄화함.
        np_arr_flatten=[]
        for ith in range(len(np_arr)):
            np_arr_flatten.append(np_arr[ith].flatten())

        # x,y로 np_arr를 그리기 위한 이차 정방 배열 좌표를 만듦
        x=[];y=[]
        for ith in range(len(np_arr)):
            x_ith=[]; y_ith=[]
            for y_ in range(square_NPs_side[ith]):
                for x_ in range(square_NPs_side[ith]):
                    x_ith.append(x_)
                    y_ith.append(y_)
            # x_ith, y_ith를 np_arr[ith] 개수 만큼 자름
            x_ith = x_ith[:element_count[ith]]; y_ith = y_ith[:element_count[ith]]
            # x_ith을 x에 넣고 y_ith을 y에 넣음
            x.append(x_ith); y.append(y_ith)

        # plt의 subplot의 크기 지정
        square_NPs_row = (len(np_arr)-1 // self.square_NPs_column) + 1

        # 넘파이 배열을 5줄 씩 그리기
        plt.figure(figsize=self.square_NPs_figsize)
        for i in range(len(np_arr)):
            plt.subplot(square_NPs_row, self.square_NPs_column,i+1)
            plt.scatter(x=x[i], y=y[i], c=np_arr_flatten[i], cmap='jet')
        plt.colorbar()

    def show_eval_infos(self):
        self.set_square_NPs_infos([20,10])
        for eval_name in self.eval_names:
            print(f'In eval/{eval_name}')
            # 배열에 넘파이 배열을 넣어서 show_square_NPs에 전달해야 하므로
            # 배열에 넘파이 배열을 넣어서 show_square_NPs에 전달함
            print(f'self.eval_U[{eval_name}]', f'self.is_eval_FMD[{eval_name}]', f'self.fmds[{eval_name}]');
            self.show_square_NPs([self.eval_U[eval_name], self.is_eval_FMD[eval_name], self.fmds[eval_name]], color_map=False) # 배열에 담아 포인터가 아닌 복사본을 인자로 넘겨줌
            print()

    def show_fmd_right_ratio_graph(self):
        # subplot 모양을 지정하기 위한 row, column
        column = 5; row = (len(self.eval_names)-1 // column) + 1
        # figure 크기를 지정함
        plt.figure(figsize=[30,20])

        for i, eval_name in enumerate(self.eval_names):
            # fmdX는 그래프에서 X축(fmd)에 해당하는 부분, RRY는 그래프에서 Y축(Right Ratio)에 해당하는 부분
            fmdX = []; RRY = []
            fmds = copy.deepcopy(self.fmds[eval_name])
            fmd_min = fmds.min(); fmd_max = fmds.max() # fmds 최소값, 최대값 찾음.
            fmd_slice = (self.eval_K[eval_name] // 10) + 1
            fmd_interval = (fmd_max - fmd_min) / fmd_slice

            # 각 interval을 순회하며 fmd(interval의 중앙값)과 RR(정분류 비율)을 구함
            for s in range(fmd_slice):
                # 각 interval의 중앙값으로 fmd에 할당
                interval_min = fmd_min + s*fmd_interval; interval_max = fmd_min + (s+1)*fmd_interval
                fmd = (interval_min + interval_max) / 2
                # 각 interval의 정분류 비율을 RR에 할당
                upper_than_interval_min = 0
                lower_than_interval_max = 0
                if s != fmd_slice-1:
                    upper_than_interval_min = fmds >= interval_min
                    lower_than_interval_max = fmds < interval_max
                else:
                    upper_than_interval_min = fmds >= interval_min
                    lower_than_interval_max = fmds <= interval_max
                
                interval_values_maker = np.logical_and(upper_than_interval_min, lower_than_interval_max)
                is_right_interval_values = self.eval_U[eval_name][interval_values_maker]
                R = len(np.nonzero(is_right_interval_values)[0]); W  = len(is_right_interval_values) - R
                
                if R+W == 0:
                    RR = -1
                else:
                    RR = R / (R + W)

                fmdX.append(fmd)
                RRY.append(RR)
            
            ones = [1 for i in range(fmd_slice)]

            plt.subplot(row, column, i+1)
            plt.scatter(x=fmdX, y=RRY, s=ones)
            plt.ylim(-0.1, 1.1)
        
        plt.show()

    def show_eval_venn_diagrams(self):
        def show_eval_venn_diagram(eval_name, TP, FP, TN, FN):
            plt.figure(figsize=(10,8))
            # eval_U, 직사각형
            box_left = np.array([[-100, i] for i in range(-50, 50+1)])
            box_right = np.array([[100, i] for i in range(-50, 50+1)])
            box_top = np.array([[i, 50] for i in range(-99, 99+1)])
            box_bottom = np.array([[i, -50] for i in range(-99, 99+1)])

            box = np.append(box_left, box_right, axis=0)
            box = np.append(box, box_top, axis=0)
            box = np.append(box, box_bottom, axis=0)
            box = box.T

            box_x = box[0]
            box_y = box[1]

            plt.plot(box_x, box_y, 'go')
            # eval_FMD, 타원
            ellipse = [[-75,0]]
            x_range = [-75 + i*(75*2/999) for i in range(999+1)]
            for i in x_range:
                ellipse.append([i,  np.sqrt((30**2)*(1-(i**2)/(75**2)))])
            ellipse.append([75,0])
            for i in x_range:
                ellipse.append([i, -np.sqrt((30**2)*(1-(i**2)/(75**2)))])
            ellipse = np.array(ellipse)
            ellipse = ellipse.T

            ellipse_x = ellipse[0]
            ellipse_y = ellipse[1]

            plt.plot(ellipse_x, ellipse_y, 'ko')
            # 정분류, 직사각형
            box_left = np.array([[-97, i] for i in range(-48, 48+1)])
            box_right = np.array([[-2, i] for i in range(-48, 48+1)])
            box_top = np.array([[i, 48] for i in range(-97+1, -2-1+1)])
            box_bottom = np.array([[i, -48] for i in range(-97+1, -2-1+1)])

            box = np.append(box_left, box_right, axis=0)
            box = np.append(box, box_top, axis=0)
            box = np.append(box, box_bottom, axis=0)
            box = box.T

            box_x = box[0]
            box_y = box[1]

            plt.plot(box_x, box_y, 'bo')
            # 오분류, 직사각형
            box_left = np.array([[2, i] for i in range(-48, 48+1)])
            box_right = np.array([[97, i] for i in range(-48, 48+1)])
            box_top = np.array([[i, 48] for i in range(2+1, 97-1+1)])
            box_bottom = np.array([[i, -48] for i in range(2+1, 97-1+1)])

            box = np.append(box_left, box_right, axis=0)
            box = np.append(box, box_top, axis=0)
            box = np.append(box, box_bottom, axis=0)
            box = box.T

            box_x = box[0]
            box_y = box[1]

            plt.plot(box_x, box_y, 'ro')
            # 집합 표시를 위한 텍스트 넣기
            plt.text(x=73, y=40, s="eval_U", fontdict={'color': 'green','size': 16})
            plt.text(x=13, y=20, s="eval_FMD", fontdict={'color': 'black','size': 16})
            plt.text(x=50, y=57, s="Wrong", fontdict={'color': 'red','size': 16})
            plt.text(x=-50, y=57, s="Right", fontdict={'color': 'blue','size': 16})
            # TP, FN, TN, FP 에 대한 숫자 넣기
            plt.text(x=-75, y=40, s=f"TP: {TP}", fontdict={'color': 'purple','size': 16})
            plt.text(x=-50, y=0, s=f"FN: {FN}", fontdict={'color': 'purple','size': 16})
            plt.text(x=25, y=0, s=f"TN: {TN}", fontdict={'color': 'purple','size': 16})
            plt.text(x=25, y=40, s=f"FP: {FP}", fontdict={'color': 'purple','size': 16})
            plt.text(x=25, y=-40, s=f"{eval_name}", fontdict={'color': 'orange','size': 16})
            plt.show()
        for eval_name in self.eval_names:
            show_eval_venn_diagram(eval_name, self.TP[eval_name], self.FP[eval_name], self.TN[eval_name], self.FN[eval_name])
    
    def show_efficience_and_FMD_ratio(self):
        for eval_name in self.eval_names:
            print(f"In {eval_name}")
            print(f"[오분류 비율 U = WR_U(N)] = {self.N[eval_name]}") # 오분류 비율 U = WR_U(N)
            print(f"[오분류 비율 FMD = WR_FMD(NPV)] = {self.NPV[eval_name]}") # 오분류 비율 FMD = WR_FMD(NPV)
            print()
            print(f"[긍정적 정밀성(TPR)] = {self.TPR[eval_name]}") # 정분류 정밀성(TPR)
            print(f"[부정적 정밀성(TNR)] = {self.TNR[eval_name]}")# 오분류 정밀성(TNR)
            print('-'*30) # 구분선
            print(f"[정밀성 합] = {self.TPR[eval_name] + self.TNR[eval_name]}")# 정밀성 합
            print()
            # FMD ratio
            eval_U_size = len(self.eval_U[eval_name])
            eval_FMD_size = len(np.nonzero(self.eval_U[eval_name][self.is_eval_FMD[eval_name]])[0])
            print(f"[FMD 비율(|eval_FMD|/|eval_U|)] = {eval_FMD_size/eval_U_size}")
            print('-'*30) # 구분선

    def save(self, model_name):
        # os.system(f'touch ./instances/{model_name}.pickle')
        # 인자로 지정된 경로와 이름으로 파일 저장
        with open(f"{self.root_dir}/instances/{model_name}.pickle", "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self, model_name):
        # 인자로 지정된 경로와 이름으로 파일 불러오기
        with open(f"{self.root_dir}/instances/{model_name}.pickle", "rb" ) as f:
            return copy.deepcopy(pickle.load(f))

    def create_practice(self):
        '''
        아마도 나중에 adapter를 만들 때 사용될 수 있을 것 같음.
        '''

        # 연습 데이터 랜덤 시드 설정
        random_seed_id = 42
        np.random.seed(random_seed_id)
        # 연습 데이터 범위 설정
        '''
        연습 데이터 범위가 0~1로로 고정, np.random.rand(or random)으로 바로 구현 가능
        '''
        # random_start = 1; random_end = 2

        # 디렉토리 설정
        root_dir = "practice"
        origin_dir = f"{root_dir}/origin"
        eval_dir = f"{root_dir}/eval"
        # 디렉토리 생성
        os.system(f"mkdir {root_dir}")
        os.system(f"mkdir {origin_dir}")
        os.system(f"mkdir {eval_dir}")

        # origin 데이터 종류 적기
        origin_names="train rvalid wvalid"
        # origin_K 각 데이터 종류마다의 개수
        origin_K="10000 2000 4000"
        # eval 데이터 종류 적기
        eval_names="rotation_90 brightness_10 whiteness_10"
        # eval_K 각 데이터 종류마다의 개수
        eval_K="12000 4000 6000"
        # L: 레이어의 개수, shape: 레이어의 모양
        L="3";
        shape = "12 12" + "\n"
        shape += "12 2 3 4" + "\n"
        shape += "24 8"

        # data_infos.txt 파일로 저장
        # origin 데이터 종류는 train rvalid wvalid로 이미 정해져 있으므로 파일에 저장하지 않음
        os.system(f"touch {root_dir}/data_infos.txt")
        data_infos_txt = open(f"{root_dir}/data_infos.txt", 'w')
        data_infos_txt.write(origin_K + "\n")
        data_infos_txt.write(eval_names + "\n")
        data_infos_txt.write(eval_K + "\n")
        data_infos_txt.write(L + "\n")
        data_infos_txt.write(shape)
        data_infos_txt.close()

        # 자료구조를 바꾸어 for문 활용을 쉽게 하기
        origin_names = origin_names.split()
        origin_K = list(map(int,origin_K.split()))
        origin_K_list = origin_K
        origin_K = {}
        for i in range(len(origin_names)):
            origin_K[origin_names[i]] = origin_K_list[i]
        
        eval_names = eval_names.split()
        eval_K = list(map(int,eval_K.split()))
        eval_K_list = eval_K
        eval_K = {}
        for i in range(len(eval_names)):
            eval_K[eval_names[i]] = eval_K_list[i]

        L=int(L)
        
        shape = shape.split('\n')
        for ith in range(len(shape)):
            shape[ith] = list(map(int,shape[ith].split()))

        # origin 연습 데이터 생성
        for origin_name in origin_names:
            os.system(f"mkdir {origin_dir}/{origin_name}")
            # datas: 데이터들을 최대 1000개씩 담음.
            # file_index: 0 ~ K-1 까지 파일을 1000개씩 담을건데 0에서부터 시작해서 1000개마다 1 증가함
            datas = []; file_index = 0

            for k in range(origin_K[origin_name]):
                # data: 데이터는 레이어들을 담고 있음
                data = []
                # data에 레이어들을 담은 후
                for l in range(L):
                    lth_layer = np.random.random(shape[l])
                    data.append(lth_layer)
                # datas에 레이어들을 담은 data를 담음
                datas.append(data)

                # 1000개마다 데이터가 저장됨
                if len(datas) == 1000:
                    # file_index번째 datas를 저장
                    with open(f'{origin_dir}/{origin_name}/{origin_name}_{file_index}.pickle', 'wb') as f:
                        pickle.dump(datas, f, pickle.HIGHEST_PROTOCOL)
                    # datas의 데이터를 램에서 제거
                    del datas
                    # datas를 빈 배열로 만듦
                    datas = []
                    # 파일 인덱스 1 증가
                    file_index += 1

            # datas의 데이터가 남아있다면, 1~999개의 데이터가 남아있다면 그 데이터도 마저 저장함
            if len(datas) != 0:
                with open(f'{origin_dir}/{origin_name}/{origin_name}_{file_index}.pickle', 'wb') as f:
                    pickle.dump(datas, f, pickle.HIGHEST_PROTOCOL)

            # 훈련 데이터의 경우
            # 정분류(1), 오분류(0) 여부를 알려주는 넘파이 배열을 파일로 저장함
            if origin_name == "train":
                origin_result = np.random.randint(0,2, origin_K[origin_name])
                # 1 -> True, 0 -> False로 변경
                origin_result = np.array(origin_result, dtype="bool")
                np.save(f"{origin_dir}/{origin_name}/{origin_name}_result.npy", origin_result)

        # eval 연습 데이터 생성
        for eval_name in eval_names:
            os.system(f"mkdir {eval_dir}/{eval_name}")
            # datas: 데이터들을 최대 1000개씩 담음.
            # file_index: 0 ~ K-1 까지 파일을 1000개씩 담을건데 0에서부터 시작해서 1000개마다 1 증가함
            datas = []; file_index = 0

            for k in range(eval_K[eval_name]):
                # data: 데이터는 레이어들을 담고 있음
                data = []
                # data에 레이어들을 담은 후
                for l in range(L):
                    lth_layer = np.random.random(shape[l])
                    data.append(lth_layer)
                # datas에 레이어들을 담은 data를 담음
                datas.append(data)

                # 1000개마다 데이터가 저장됨
                if len(datas) == 1000:
                    # file_index번째 datas를 저장
                    with open(f'{eval_dir}/{eval_name}/{eval_name}_{file_index}.pickle', 'wb') as f:
                        pickle.dump(datas, f, pickle.HIGHEST_PROTOCOL)
                    # datas의 데이터를 램에서 제거
                    del datas
                    # datas를 빈 배열로 만듦
                    datas = []
                    # 파일 인덱스 1 증가
                    file_index += 1

            # datas의 데이터가 남아있다면, 1~999개의 데이터가 남아있다면 그 데이터도 마저 저장함
            if len(datas) != 0:
                with open(f'{eval_dir}/{eval_name}/{eval_name}_{file_index}.pickle', 'wb') as f:
                    pickle.dump(datas, f, pickle.HIGHEST_PROTOCOL)

            # 정분류(1), 오분류(0) 여부를 알려주는 넘파이 배열을 파일로 저장함
            eval_result = np.random.randint(0,2, eval_K[eval_name])
            # 1 -> True, 0 -> False로 변경
            eval_result = np.array(eval_result, dtype="bool")
            np.save(f"{eval_dir}/{eval_name}/{eval_name}_result.npy", eval_result)
