# import cupy as np # CUDA를 지원하는 컴퓨터라면 CUDA 버전에 맞게 cupy를 설치하여 cupy 임포트
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import pickle
import copy
import itertools

class FMD():
    square_NPs_figsize=[60, 60]; square_NPs_column=7
    '''square_NPs_infos
    square_NPs_figsize: show_square_NPs에서 그려지는 그래프의 figsize를 조절한다.
    square_NPs_columns: show_square_NPs에서 그려지는 그래프의 column을 조절한다.
    '''
    root_dir=""; origin_dir=""; train_dir=""; rvalid_dir=""; wvalid_dir=""; eval_dir=""
    ''' dir_infos
    root_dir:   관련된 데이터가 모두 저장되어 있는 디렉토리 경로
    origin_dir: train, rvalid, wvalid가 있는 디렉토리
    eval_dir:   거리를 잴 데이터가 있는 디렉토리
    train_dir:  검증 데이터가 아닌 정분류된 훈련 피처 맵이 들어있는 디렉토리
    rvalid_dir: 정분류 검증 피처 맵이 들어있는 디렉토리
    wvalid_dir: 오분류 검증 피처 맵이 들어있는 디렉토리
    '''
    origin_names=["train", "rvalid", "wvalid"]; origin_K={}; eval_names=[]; eval_K={}
    L=0; shape=[]; FMP_count=0; eval_U={}
    ''' data_infos
    origin_names:          근원 데이터 타입에 대한 이름
    eval_names:            평가 타입에 대한 이름
    origin_K:              근원 피처 맵의 개수
    eval_K:                평가 피처 맵의 개수
    L:                     레이어의 개수(0~L-1는 각 레이어의 인덱스)
    shape:                 레이어의 넘파이 모양
    FMP_count:             피처 맵 패키지에 저장된 넘파이 개수
    eval_U:                평가 데이터의 정분류(True), 오분류(False) 유무를 True, False로 담는다
    '''
    TFM_repre={}; RFM_repre={}; WFM_repre={}
    ''' fixed_FM_repres
    TFM_repre: 훈련 대표 피처 맵(베이스 피처 맵)
    RFM_repre: 정분류 대표 피처 맵
    WFM_repre: 오분류 대표 피처 맵
    '''
    alpha_min={}; alpha_max={}; rmw_min=[]; rmw_max=[]
    ''' fixed_alpha_infos
    alpha_min, alpha_max: 각 레이어에서의 alpha가 가질 수 있는 최소값 최대값을 나타낸 것이다.
    rmw_min, rmw_max: 각 레이어에서의 rmw가 가질 수 있는 최소값 최대값을 나타낸 것이다.
    '''
    norm_min=0; norm_max=1
    ''' fixed_layer_infos
    norm_min, norm_max는 레이어 피처 맵 거리를 구하기 전에 정규화할 범위의 최소 최대를 나타낸다.
    '''
    FM_repre_MHP=[]; alpha_MHP=[]; DAM_MHP=[]; lfmd_MHP=[]; W_MHP=[]; fmdc_MHP=[]
    ''' [MHP = meta hyper parameter]
    각 하이퍼 파라미터의 경우의 수들의 곱만큼 평가한다.
    FM_repre_MHP: 평가하고 싶은 FM_repre 종류 모두를 지정한다. FM_repre_MHP는 'mean', 'min', 'max'가 있다.
    alpha_MHP:    rmw_max로 할지, 특정 alpha 값들로 할지 지정할 수 있다. Ex. alpha_MHP = [['rmw', 100], [1,2,3,4,5,6]]
    DAM_MHP:      평가하고 싶은 DAM 종류 모두를 지정한다. DAM_MHP은 'and', 'or', 'wfm', 'all'이 있다.
    W_MHP:        평가하고 싶은 W 종류 모두를 지정한다. W_MHP은 'C'=constant, 'I'=increasing 있다.
    lfmd_MHP:     평가하고 싶은 lfmd 종류 모두를 지정한다. lfmd_MHP은 'se_lfmd', 'Ln_lfmd'가 있다.
    fmdc_MHP:     평가하고 싶은 fmdc 종류 모두를 지정한다. fmdc_MHP는 'rvalid_fmds_max', 'rvalid_fmds_average', 'rvalid_fmds_middle', 'wvalid_fmds_min', 'wvalid_fmds_average', 'wvalid_fmds_middle', 'rvalid_fmds_max_wvalid_fmds_min_average'(M: Max, m: min, A:Average, mid: middle)가 있다.
    '''

    INSTs={}; INST_names=[]
    ''' INST_infos
    INSTs:      하이퍼 파라미터들이 INST 이름이고 하이퍼 파라미터마다 변하는 속성을 담는 딕셔너리
    INST_names: 하이퍼 파라미터들로 만들어진 INST 이름들의 모임
    '''

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

    def set_data_infos(self):
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
        self.shape = []
        for l in range(self.L):
            shape_l = list(map(int,data_infos_str_list[4+l].split()))
            self.shape.append(shape_l)
        # * 4+Lth: FMP_count
        self.FMP_count = int(data_infos_str_list[4+self.L])
        # * root dir의 data_infos.txt 닫기
        data_infos.close()

        # * 이것도 초기 데이터에 포함됨.
        for eval_name in self.eval_names:
            self.eval_U[eval_name] = np.load(f'{self.eval_dir}/{eval_name}/{eval_name}_eval_U.npy')

    def set_FM_repres(self):
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

            # * OFM_repre의 min, mean, max 리스트 생성
            OFM_repre['FM_min']=[]; OFM_repre['FM_mean']=[]; OFM_repre['FM_max']=[]

            # * 각 레이어의 피처 맵을 0으로 초기화하여 생성
            for l in range(L):
                OFM_repre_zeros_l = np.zeros(shape[l])
                OFM_repre['FM_min'].append(OFM_repre_zeros_l)
                OFM_repre['FM_mean'].append(OFM_repre_zeros_l)
                OFM_repre['FM_max'].append(OFM_repre_zeros_l)

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
                OFM_repre['FM_min'][l] = OFM_repre['FM_min'][l] + OFM_k_l
                OFM_repre['FM_mean'][l] = OFM_repre['FM_mean'][l] + OFM_k_l
                OFM_repre['FM_max'][l] = OFM_repre['FM_max'][l] + OFM_k_l

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
                    OFM_repre['FM_mean'][l] = (OFM_repre['FM_mean'][l]*k + OFM_k_l)/(k+1)
                    OFM_repre_min_l_mask = OFM_repre['FM_min'][l] > OFM_k_l
                    OFM_repre_max_l_mask = OFM_repre['FM_max'][l] < OFM_k_l
                    np.place(OFM_repre['FM_min'][l], OFM_repre_min_l_mask, OFM_k_l)
                    np.place(OFM_repre['FM_max'][l], OFM_repre_max_l_mask, OFM_k_l)

            # OFM을 모두 순회한 후 OFMP_k의 기억공간을 램에서 제거
            del OFMP_k

        # * 훈련, 정분류 테스트, 오분류 테스트 데이터에 대한 FM_repre을 구함
        set_FM_repre(train); set_FM_repre(rvalid); set_FM_repre(wvalid)

    def set_alpha_rmw_min_max(self):
        # * 1. 새로운 데이터 입력을 받기 전에 0으로 초기화
        # alpha_min, alpha_max
        self.alpha_min['FM_min'] = []; self.alpha_max['FM_min'] = []
        self.alpha_min['FM_mean'] = []; self.alpha_max['FM_mean'] = []
        self.alpha_min['FM_max'] = []; self.alpha_max['FM_max'] = []
        # rmw_min, rmw_max
        self.rmw_min=[]; self.rmw_max=[]

        # * 2. alpha_min, alpha_max 구하기
        for l in range(self.L):
            self.alpha_min['FM_min'].append(np.array([self.RFM_repre['FM_min'][l].min(),
                                                      self.TFM_repre['FM_min'][l].min(),
                                                      self.WFM_repre['FM_min'][l].min()]).min())
            self.alpha_max['FM_min'].append(np.array([self.RFM_repre['FM_min'][l].max(),
                                                      self.TFM_repre['FM_min'][l].max(),
                                                      self.WFM_repre['FM_min'][l].max()]).max())

            self.alpha_min['FM_mean'].append(np.array([self.RFM_repre['FM_mean'][l].min(),
                                                       self.TFM_repre['FM_mean'][l].min(),
                                                       self.WFM_repre['FM_mean'][l].min()]).min())
            self.alpha_max['FM_mean'].append(np.array([self.RFM_repre['FM_mean'][l].max(),
                                                       self.TFM_repre['FM_mean'][l].max(),
                                                       self.WFM_repre['FM_mean'][l].max()]).max())

            self.alpha_min['FM_max'].append(np.array([self.RFM_repre['FM_max'][l].min(),
                                                      self.TFM_repre['FM_max'][l].min(),
                                                      self.WFM_repre['FM_max'][l].min()]).min())
            self.alpha_max['FM_max'].append(np.array([self.RFM_repre['FM_max'][l].max(),
                                                      self.TFM_repre['FM_max'][l].max(),
                                                      self.WFM_repre['FM_max'][l].max()]).max())
        # alpha_min, alpha_max 넘파이로 변경
        self.alpha_min['FM_min'] = np.array(self.alpha_min['FM_min'])
        self.alpha_max['FM_min'] = np.array(self.alpha_max['FM_min'])

        self.alpha_min['FM_mean'] = np.array(self.alpha_min['FM_mean'])
        self.alpha_max['FM_mean'] = np.array(self.alpha_max['FM_mean'])

        self.alpha_min['FM_max'] = np.array(self.alpha_min['FM_max'])
        self.alpha_max['FM_max'] = np.array(self.alpha_max['FM_max'])

        # * 3. rmw_min, rmw_max 구하기
        for l in range(self.L):
            shape_l = self.shape[l]
            shape_l_count = 1
            for shape_l_ele in shape_l:
                shape_l_count *= shape_l_ele
            self.rmw_min.append(-shape_l_count); self.rmw_max.append(shape_l_count)
        # rmw_min, rmw_max 넘파이로 변경
        self.rmw_min = np.array(self.rmw_min); self.rmw_max = np.array(self.rmw_max)

    def set_MHP(self, FM_repre_MHP=['FM_mean'], alpha_MHP=[['rmw_max', 1000]], DAM_MHP=['all'], W_MHP=['C'], lfmd_MHP=['se_lfmd'], fmdc_MHP=['rvalid_fmds_average']):
        # * 램 용량을 초과하지 않도록 나중에 이 부분을 추가함.
        limited_number = 1219 # instance 하나 당 820KB일 때, limited_number: instance들의 총량이 1GB가 되는 개수.
        number_of_all_case = len(FM_repre_MHP) * len(alpha_MHP) * len(DAM_MHP) * len(W_MHP) * len(lfmd_MHP) * len(fmdc_MHP)
        if number_of_all_case > limited_number:
            print(f'하이퍼 파라미터 경우의 수가 {limited_number}을 넘김')
            return
        elif number_of_all_case <= 0:
            print(f'하이퍼 파라미터 경우의 수가 0이하임')
            return
            
        # * MHP 초기화
        self.FM_repre_MHP = FM_repre_MHP; self.alpha_MHP = alpha_MHP; self.DAM_MHP = DAM_MHP
        self.W_MHP = W_MHP; self.lfmd_MHP = lfmd_MHP; self.fmdc_MHP = fmdc_MHP

    def init_INSTs(self):
        # * 1. alpha_MHP_str 생성
        # alpha_MHP 종류마다 문자열로 바꾸는 방식을 달리함.
        alpha_MHP_str = []
        for i in range(len(self.alpha_MHP)):
            # 'rmw_max'인 경우
            if self.alpha_MHP[i][0] == "rmw_max":
                alpha_MHP_str.append(str(i)+','+self.alpha_MHP[i][0]+','+str(self.alpha_MHP[i][1]))
            else:
                alpha_MHP_i_str = ''
                for ele_index, alpha_MHP_i_ele in enumerate(self.alpha_MHP[i]):
                    if ele_index == len(self.alpha_MHP[i]) - 1:
                        alpha_MHP_i_str += f"{alpha_MHP_i_ele: 0.4f}".strip()
                    else:
                        alpha_MHP_i_str += f"{alpha_MHP_i_ele: 0.4f}".strip() + str(',')

                alpha_MHP_str.append(str(i)+','+alpha_MHP_i_str)

        # * 2. INST_name 생성
        # INST_name을 리스트로 생성
        self.INST_names = list(itertools.product(self.FM_repre_MHP, alpha_MHP_str, self.DAM_MHP, self.lfmd_MHP, self.W_MHP, self.fmdc_MHP))
        # INST_name을 리스트에서 문자열로 바꿈
        for i, INST_name_list in enumerate(self.INST_names):
            # INST_name_list의 각 원소를 해당 HP 변수에 할당
            FM_repre_HP=INST_name_list[0]; alpha_HP_str=INST_name_list[1]; DAM_HP=INST_name_list[2]; lfmd_HP=INST_name_list[3]; W_HP=INST_name_list[4]; fmdc_HP=INST_name_list[5]
            # HP 변수들을 이용해 INST_name_list를 INST_name_str로 바꿈
            INST_name_str=FM_repre_HP+' '+alpha_HP_str+' '+DAM_HP+' '+lfmd_HP+' '+W_HP+' '+fmdc_HP
            # INST_name_str를 INST_name에 할당
            self.INST_names[i] = INST_name_str

        # * 3. 모든 INST 초기화
        #  INST마다 하이퍼 파라미터를 초기화하고
        #  나머지는 빈 배열, 빈 딕셔너리, 초기값으로 초기화함.
        for INST_name in self.INST_names:
            # * 3.1. INST_name으로부터 HP 변수를 초기화
            INST_name_list=INST_name.split()
            FM_repre_HP=INST_name_list[0]; alpha_HP_str=INST_name_list[1]; DAM_HP=INST_name_list[2]
            lfmd_HP=INST_name_list[3]; W_HP=INST_name_list[4]; fmdc_HP=INST_name_list[5]

            # * 3.2. INST를 딕셔너리로 초기화
            self.INSTs[INST_name] = {}

            # * 3.3. FM_repre_infos
            self.INSTs[INST_name]['FM_repre_HP']=FM_repre_HP
            ''' FM_repre_infos
            FM_repre_HP: FM_repre를 어떤 것으로 선택할지 정하는 HP, FM_min, FM_mean, FM_max가 있다.
            '''

            # * 3.4. alpha_infos
            self.INSTs[INST_name]['alpha_slice']=0
            self.INSTs[INST_name]['alpha']=[]
            self.INSTs[INST_name]['alpha_percent']=[]
            self.INSTs[INST_name]['rmw']=[]
            self.INSTs[INST_name]['rmw_percent']=[]

            alpha_HP_list = alpha_HP_str.split(',')
            # [rmw_max]: alpha_slice, alpha 초기화
            # alpha_HP가 rmw_max 방식이면 alpha_slice에 양수를 할당하고
            # alpha에 L 크기 만큼 -1로 초기화한다.
            if alpha_HP_list[1] == 'rmw_max':
                alpha_MHP_index = int(alpha_HP_list[0])
                self.INSTs[INST_name]['alpha_slice']=self.alpha_MHP[alpha_MHP_index][1]
                for l in range(self.L):
                    self.INSTs[INST_name]['alpha'].append(-1)
            # [특정 alpha 값들을 선택하는 방식]: alpha 초기화
            # alpha_HP가 특정 alpha 값들을 선택하는 방식이면 alpha_slice에 아무것도 할당하지 않는다.(0을 유지한다.)
            # 대신 alpha_HP의 특정 값들을 alpha에 할당한다.
            else:
                alpha_MHP_index = int(alpha_HP_list[0])
                self.INSTs[INST_name]['alpha'] = self.alpha_MHP[alpha_MHP_index]

            # alpha를 넘파이로 변경
            self.INSTs[INST_name]['alpha'] = np.array(self.INSTs[INST_name]['alpha'], dtype='float')
            ''' alpha_infos
            alpha_slice:   alpha_min에서 alpha_max로 몇 번의의 간격으로 도착할지 알려주는 변수임.
            alpha:         거리 계산을 위한 인덱스를 고르기 위해 필요한 변수이다.
            alpha_percent: alpha_min, alpha_max 사이에 alpha가 위치한 곳이다.
            rmw:           훈련과 정분류이 비슷하고 훈련과 오분류가 비슷하지 않을수록 값이 커진다.
            rmw_percent:   rmw_min, rmw_max 사이에 rmw가 위치한 곳이다.
            '''

            # * 3.5. AMs
            self.INSTs[INST_name]['TAM']=[]; self.INSTs[INST_name]['RAM']=[]; self.INSTs[INST_name]['WAM']=[]
            ''' AMs
            TAM: 훈련 활성화 피처 맵
            RAM: 정분류 활성화 피처 맵
            WAM: 오분류 활성화 피처 맵
            '''

            # * 3.6. DAM_infos
            self.INSTs[INST_name]['DAM_indexes']=[]; self.INSTs[INST_name]['DAM']=[]; self.INSTs[INST_name]['DAM_HP']=DAM_HP
            self.INSTs[INST_name]['DAM_error_flag']=[]
            ''' DAM_infos
            DAM_indexes:    나중에 거리 계산할 때 쓰이는 다차원 인덱스들의 집합이다. 각 원소는 피처 맵의 한 원소의 인덱스를 나타낸다.
                            각 레이어마다 튜플들 세트가 있어야 함. np.array의 item 메소드를 사용할 것이기 때문
            DAM:            거리 활성화 맵, DAM는 거리 계산을 위한 인덱스만 활성된 맵이다.
            DAM_HP:         DAM를 고르는 방법을 알려줌.
            DAM_error_flag: 예외 처리된 경우. 0은 예외 처리되지 않음. 1은 예외 처리되어 WFM 방식을 택함. 2는 예외 처리되어 모든 인덱스를 택함.
            '''

            # * 3.7. layer_infos
            self.INSTs[INST_name]['W']=[]
            if W_HP == 'C':
                for l in range(self.L):
                    self.INSTs[INST_name]['W'].append(1/self.L)
            elif W_HP == 'I':
                for l in range(self.L):
                    self.INSTs[INST_name]['W'].append((l+1)*(2/(self.L*(self.L+1))))
            self.INSTs[INST_name]['lfmd_HP']=lfmd_HP
            ''' layer_infos
            W:       각 레이어의 피처 맵에 곱할 weight 중요도이다
            lfmd_HP: 각 레이어에 대한 피처 맵을 구하는 방법을 저장한다.
            '''

            # * 3.8. fmdc_infos
            self.INSTs[INST_name]['fmdc']=-1; self.INSTs[INST_name]['fmdc_HP']=fmdc_HP; self.INSTs[INST_name]['HP_fmdcs']={}
            self.INSTs[INST_name]['rvalid_fmds']=[]; self.INSTs[INST_name]['wvalid_fmds']=[]
            fmdc_HPs = ['rvalid_fmds_max', 'rvalid_fmds_average', 'rvalid_fmds_middle', 'wvalid_fmds_min', 'wvalid_fmds_average', 'wvalid_fmds_middle', 'rvalid_fmds_max_wvalid_fmds_min_average']
            for fmdc_HP in fmdc_HPs:
                self.INSTs[INST_name]['HP_fmdcs'][fmdc_HP]=-1
            ''' fmdc_infos
            fmdc:     피처 맵 거리 기준으로 어떤 데이터가 나중에 오분류 될 거 같은지 판단함
            fmdc_HP:  피처 맵 거리 기준을 정하는 하이퍼 파라미터로 이 값에 따라 fmdc가 정해짐
            HP_fmdcs: 각 fmdc_HP에 대한 fmdc를 모두 저장한다
            rvalid_fmds:    정분류 검증 피처 맵 거리들을 모아둔 것
            wvalid_fmds:    오분류 검증 피처 맵 거리들을 모아둔 것
            '''

            # * 3.9. eval_infos
            # eval_U도 포함.
            self.INSTs[INST_name]['is_eval_FMD']={}; self.INSTs[INST_name]['eval_fmds']={}
            self.INSTs[INST_name]['TP']={}; self.INSTs[INST_name]['FN']={}; self.INSTs[INST_name]['TN']={}; self.INSTs[INST_name]['FP']={}
            self.INSTs[INST_name]['P']={}; self.INSTs[INST_name]['N']={}
            self.INSTs[INST_name]['TPR']={}; self.INSTs[INST_name]['TNR']={}; self.INSTs[INST_name]['PPV']={}; self.INSTs[INST_name]['NPV']={}
            self.INSTs[INST_name]['FNR']={}; self.INSTs[INST_name]['FPR']={}; self.INSTs[INST_name]['FDR']={}; self.INSTs[INST_name]['FOR']={}
            self.INSTs[INST_name]['fmdcs']={}; self.INSTs[INST_name]['TPRs']={}; self.INSTs[INST_name]['TNRs']={}; self.INSTs[INST_name]['AUC']={}
            for eval_name in self.eval_names:
                self.INSTs[INST_name]['is_eval_FMD'][eval_name]=[]; self.INSTs[INST_name]['eval_fmds'][eval_name]=[]
                self.INSTs[INST_name]['TP'][eval_name]=-1; self.INSTs[INST_name]['FN'][eval_name]=-1; self.INSTs[INST_name]['TN'][eval_name]=-1; self.INSTs[INST_name]['FP'][eval_name]=-1
                self.INSTs[INST_name]['P'][eval_name]=-1; self.INSTs[INST_name]['N'][eval_name]=-1
                self.INSTs[INST_name]['TPR'][eval_name]=-1; self.INSTs[INST_name]['TNR'][eval_name]=-1; self.INSTs[INST_name]['PPV'][eval_name]=-1; self.INSTs[INST_name]['NPV'][eval_name]=-1
                self.INSTs[INST_name]['FNR'][eval_name]=-1; self.INSTs[INST_name]['FPR'][eval_name]=-1; self.INSTs[INST_name]['FDR'][eval_name]=-1; self.INSTs[INST_name]['FOR'][eval_name]=-1
                self.INSTs[INST_name]['fmdcs'][eval_name]=[]; self.INSTs[INST_name]['TPRs'][eval_name]=[]; self.INSTs[INST_name]['TNRs'][eval_name]=[]; self.INSTs[INST_name]['AUC'][eval_name]=-1

            ''' eval_infos
            eval_U:         평가 데이터의 정분류(True), 오분류(False) 유무를 True, False로 담는다
            is_eval_FMD:    평가 데이터가 is_eval_FMD이면 True, eval_fmd가 아니면 False를 담는다.
            eval_fmds:      평가 데이터들의 fmd를 담는다.
            TP, FN, TN, FP: confusion matrix를 표현하기 위한 가장 기본적인 속성이다.
                            TPR, TNR, PPV, NPV, FNR, FPR, FDR, FOR 등을 표현할 수 있다.
            fmdcs:          eval_fmds를 density(=1000)만큼 조각 내서 구한 fmdc를 모두 담는다.
            TPRs, TNRs:     fmdcs의 원소마다 TPRs, TNRs의 원소를 구한다.
            AUC:            위 TPRs, TNRs을 가지고 만든 ROC curve의 아래 부분과 y=0, x=1 사이의 넓이이다.
            '''

    def set_AMs_and_alpha_infos_and_DAM_infos(self):
        for INST_name in self.INST_names:

           # * alpha_infos, AM_infos, DAM_info을 초기값으로 초기화
            # alpha_infos
            self.INSTs[INST_name]['rmw']=[]; self.INSTs[INST_name]['alpha_percent']=[]; self.INSTs[INST_name]['rmw_percent']=[]
            # AM_infos
            self.INSTs[INST_name]['TAM']=[]; self.INSTs[INST_name]['RAM']=[]; self.INSTs[INST_name]['WAM']=[]
            # DAM_infos
            self.INSTs[INST_name]['DAM_indexes']=[]; self.INSTs[INST_name]['DAM']=[]
            self.INSTs[INST_name]['DAM_error_flag']=[]

            # * 1. rmw에 L 크기 만큼 -987654321로 초기화한다. alpha_percent, rmw_percent에 L 크기 만큼 -1로 초기화한다.
            for l in range(self.L):
                self.INSTs[INST_name]['rmw'].append(-987654321)
            self.INSTs[INST_name]['alpha_percent'] = -np.ones(self.L, dtype="int32")
            self.INSTs[INST_name]['rmw_percent'] = -np.ones(self.L, dtype="int32")
            # * 2. AMs 0으로 초기화
            for l in range(self.L):
                TAM_l = np.zeros(self.shape[l])
                self.INSTs[INST_name]['TAM'].append(TAM_l)
            for l in range(self.L):
                RAM_l = np.zeros(self.shape[l])
                self.INSTs[INST_name]['RAM'].append(RAM_l)
            for l in range(self.L):
                WAM_l = np.zeros(self.shape[l])
                self.INSTs[INST_name]['WAM'].append(WAM_l)
            # * 3. DAM_error_flag를 0으로 초기화
            for l in range(self.L):
                self.INSTs[INST_name]['DAM_error_flag'].append(0)

            # * INST_name으로부터 alpha_HP_str 변수를 초기화
            INST_name_list = INST_name.split()
            alpha_HP_str = INST_name_list[1]
            alpha_HP_list = alpha_HP_str.split(',')
            FM_repre_HP = self.INSTs[INST_name]['FM_repre_HP']
            # * 1. rmw_max 방식일 경우, 문자열이 'rmw_max'일 경우
            if alpha_HP_list[1] == 'rmw_max':
                # rmw_max 방식일 경우 alpha_slice를 사용함.
                alpha_slice = self.INSTs[INST_name]['alpha_slice']
                # r-w가 최대가 되는 alpha, TAM, RAM, WAM을 찾음
                for l in range(self.L):
                    alpha_min_l = self.alpha_min[FM_repre_HP][l]; alpha_max_l = self.alpha_max[FM_repre_HP][l]
                    alpha_interval_l = (alpha_max_l - alpha_min_l)/alpha_slice
                    # range(a_slice_l+1) 해야 a_min_l 부터 a_max_l 까지 감
                    for alpha_interval_offset_l, alpha_l in enumerate([alpha_min_l + alpha_interval_l*alpha_interval_offset_l for alpha_interval_offset_l in range(alpha_slice+1)]):
                        TAM_l = np.array(self.TFM_repre[FM_repre_HP][l] > alpha_l)
                        RAM_l = np.array(self.RFM_repre[FM_repre_HP][l] > alpha_l)
                        WAM_l = np.array(self.WFM_repre[FM_repre_HP][l] > alpha_l)

                        TAM_l_xnor_RAM_l = np.logical_not(np.logical_xor(TAM_l, RAM_l))
                        TAM_l_xnor_WAM_l = np.logical_not(np.logical_xor(TAM_l, WAM_l))
                        # r_l은 TAM_l과 RAM_l이 얼마나 유사한지 보여준다.
                        # 즉, TAM_l과 RAM_l이 유사할수록 r_l 값이 커진다.
                        # w_l도 마찬가지이다.
                        r_l = len(np.where(TAM_l_xnor_RAM_l == True)[0])
                        w_l = len(np.where(TAM_l_xnor_WAM_l == True)[0])
                        # 처음에는 alpha_interval_offset_l 0으로 초기화하고
                        rmw_min_l = self.rmw_min[l]; rmw_max_l = self.rmw_max[l]
                        if alpha_interval_offset_l == 0:
                            self.INSTs[INST_name]['alpha'][l] = alpha_l
                            # alpha_min이 0%이고 alpha_max가 100%일 때 alpha_percent은 alpha의 위치가 몇 퍼센트인지 알려줌.
                            alpha_percent_l = np.round(((alpha_l - alpha_min_l) / (alpha_max_l - alpha_min_l)) * 100)
                            self.INSTs[INST_name]['alpha_percent'][l] = alpha_percent_l
                            self.INSTs[INST_name]['rmw'][l] = r_l - w_l
                            # rmw_min이 0%이고 rmw_max가 100%일 때 rmw_percent은 rmw의 위치가 몇 퍼센트인지 알려줌.
                            rmw_percent_l = np.round(((self.INSTs[INST_name]['rmw'][l] - rmw_min_l) / (rmw_max_l - rmw_min_l)) * 100)
                            self.INSTs[INST_name]['rmw_percent'][l] = rmw_percent_l
                            self.INSTs[INST_name]['TAM'][l] = TAM_l
                            self.INSTs[INST_name]['RAM'][l] = RAM_l
                            self.INSTs[INST_name]['WAM'][l] = WAM_l
                        # r-w가 이전의 r-w보다 클 때, 즉, TAM과 RAM이 더 유사해지거나 TAM과 WAM이 더 다를 때
                        # alpha, r-w, TAM, RAM, WAM를 최신화함.
                        elif alpha_interval_offset_l > 0 and r_l - w_l > self.INSTs[INST_name]['rmw'][l]:
                            self.INSTs[INST_name]['alpha'][l] = alpha_l
                            # alpha_min이 0%이고 alpha_max가 100%일 때 alpha_percent은 alpha의 위치가 몇 퍼센트인지 알려줌.
                            alpha_percent_l = np.round(((alpha_l - alpha_min_l) / (alpha_max_l - alpha_min_l)) * 100)
                            self.INSTs[INST_name]['alpha_percent'][l] = alpha_percent_l
                            self.INSTs[INST_name]['rmw'][l] = r_l - w_l
                            # rmw_min이 0%이고 rmw_max가 100%일 때 rmw_percent은 rmw의 위치가 몇 퍼센트인지 알려줌.
                            rmw_percent_l = np.round(((self.INSTs[INST_name]['rmw'][l] - rmw_min_l) / (rmw_max_l - rmw_min_l)) * 100)
                            self.INSTs[INST_name]['rmw_percent'][l] = rmw_percent_l
                            self.INSTs[INST_name]['TAM'][l] = TAM_l
                            self.INSTs[INST_name]['RAM'][l] = RAM_l
                            self.INSTs[INST_name]['WAM'][l] = WAM_l
            # * 2. 특정 alpha 값들을 선택하는 방식일 경우, 문자열이 '숫자'일 경우
            else:
                # 특정 alpha 값들을 선택하는 방식일 경우 alpha를 바로 사용함.
                alpha = self.INSTs[INST_name]['alpha']
                # alpha_percent을 구함
                alpha_min = self.alpha_min[FM_repre_HP]; alpha_max = self.alpha_max[FM_repre_HP]
                # alpha_min이 0%이고 alpha_max가 100%일 때 alpha_percent은 alpha의 위치가 몇 퍼센트인지 알려줌.
                alpha_percent = np.round(((alpha - alpha_min) / (alpha_max - alpha_min)) * 100)
                alpha_percent = np.array(alpha_percent, dtype="int32")
                self.INSTs[INST_name]['alpha_percent'] = alpha_percent
                # OAM 및 rmw 구하기
                for l in range(self.L):

                    self.INSTs[INST_name]['TAM'][l] = np.array(self.TFM_repre[FM_repre_HP][l] > alpha[l])
                    self.INSTs[INST_name]['RAM'][l] = np.array(self.RFM_repre[FM_repre_HP][l] > alpha[l])
                    self.INSTs[INST_name]['WAM'][l] = np.array(self.WFM_repre[FM_repre_HP][l] > alpha[l])

                    TAM_l_xnor_RAM_l = np.logical_not(np.logical_xor(self.INSTs[INST_name]['TAM'][l], self.INSTs[INST_name]['RAM'][l]))
                    TAM_l_xnor_WAM_l = np.logical_not(np.logical_xor(self.INSTs[INST_name]['TAM'][l], self.INSTs[INST_name]['WAM'][l]))

                    r_l = len(np.where(TAM_l_xnor_RAM_l == True)[0])
                    w_l = len(np.where(TAM_l_xnor_WAM_l == True)[0])

                    # rmw 구하기
                    rmw_min_l = self.rmw_min[l]; rmw_max_l = self.rmw_max[l]
                    self.INSTs[INST_name]['rmw'][l] = (r_l - w_l)
                    # rmw_min이 0%이고 rmw_max가 100%일 때 rmw_percent은 rmw의 위치가 몇 퍼센트인지 알려줌.
                    rmw_percent_l = np.round(((self.INSTs[INST_name]['rmw'][l] - rmw_min_l) / (rmw_max_l - rmw_min_l)) * 100)
                    self.INSTs[INST_name]['rmw_percent'][l] = rmw_percent_l

            # * 1. DAM_HP 방식으로 DAM_infos 구하기
            TAM = self.INSTs[INST_name]['TAM']; RAM = self.INSTs[INST_name]['RAM']; WAM = self.INSTs[INST_name]['WAM']
            DAM_HP = self.INSTs[INST_name]['DAM_HP']
            # DAM를 WAM로 초기화
            self.INSTs[INST_name]['DAM'] = copy.deepcopy(WAM)
            # DAM 구하기
            if DAM_HP == "and":
                for l in range(self.L):
                    TAM_l_and_RAM_l = np.logical_and(TAM[l], RAM[l])
                    np.place(self.INSTs[INST_name]['DAM'][l], TAM_l_and_RAM_l, False)
            elif DAM_HP == "or":
                for l in range(self.L):
                    TAM_l_or_RAM_l = np.logical_or(TAM[l], RAM[l])
                    np.place(self.INSTs[INST_name]['DAM'][l], TAM_l_or_RAM_l, False)
            elif DAM_HP == "wfm":
                pass
            elif DAM_HP == "all":
                for l in range(self.L):
                    not_DAM_l = np.logical_not(self.INSTs[INST_name]['DAM'][l])
                    np.place(self.INSTs[INST_name]['DAM'][l], not_DAM_l, True)
            # * 2. 만약 DAM[l]의 원소가 모두 False라면 WAM[l]로 초기화함.
            for l in range(self.L):
                shape_l_size = 1
                for shaple_l_i in self.shape[l]:
                    shape_l_size *= shaple_l_i
                if len(np.where(self.INSTs[INST_name]['DAM'][l] == False)[0]) == shape_l_size:
                    self.INSTs[INST_name]['DAM_error_flag'][l] += 1 # error_flag 1 증가
                    DAM_l = WAM[l].copy()
                    self.INSTs[INST_name]['DAM'][l] = DAM_l
            # * 3. 그래도 DAM[l]의 원소가 모두 False라면 모든 인덱스를 True로 초기화함.
            for l in range(self.L):
                shape_l_size = 1
                for shaple_l_i in self.shape[l]:
                    shape_l_size *= shaple_l_i
                if len(np.where(self.INSTs[INST_name]['DAM'][l] == False)[0]) == shape_l_size:
                    self.INSTs[INST_name]['DAM_error_flag'][l] += 1
                    DAM_l = np.ones(self.shape[l], dtype='bool')
                    self.INSTs[INST_name]['DAM'][l] = DAM_l

            # * DAM_indexes를 지정함
            for l in range(self.L):
                nonzero_DAM_l = np.nonzero(self.INSTs[INST_name]['DAM'][l])
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

                self.INSTs[INST_name]['DAM_indexes'].append(DAM_indexes_l)

    def se_lfmd(self, INST_name, FM_k_l, l, percent=50, sensitive=1):
        '''
        se_lfmd: shift exponential layer feature map distance
        일단 디폴트로 length_min length_max의 50(정중앙 값)에 해당하는 부분을 origin(원점)으로 이동
        그리고 민감도는 디폴트로 1로 설정함
        '''
        se_lfmd = 0
        # 가독성을 위해 간단한 변수명으로 초기화
        norm_min = self.norm_min; norm_max = self.norm_max
        FM_repre_HP = self.INSTs[INST_name]['FM_repre_HP']
        # self.TFM_repre[l], FM_k_l를 self.norm_min, self.norm_max으로 정규화
        TFM_repre_l_norm = self.normalize_layer(self.TFM_repre[FM_repre_HP][l], norm_min, norm_max)
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
        for index in self.INSTs[INST_name]['DAM_indexes'][l]:
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

    def Ln_lfmd(self, INST_name, FM_k_l, l, n=1):
        '''
        Ln_lfmd: Ln layer feature map distance
        레이어의 인덱스들 간의 절대값을 모두 더한다.
        '''
        Ln_lfmd = 0
        # 가독성을 위해 간단한 변수명으로 초기화
        norm_min = self.norm_min; norm_max = self.norm_max
        FM_repre_HP = self.INSTs[INST_name]['FM_repre_HP']
        # self.TFM_repre[l], FM_k_l를 self.norm_min, self.norm_max으로 정규화
        TFM_repre_l_norm = self.normalize_layer(self.TFM_repre[FM_repre_HP][l], norm_min, norm_max)
        FM_k_l_norm = self.normalize_layer(FM_k_l, norm_min, norm_max)

        # lengths: 인덱스 마다 TFM_repre_norm과 FM_k_l_norm 사이의 거리(절대값)를 구함
        lengths = abs(TFM_repre_l_norm - FM_k_l_norm)
        # DAM에서 True인 부분만 가지고 옴.
        lengths = lengths[self.INSTs[INST_name]['DAM'][l]]
        # lengths의 각 원소에 지수 n을 취함
        lengths_pow_n = np.power(lengths, n)
        # lengths_pow_n을 모두 더한 후 1/n 지수를 취하면 Ln_lfmd임
        Ln_lfmd = np.power(lengths_pow_n.sum(), 1/n)

        return Ln_lfmd

    def lfmd(self, lfmd_HP):
        if lfmd_HP == "se_lfmd":
            return self.se_lfmd
        elif lfmd_HP == "Ln_lfmd":
            return self.Ln_lfmd

    def fmd(self, INST_name, FM_k):
        # 피처 맵 거리를 0으로 초기화
        fmd=0
        # lfmds: 레이어 피처 맵 거리를 담는 곳
        lfmds=[]
        # 각 레이어에 대한 레이어 피처 맵 거리 계산법으로 레이어 피처 맵 계산
        for l in range(self.L):
            FM_k_l = FM_k[l]
            lfmd_l = self.lfmd(self.INSTs[INST_name]['lfmd_HP'])(INST_name, FM_k_l, l)
            lfmds.append(lfmd_l)
        # 레이어 피처 맵마다 weight를 줌
        for l in range(self.L):
            fmd += self.INSTs[INST_name]['W'][l]*lfmds[l]

        return fmd

    def set_fmds(self):
        def set_fmds(INST_name, valid_name):
            # RFM와 WFM을 부르기 위한 변수들을 선언함
            if valid_name=="rvalid":
                valid=self.origin_names[1]; valid_dir=self.rvalid_dir; valid_K=self.origin_K[valid]
                FMP_k=None; prev_FMPI_k=None; cur_FMPI_k=None; fmds=self.INSTs[INST_name]['rvalid_fmds']
            elif valid_name=="wvalid":
                valid=self.origin_names[2]; valid_dir=self.wvalid_dir; valid_K=self.origin_K[valid]
                FMP_k=None; prev_FMPI_k=None; cur_FMPI_k=None; fmds=self.INSTs[INST_name]['wvalid_fmds']

            for k in range(valid_K):
                # k번 째 데이터의 FMPI, FMPO 구함
                cur_FMPI_k = k // self.FMP_count
                FMPO_k = k % self.FMP_count

                # FMP_k가 이미 있다면 가지고 오지 않고 없다면 이전 FMP를 지우고 현재 FMP를 가지고 온다.
                if cur_FMPI_k == prev_FMPI_k:
                    pass
                else:
                    del FMP_k
                    with open(f'{valid_dir}/{valid}_{cur_FMPI_k}.pickle', 'rb') as f:
                        FMP_k = pickle.load(f)

                prev_FMPI_k = cur_FMPI_k

                FM_k = FMP_k[FMPO_k]

                fmds.append(self.fmd(INST_name, FM_k))

            # valid를 모두 순회하고 난 후 FMP_k의 기억공간을 램에서 제거
            del FMP_k

        for INST_name in self.INST_names:
            # * INST에 있는 rvalid_fmds, wvalid_fmds 빈 배열로 초기화
            self.INSTs[INST_name]['rvalid_fmds']=[]; self.INSTs[INST_name]['wvalid_fmds']=[]
            # * INST에 있는 rvalid_fmds, wvalid_fmds에 fmd로 초기화
            rvalid = self.origin_names[1]; wvalid = self.origin_names[2]
            set_fmds(INST_name, rvalid); self.INSTs[INST_name]['rvalid_fmds'] = np.array(self.INSTs[INST_name]['rvalid_fmds'])
            set_fmds(INST_name, wvalid); self.INSTs[INST_name]['wvalid_fmds'] = np.array(self.INSTs[INST_name]['wvalid_fmds'])

    def set_fmdc(self):
        for INST_name in self.INST_names:
            # * 가독성을 위해 간단한 변수명을 사용함
            fmdc_HP = self.INSTs[INST_name]['fmdc_HP']
            rvalid_fmds = self.INSTs[INST_name]['rvalid_fmds']
            wvalid_fmds = self.INSTs[INST_name]['wvalid_fmds']
            # * threshold(fmdc) 가시화를 위해 HP_fmdcs 초기화
            self.INSTs[INST_name]['HP_fmdcs']['rvalid_fmds_max'] = rvalid_fmds.max()
            self.INSTs[INST_name]['HP_fmdcs']['rvalid_fmds_average'] = rvalid_fmds.mean()
            self.INSTs[INST_name]['HP_fmdcs']['rvalid_fmds_middle'] = rvalid_fmds[len(rvalid_fmds) // 2]
            self.INSTs[INST_name]['HP_fmdcs']['wvalid_fmds_min'] = wvalid_fmds.min()
            self.INSTs[INST_name]['HP_fmdcs']['wvalid_fmds_average'] = wvalid_fmds.mean()
            self.INSTs[INST_name]['HP_fmdcs']['wvalid_fmds_middle'] = wvalid_fmds[len(wvalid_fmds) // 2]
            self.INSTs[INST_name]['HP_fmdcs']['rvalid_fmds_max_wvalid_fmds_min_average'] = (rvalid_fmds.max() + wvalid_fmds.min()) / 2
            # * fmdc_HP 방식대로 fmdc 지정하기
            # * 정분류에 대한 fmdc
            if fmdc_HP == 'rvalid_fmds_max':
                self.INSTs[INST_name]['fmdc'] = self.INSTs[INST_name]['HP_fmdcs']['rvalid_fmds_max']
            elif fmdc_HP == 'rvalid_fmds_average':
                self.INSTs[INST_name]['fmdc'] = self.INSTs[INST_name]['HP_fmdcs']['rvalid_fmds_average']
            elif fmdc_HP == 'rvalid_fmds_middle':
                self.INSTs[INST_name]['fmdc'] = self.INSTs[INST_name]['HP_fmdcs']['rvalid_fmds_middle']
            # * 오분류에 대한 fmdc
            elif fmdc_HP == 'wvalid_fmds_min':
                self.INSTs[INST_name]['fmdc'] = self.INSTs[INST_name]['HP_fmdcs']['wvalid_fmds_min']
            elif fmdc_HP == 'wvalid_fmds_average':
                self.INSTs[INST_name]['fmdc'] = self.INSTs[INST_name]['HP_fmdcs']['wvalid_fmds_average']
            elif fmdc_HP == 'wvalid_fmds_middle':
                self.INSTs[INST_name]['fmdc'] = self.INSTs[INST_name]['HP_fmdcs']['wvalid_fmds_middle']
            # * 정분류와 오분류에 대한 fmdc
            elif fmdc_HP == 'rvalid_fmds_max_wvalid_fmds_min_average':
                self.INSTs[INST_name]['fmdc'] = self.INSTs[INST_name]['HP_fmdcs']['rvalid_fmds_max_wvalid_fmds_min_average']

    def fit(self, FM_repre_MHP=['FM_mean'], alpha_MHP=[['rmw_max', 1000]], DAM_MHP=['all'], W_MHP=['C'], lfmd_MHP=['se_lfmd'], fmdc_MHP=['rvalid_fmds_average']):
        '''
        기본값:
            FM_repre_MHP=['FM_mean']
            alpha_MHP=[['rmw_max', 1000]]
            DAM_MHP=['all']
            W_MHP=['C']
            lfmd_MHP=['se_lfmd']
            fmdc_MHP=['rvalid_fmds_average']
        '''
        self.set_data_infos()
        self.set_FM_repres()
        self.set_alpha_rmw_min_max()
        self.set_MHP(FM_repre_MHP, alpha_MHP, DAM_MHP, W_MHP, lfmd_MHP, fmdc_MHP)
        self.init_INSTs()
        self.set_AMs_and_alpha_infos_and_DAM_infos()
        self.set_fmds()
        self.set_fmdc()

    def set_TPRs_TNRs_AUC(self):
        density = 1000 # 얼마나 많이 점을 찍을지 정함.
        for INST_name in self.INST_names:
            for eval_name in self.eval_names:
                # * 1. fmdcs 초기화
                self.INSTs[INST_name]['fmdcs'][eval_name]=[]
                eval_fmds = self.INSTs[INST_name]['eval_fmds'][eval_name]
                fmdc_min = eval_fmds.min(); fmdc_max = eval_fmds.max()
                fmdc_interval_length = (fmdc_max - fmdc_min) / density
                self.INSTs[INST_name]['fmdcs'][eval_name] = np.array([fmdc_min + interval_offset*fmdc_interval_length for interval_offset in range(density+1)])

                # * 2. TPRs, TNRs 구하기
                self.INSTs[INST_name]['TPRs'][eval_name]=[]; self.INSTs[INST_name]['TNRs'][eval_name]=[]; self.INSTs[INST_name]['AUC'][eval_name]=-1
                # eval_U, eval_U_K, eval_U_r, eval_U_w, eval_fmds 구하기
                eval_U = self.eval_U[eval_name]
                eval_U_K = self.eval_K[eval_name]; eval_U_r = len(np.nonzero(eval_U)[0]); eval_U_w = eval_U_K - eval_U_r
                eval_fmds = self.INSTs[INST_name]['eval_fmds'][eval_name]
                # 모든 fmdc에 대한 TPR, TNR 구해서 TPRs, TNRs에 모두 넣기
                for fmdc in self.INSTs[INST_name]['fmdcs'][eval_name]:
                    # is_eval_FM 구하기
                    is_eval_FMD = eval_fmds >= fmdc
                    # eval_FMD_K, eval_FMD_r, eval_FMD_w 구하기
                    eval_FMD_K = len(eval_U[is_eval_FMD]); eval_FMD_r = len(np.nonzero(eval_U[is_eval_FMD])[0]); eval_FMD_w = eval_FMD_K - eval_FMD_r
                    # fmdc에 대한 TPR, TNR 구해서 TPRs, TNRs에 각각 넣기
                    TPR = (eval_U_r - eval_FMD_r)/eval_U_r; TNR = eval_FMD_w/eval_U_w
                    self.INSTs[INST_name]['TPRs'][eval_name].append(TPR); self.INSTs[INST_name]['TNRs'][eval_name].append(TNR)

                self.INSTs[INST_name]['TPRs'][eval_name] = np.array(self.INSTs[INST_name]['TPRs'][eval_name]); self.INSTs[INST_name]['TNRs'][eval_name] = np.array(self.INSTs[INST_name]['TNRs'][eval_name])

                # * 3. AUC 구하기
                self.INSTs[INST_name]['AUC'][eval_name] = 0
                for i in range(len(self.INSTs[INST_name]['TNRs'][eval_name])-1):
                    TPR_middle_interval_i = (self.INSTs[INST_name]['TPRs'][eval_name][i+1] + self.INSTs[INST_name]['TPRs'][eval_name][i])/2;
                    one_minus_TNR_length_interval_i = (1-self.INSTs[INST_name]['TNRs'][eval_name][i+1]) - (1-self.INSTs[INST_name]['TNRs'][eval_name][i]) # 1-TNR = FPR
                    AUC_interval_i = one_minus_TNR_length_interval_i * TPR_middle_interval_i
                    self.INSTs[INST_name]['AUC'][eval_name] += AUC_interval_i

    def eval(self):
        for INST_name in self.INST_names:
            # * 1. 새로운 데이터를 받을 수 있게끔 속성을 초기 상태로 만듦.
            for eval_name in self.eval_names:
                self.INSTs[INST_name]['is_eval_FMD'][eval_name] = []; self.INSTs[INST_name]['eval_fmds'][eval_name] = []

            # * 2. is_eval_FMD, fmds를 구함
            for eval_name in self.eval_names:
                # self.fmds[eval_name], self.is_eval_FMD[eval_name]는
                # self.set_data_infos_and_related()에서 []로 초기화됨

                # EFM을 부르기 위한 변수들을 선언함
                eval_dir=f'{self.eval_dir}/{eval_name}'; eval_K=self.eval_K[eval_name]
                EFMP_k=None; prev_EFMPI_k=None; cur_EFMPI_k=None

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

                    fmd = self.fmd(INST_name, EFM_k)

                    self.INSTs[INST_name]['is_eval_FMD'][eval_name].append(fmd >= self.INSTs[INST_name]['fmdc'])
                    self.INSTs[INST_name]['eval_fmds'][eval_name].append(fmd)

                # eval_name을 모두 순회한 후 EFMP_k의 기억공간을 램에서 제거
                del EFMP_k

                self.INSTs[INST_name]['is_eval_FMD'][eval_name] = np.array(self.INSTs[INST_name]['is_eval_FMD'][eval_name])
                self.INSTs[INST_name]['eval_fmds'][eval_name] = np.array(self.INSTs[INST_name]['eval_fmds'][eval_name])


            # * 3. confusion matrix를 표현하기 위한 가장 기초적인 원소들을 초기화
            for eval_name in self.eval_names:
                # * 3.1. TP, FN, TN, FP을 초기화하기 위한 변수 초기화
                eval_U_K = self.eval_K[eval_name]
                eval_U_r = len(np.nonzero(self.eval_U[eval_name])[0])
                eval_U_w = eval_U_K - eval_U_r
                eval_U = self.eval_U[eval_name]; is_eval_FMD = self.INSTs[INST_name]['is_eval_FMD'][eval_name]
                eval_FMD_K = len(eval_U[is_eval_FMD])
                eval_FMD_r = len(np.nonzero(eval_U[is_eval_FMD])[0])
                eval_FMD_w = eval_FMD_K - eval_FMD_r

                # * 3.2. TP, FN, TN, FP 초기화
                self.INSTs[INST_name]['TP'][eval_name] = (eval_U_r - eval_FMD_r) / eval_U_K
                self.INSTs[INST_name]['FN'][eval_name] = eval_FMD_r / eval_U_K
                self.INSTs[INST_name]['TN'][eval_name] = eval_FMD_w / eval_U_K
                self.INSTs[INST_name]['FP'][eval_name] = (eval_U_w - eval_FMD_w) / eval_U_K
                # 변수명을 간단히 함
                TP_eval_name = self.INSTs[INST_name]['TP'][eval_name]; FN_eval_name = self.INSTs[INST_name]['FN'][eval_name]
                TN_eval_name = self.INSTs[INST_name]['TN'][eval_name]; FP_eval_name = self.INSTs[INST_name]['FP'][eval_name]
                # * 3.3. P, N 초기화
                self.INSTs[INST_name]['P'][eval_name] = TP_eval_name + FN_eval_name; self.INSTs[INST_name]['N'][eval_name] = TN_eval_name + FP_eval_name
                # 변수명을 간단히 함
                P_eval_name = self.INSTs[INST_name]['P'][eval_name]; N_eval_name = self.INSTs[INST_name]['N'][eval_name]
                # * 3.4. TPR, TNR, FNR, FPR 초기화
                self.INSTs[INST_name]['TPR'][eval_name] = TP_eval_name / P_eval_name; self.INSTs[INST_name]['TNR'][eval_name]= TN_eval_name / N_eval_name
                self.INSTs[INST_name]['FNR'][eval_name] = FN_eval_name / P_eval_name; self.INSTs[INST_name]['FPR'][eval_name] = FP_eval_name / N_eval_name
                # * 3.5. PPV, NPV, FDR, FOR 초기화
                # self.TP[eval_name] + self.FP[eval_name]가 0이 되는 경우가 발생할 때 그것으로 나누지 않음
                # 초기화된 -1 값을 그대로 가지고 감
                if TP_eval_name + FP_eval_name > 0:
                    self.INSTs[INST_name]['PPV'][eval_name] = TP_eval_name / (TP_eval_name + FP_eval_name)
                    self.INSTs[INST_name]['FDR'][eval_name] = FP_eval_name / (TP_eval_name + FP_eval_name)
                if FN_eval_name + TN_eval_name > 0:
                    self.INSTs[INST_name]['NPV'][eval_name] = TN_eval_name / (FN_eval_name + TN_eval_name)
                    self.INSTs[INST_name]['FOR'][eval_name] = FN_eval_name / (FN_eval_name + TN_eval_name)

        # * 4. TPRs, TNRs, AUC 구하기
        self.set_TPRs_TNRs_AUC()

    def set_square_NPs_infos(self, figsize=None, column=None):
        if figsize!=None:
            self.square_NPs_figsize = figsize
        if column!=None:
            self.square_NPs_column = column

    def show_square_NPs(self, nps, np_names=[], color_map=True, title_fontsize=12):
        # * nps_name가 지정되지 않았을 경우 Untitled로 이름 지정함
        Untitled_count = len(nps) - len(np_names)
        for i in range(Untitled_count):
            np_names.append('Untitled')

        if color_map==True:
            # * color_map이 활성화되었을 경우 'color map'을 np_names 맨 마지막에 추가함.
            np_names.append('color map')

            # nps에 nps의 최소의 값부터 최대 값까지 그리는 넘파이 추가
            # 이 넘파이 배열을 그리는 이유는 최소 최대에 대한 시각적인 표현을 할 수 있을 뿐만 아니라
            # color bar로 인해 좁게 그려지는 넘파이 배열을 없앨 수 있다.
            # nps_min 찾기
            nps_min = nps[0].min()
            for i in range(1, len(nps)):
                if nps_min > nps[i].min():
                    nps_min = nps[i].min()
            # nps_max 찾기
            nps_max = nps[0].max()
            for i in range(1, len(nps)):
                if nps_max > nps[i].max():
                    nps_max = nps[i].max()
            # bool 타입이라면 마지막에 넘파이에 True, False만 넣고 즉, (T,T), (T,F), (F,T), (F,F)만 넣고
            if str(nps[0].dtype) == 'bool':
                nps.append(np.array([True, False]))
            # 그게 아니라면 마지막에 넘파이에 nps_slice로 잘린 연속적인 값들을 넣음.
            else:
                # nps_slice: nps_min 부터 nps_max 까지 그리는데 몇 번에 걸쳐서 그릴지 정하기
                nps_slice = 1024
                nps_interval = (nps_max - nps_min) / nps_slice
                # nps에 nps의 최소의 값부터 최대 값까지 그리는 넘파이 생성
                np_min_to_max = np.array([nps_min + i*nps_interval for i in range(nps_slice+1)])
                nps.append(np_min_to_max)

        # 레이어 개수 만큼 레이어 원소 개수 계산
        element_count=[]
        for ith in range(len(nps)):
            ith_element_count = 1
            for ith_shape_ele in nps[ith].shape:
                ith_element_count *= ith_shape_ele

            element_count.append(ith_element_count)

        # 레이어 원소 개수로 정방 이차 배열의 한 변의 길이 구함
        square_NPs_side=[]
        for ith in range(len(nps)):
            square_NPs_side.append(math.ceil(math.sqrt((element_count[ith]))))

        # 레이어 피처 맵을 평탄화함.
        nps_flatten=[]
        for ith in range(len(nps)):
            nps_flatten.append(nps[ith].flatten())

        # x,y로 nps를 그리기 위한 이차 정방 배열 좌표를 만듦
        x=[];y=[]
        for ith in range(len(nps)):
            x_ith=[]; y_ith=[]
            for y_ in range(square_NPs_side[ith]):
                for x_ in range(square_NPs_side[ith]):
                    x_ith.append(x_)
                    y_ith.append(y_)
            # x_ith, y_ith를 nps[ith] 개수 만큼 자름
            x_ith = x_ith[:element_count[ith]]; y_ith = y_ith[:element_count[ith]]
            # x_ith을 x에 넣고 y_ith을 y에 넣음
            x.append(x_ith); y.append(y_ith)

        # plt의 subplot의 크기 지정
        square_NPs_row = (len(nps)-1 // self.square_NPs_column) + 1

        # 넘파이 배열을 5줄 씩 그리기
        plt.figure(figsize=self.square_NPs_figsize)
        for i in range(len(nps)):
            plt.subplot(square_NPs_row, self.square_NPs_column,i+1)
            # fontsize가 square_NPs_side[i] * 5 / 6 정도 되면 그래프 보여지는 x 축 길이만큼 폰트가 조절됨.
            # fontsize = square_NPs_side[i] * 5 / 6
            plt.title(np_names[i],fontdict={'size': f'{title_fontsize}'})
            plt.scatter(x=x[i], y=y[i], c=nps_flatten[i], cmap='jet')
            plt.axis('off')
        plt.colorbar()

        plt.show()

    def show_dir_infos(self):
        # * 1. 이 show 메소드를 실행하는 범주 출력
        root_dir_splited = self.root_dir.split('/')
        data_set_name = root_dir_splited[-2]; class_name = root_dir_splited[-1]
        print(f'[{data_set_name}, {class_name}]')
        print()
        # * 2. dir_infos 출력
        print(f"root_dir:\t{self.root_dir}")
        print(f"origin_dir:\t{self.origin_dir}")
        print(f"train_dir:\t{self.train_dir}")
        print(f"rvalid_dir:\t{self.rvalid_dir}")
        print(f"wvalid_dir:\t{self.wvalid_dir}")
        print(f"eval_dir:\t{self.eval_dir}")

        print('-'*100)

    def show_data_infos(self):
        # * 1. 이 show 메소드를 실행하는 범주 출력
        root_dir_splited = self.root_dir.split('/')
        data_set_name = root_dir_splited[-2]; class_name = root_dir_splited[-1]
        print(f'[{data_set_name}, {class_name}]')
        print()
        # * 2. data_infos 출력
        print(f"origin_names:\t{self.origin_names}")
        print(f"origin_K:\t{self.origin_K}")
        print(f"eval_names:\t{self.eval_names}")
        print(f"eval_K:\t\t{self.eval_K}")
        print(f"L:\t\t{self.L}")
        print(f"shape:\t\t{self.shape}")
        print(f"FMP_count:\t{self.FMP_count}")
        # * 3. eval_U 출력
        # * eval_U, eval_U_names 초기화
        eval_U = [self.eval_U[eval_name] for eval_name in self.eval_names] # eval_U 초기화
        eval_U_name = ['eval_U_' + eval_name for eval_name in self.eval_names] # eval_U_name 초기화
        # * eval_U의 크기에 따라 set_square_NPs_infos를 달리함.
        if len(eval_U) == 1:
            self.set_square_NPs_infos([20,5])
        else:
            self.set_square_NPs_infos([20,10])
        # * eval_U 이차 정방 행렬로 출력
        self.show_square_NPs(eval_U, eval_U_name, color_map=False)

    def show_FM_repres(self, INST_name=None):
        # * 1. 이 show 메소드를 실행하는 범주 출력
        root_dir_splited = self.root_dir.split('/')
        data_set_name = root_dir_splited[-2]; class_name = root_dir_splited[-1]
        # * 특정 INST가 없다면 데이터셋, 클래스를 출력하고, 특정 INST가 있다면 데이터셋, 클래스, INST_name을 출력한다.
        if INST_name == None:
            print(f'[{data_set_name}, {class_name}]')
        else:
            print(f'[{data_set_name}, {class_name}, [{INST_name}]]')
        print()

        # * 2. FM_repres 출력
        self.set_square_NPs_infos([60,60],column=7)
        # * 특정 INST가 없다면 모든 HP들을 출력하고, 특정 INST가 있다면 그 HP만 출력한다.
        if INST_name == None:
            FM_repre_HPs = ['FM_min', 'FM_mean', 'FM_max']
        else:
            FM_repre_HPs = [self.INSTs[INST_name]['FM_repre_HP']]
        # * 모든 FM_repre_HP들을 출력
        for FM_repre_HP in FM_repre_HPs:
            # * FM_repre, FM_repre_name 초기화
            TFM_repre=self.TFM_repre[FM_repre_HP].copy(); TFM_repre_name = [f'T{FM_repre_HP}_' + str(l) for l in range(self.L)]
            RFM_repre=self.RFM_repre[FM_repre_HP].copy(); RFM_repre_name = [f'R{FM_repre_HP}_' + str(l) for l in range(self.L)]
            WFM_repre=self.WFM_repre[FM_repre_HP].copy(); WFM_repre_name = [f'W{FM_repre_HP}_' + str(l) for l in range(self.L)]
            # * FM_repre, FM_repre_name 이차 정방 행렬로 출력
            self.show_square_NPs(TFM_repre, TFM_repre_name, title_fontsize=40); self.show_square_NPs(RFM_repre, RFM_repre_name, title_fontsize=40); self.show_square_NPs(WFM_repre, WFM_repre_name, title_fontsize=40)

    def show_alpha_infos(self, INST_name=None):
        # * 1. 이 show 메소드를 실행하는 범주 출력
        root_dir_splited = self.root_dir.split('/')
        data_set_name = root_dir_splited[-2]; class_name = root_dir_splited[-1]
        # * 특정 INST가 없다면 데이터셋, 클래스를 출력하고, 특정 INST가 있다면 데이터셋, 클래스, INST_name을 출력한다.
        if INST_name == None:
            print(f'[{data_set_name}, {class_name}]')
        else:
            print(f'[{data_set_name}, {class_name}, [{INST_name}]]')
        print()
        # * 2. alpha_infos 출력
        # * 특정 INST가 없다면 FM_min, FM_mean, FM_max 각각 alpha_min, alpha_max 출력
        if INST_name == None:
            FM_repre_HPs=['FM_min', 'FM_mean', 'FM_max']
            # 모든 FM_repre_HP에 대하여 alpha_min, alpha_max 출력
            for FM_repre_HP in FM_repre_HPs:
                # FM_repre_HP 출력
                print(f"{FM_repre_HP}:")
                # alpha_min 출력
                print("\talpha_min:", sep='', end='')
                for l in range(self.L):
                    print(f"{self.alpha_min[FM_repre_HP][l]: 0.4f}", sep='', end='')
                print()
                # alpha_max 출력
                print("\talpha_max:", sep='', end='')
                for l in range(self.L):
                    print(f"{self.alpha_max[FM_repre_HP][l]: 0.4f}", sep='', end='')
                print()
        # * 특정 INST가 있다면 alpha_slice, alpha_min, alpha, alpha_max, rmw 출력
        else:
            # 특정 FM_repre_HP에 대하여
            FM_repre_HP = self.INSTs[INST_name]['FM_repre_HP']
            # alpha_slice 출력
            print(f"alpha_slice:\t{self.INSTs[INST_name]['alpha_slice']: 11d}")
            print('='*82)
            # alpha_min 출력
            print("alpha_min:\t", sep='', end='')
            for l in range(self.L):
                print('|', sep='', end='')
                print(f"{self.alpha_min[FM_repre_HP][l]: 10.4f}", sep='', end='')
            print('|', sep='', end='')
            print()
            print(' '*16 + '|' + '-'*65 + '|')
            # alpha 출력
            print("alpha:\t\t", sep='', end='')
            for l in range(self.L):
                print('|', sep='', end='')
                print(f"{self.INSTs[INST_name]['alpha'][l]: 10.4f}", sep='', end='')
            print('|', sep='', end='')
            print()
            # alpha_percent 출력
            print("alpha_percent:\t", sep='', end='')
            for l in range(self.L):
                print('|', sep='', end='')
                print(f"{self.INSTs[INST_name]['alpha_percent'][l]: 9d}", sep='', end='')
                print('%', sep='', end='')
            print('|', sep='', end='')
            print()
            print("alpha_gage:\t", sep='', end='')
            for l in range(self.L):
                print('|', sep='', end='')
                alpha_percent_l = self.INSTs[INST_name]['alpha_percent'][l]
                alpha_sharp_count_l = alpha_percent_l // 10
                for cur_sharp_position in range(10, 0, -1):
                    if cur_sharp_position <= alpha_sharp_count_l:
                        print('#', sep='', end='')
                    else:
                        print('.', sep='', end='')
            print('|', sep='', end='')
            print()
            print(' '*16 + '|' + '-'*65 + '|')
            # alpha_max 출력
            print("alpha_max:\t", sep='', end='')
            for l in range(self.L):
                print('|', sep='', end='')
                print(f"{self.alpha_max[FM_repre_HP][l]: 10.4f}", sep='', end='')
            print('|', sep='', end='')
            print()
            print('='*82)
            # rmw_min 출력
            print("rmw_min:\t", sep='', end='')
            for l in range(self.L):
                print('|', sep='', end='')
                print(f"{self.rmw_min[l]: 10d}", sep='', end='')
            print('|', sep='', end='')
            print()
            print(' '*16 + '|' + '-'*65 + '|')
            # rmw 출력
            print("rmw:\t\t", sep='', end='')
            for l in range(self.L):
                print('|', sep='', end='')
                print(f"{self.INSTs[INST_name]['rmw'][l]: 10d}", sep='', end='')
            print('|', sep='', end='')
            print()
            # rmw_percent 출력
            print("rmw_percent:\t", sep='', end='')
            for l in range(self.L):
                print('|', sep='', end='')
                print(f"{self.INSTs[INST_name]['rmw_percent'][l]: 9d}", sep='', end='')
                print('%', sep='', end='')
            print('|', sep='', end='')
            print()
            print("rmw_gage:\t", sep='', end='')
            for l in range(self.L):
                print('|', sep='', end='')
                rmw_percent_l = self.INSTs[INST_name]['rmw_percent'][l]
                rmw_sharp_count_l = rmw_percent_l // 10
                for cur_sharp_position in range(10, 0, -1):
                    if cur_sharp_position <= rmw_sharp_count_l:
                        print('#', sep='', end='')
                    else:
                        print('.', sep='', end='')
            print('|', sep='', end='')
            print()
            print(' '*16 + '|' + '-'*65 + '|')
            # alpha_max 출력
            print("rmw_max:\t", sep='', end='')
            for l in range(self.L):
                print('|', sep='', end='')
                print(f"{self.rmw_max[l]: 10d}", sep='', end='')
            print('|', sep='', end='')
            print()

        print('-'*100)

    def show_HP(self, INST_name):
        # * 1. 이 show 메소드를 실행하는 범주 출력
        root_dir_splited = self.root_dir.split('/')
        data_set_name = root_dir_splited[-2]; class_name = root_dir_splited[-1]
        print(f'[{data_set_name}, {class_name}, [{INST_name}]]')
        print()
        # * 2. HP들 출력
        INST_name_list=INST_name.split()
        FM_repre_HP=INST_name_list[0]; alpha_HP_str=INST_name_list[1]; DAM_HP=INST_name_list[2]
        lfmd_HP=INST_name_list[3]; W_HP=INST_name_list[4]; fmdc_HP=INST_name_list[5]
        alpha_HP_list = alpha_HP_str.split(',')

        print(f"FM_repre_HP:\t{FM_repre_HP}")
        print(f"alpha_HP:\t{alpha_HP_list[1:]}")
        print(f"DAM_HP:\t\t{DAM_HP}")
        print(f"lfmd_HP:\t{lfmd_HP}")
        print(f"W_HP:\t\t{W_HP}")
        print(f"fmdc_HP:\t{fmdc_HP}")

        print('-'*100)

    def show_AMs(self, INST_name):
        # * 1. 이 show 메소드를 실행하는 범주 출력
        root_dir_splited = self.root_dir.split('/')
        data_set_name = root_dir_splited[-2]; class_name = root_dir_splited[-1]
        # * 특정 INST가 있으니 데이터셋, 클래스, INST_name을 출력한다.
        print(f'[{data_set_name}, {class_name}, [{INST_name}]]')
        print()

        # * 2. AMs 출력
        self.set_square_NPs_infos([60,60],column=7)
        # * AMs, AM_name 초기화
        TAM=self.INSTs[INST_name]['TAM'].copy(); TAM_name = ['TAM_' + str(l) for l in range(self.L)]
        RAM=self.INSTs[INST_name]['RAM'].copy(); RAM_name = ['RAM_' + str(l) for l in range(self.L)]
        WAM=self.INSTs[INST_name]['WAM'].copy(); WAM_name = ['WAM_' + str(l) for l in range(self.L)]
        DAM_error_flag = self.INSTs[INST_name]['DAM_error_flag']
        DAM=self.INSTs[INST_name]['DAM'].copy(); DAM_name = ['DAM_' + str(l) + f'({DAM_error_flag[l]})' for l in range(self.L)]
        # * AM_repres, AM_repre_names 이차 정방 행렬로 출력
        self.show_square_NPs(TAM, TAM_name, title_fontsize=40); self.show_square_NPs(RAM, RAM_name, title_fontsize=40); self.show_square_NPs(WAM, WAM_name, title_fontsize=40); self.show_square_NPs(DAM, DAM_name, title_fontsize=40)

    def show_DAM_infos(self, INST_name):
        # * 1. 이 show 메소드를 실행하는 범주 출력
        root_dir_splited = self.root_dir.split('/')
        data_set_name = root_dir_splited[-2]; class_name = root_dir_splited[-1]
        # * 특정 INST가 있으니 데이터셋, 클래스, INST_name을 출력한다.
        print(f'[{data_set_name}, {class_name}, [{INST_name}]]')
        print()

        # * 2. DAM_infos 출력
        # DAM_HP 출력
        DAM_HP = self.INSTs[INST_name]['DAM_HP']
        # DAM_error_flag 출력
        DAM_error_flag = self.INSTs[INST_name]['DAM_error_flag']
        print(f"DAM_HP:\t\t{DAM_HP}")
        print(f"DAM_error_flag:\t{DAM_error_flag}")
        # DAM 출력
        DAM=self.INSTs[INST_name]['DAM'].copy(); DAM_name = ['DAM_' + str(l) + f'({DAM_error_flag[l]})' for l in range(self.L)]
        self.show_square_NPs(DAM, DAM_name, title_fontsize=40)

    def show_layer_infos(self, INST_name):
        # * 1. 이 show 메소드를 실행하는 범주 출력
        root_dir_splited = self.root_dir.split('/')
        data_set_name = root_dir_splited[-2]; class_name = root_dir_splited[-1]
        # * 특정 INST가 있으니 데이터셋, 클래스, INST_name을 출력한다.
        print(f'[{data_set_name}, {class_name}, [{INST_name}]]')
        print()

        # * 2. layer_infos 출력
        # lfmd_HP 출력
        print(f"lfmd_HP:\t{self.INSTs[INST_name]['lfmd_HP']}")
        # W 출력
        print("W:\t\t", sep='', end='')
        for l in range(self.L):
            print(f"{self.INSTs[INST_name]['W'][l]: 7.4f}".lstrip()+" ", sep='', end='')
        print()
        # norm_min 출력
        print(f"norm_min:\t{self.norm_min}")
        # norm_max 출력
        print(f"norm_max:\t{self.norm_max}")

        print('-'*100)

    def show_rvalid_fmds_wvalid_fmds(self, INST_name, show_category=True, show_HP_fmdcs=True):
        if show_category == True:
            # * 1. 이 show 메소드를 실행하는 범주 출력
            root_dir_splited = self.root_dir.split('/')
            data_set_name = root_dir_splited[-2]; class_name = root_dir_splited[-1]
            # * 특정 INST가 있으니 데이터셋, 클래스, INST_name을 출력한다.
            print(f'[{data_set_name}, {class_name}, [{INST_name}]]')
            print()

        # * 2. rvalid_fmds, wvalid_fmds 그래프 그리기
        plt.boxplot([self.INSTs[INST_name]['rvalid_fmds'], self.INSTs[INST_name]['wvalid_fmds']], notch=True,)
        plt.xticks([1, 2], ['rvalid_fmds', 'wvalid_fmds'])

        # * s3. HP_fmdcs 그리기
        # * 3.1. value_names에 그래프에 그릴 값과 이름을 모두 넣음.
        HP_fmdcs = self.INSTs[INST_name]['HP_fmdcs']; value_names=[]
        fmdc_HP = self.INSTs[INST_name]['fmdc_HP']
        for key in HP_fmdcs.keys():
            if key == fmdc_HP:
                value_names.append([HP_fmdcs[key], key + '(fmdc)'])
            else:
                value_names.append([HP_fmdcs[key], key])
        for eval_name in self.eval_names:
            eval_fmds = self.INSTs[INST_name]['eval_fmds'][eval_name]
            # weval_fmds.min(), reval_fmds.max()의 값과 이름을 모두 value_names에 저장함
            reval_fmds = eval_fmds[self.eval_U[eval_name]]; reval_fmds_max = reval_fmds.max(); value_names.append([reval_fmds_max, f'reval_fmds_max_{eval_name}'])
            weval_fmds = eval_fmds[np.logical_not(self.eval_U[eval_name])]; weval_fmds_min = weval_fmds.min(); value_names.append([weval_fmds_min, f'weval_fmds_min_{eval_name}'])

        # * 3.2. 실선 그리기
        # value_names로 HP_fmdcs의 값과 이름을 fmds를 가로질러 표시함
        for value, name in value_names:
            name_splited = name.split('_')
            if name_splited[0][1:] == 'eval':
                plt.plot([1, 2], [value, value], label=name, linestyle='--', linewidth=4, alpha=0.4)
            elif name == fmdc_HP + '(fmdc)':
                plt.plot([1, 2], [value, value], label=name, linestyle=':', linewidth=4, alpha=0.4)
            else:
                plt.plot([1, 2], [value, value], label=name)
        # legend로 값의 label 표시
        plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', ncol=1)

        plt.xticks(fontsize = 10)
        plt.yticks(fontsize = 14)
        plt.ylabel('feature map distance', {'size': '16'})
        plt.show()

        # * 4. HP_fmdcs 값 출력하기
        if show_HP_fmdcs == True:
            for value, name in value_names:
                if name == 'rvalid_fmds_max_wvalid_fmds_min_average' or\
                   name == 'rvalid_fmds_max_wvalid_fmds_min_average(fmdc)':
                    print(f"{name}:\t{value: 0.4f}")
                elif name.split('_')[-1] == 'average(fmdc)' or name.split('_')[-1] == 'middle(fmdc)':
                    print(f"{name}:\t\t\t{value: 0.4f}")
                else:
                    print(f"{name}:\t\t\t\t{value: 0.4f}")

    def show_fmdc_infos(self, INST_name):
        # * 1. 이 show 메소드를 실행하는 범주 출력
        root_dir_splited = self.root_dir.split('/')
        data_set_name = root_dir_splited[-2]; class_name = root_dir_splited[-1]
        # * 특정 INST가 있으니 데이터셋, 클래스, INST_name을 출력한다.
        print(f'[{data_set_name}, {class_name}, [{INST_name}]]')
        print()

        # * 2. fmdc_infos 출력
        # rvalid_fmds, wvalid_fmds 출력
        self.show_rvalid_fmds_wvalid_fmds(INST_name, show_category=False, show_HP_fmdcs=False)

        # fmdc_HP, fmdc 출력
        print(f"fmdc_HP, fmdc:\t{self.INSTs[INST_name]['fmdc_HP']}:\t{self.INSTs[INST_name]['fmdc']: 0.4f}")
        # HP_fmdcs 출력하기
        HP_fmdcs = self.INSTs[INST_name]['HP_fmdcs']
        print(f"HP_fmdcs:")
        for key in HP_fmdcs.keys():
            if key == 'rvalid_fmds_max_wvalid_fmds_min_average':
                print(f"\t\t{key}:\t{HP_fmdcs[key]: 0.4f}")
            else:
                print(f"\t\t{key}:\t\t\t\t{HP_fmdcs[key]: 0.4f}")

    def show_eval_venn_diagram(self, INST_name, eval_name, show_category=True):
        if show_category == True:
            # * 1. 이 show 메소드를 실행하는 범주 출력
            root_dir_splited = self.root_dir.split('/')
            data_set_name = root_dir_splited[-2]; class_name = root_dir_splited[-1]
            # * 특정 INST가 있으니 데이터셋, 클래스, INST_name을 출력한다.
            print(f'[{data_set_name}, {class_name}, [{INST_name}], {eval_name}]')
            print()
        # * 2. 벤 다이어그램 그리기
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
        # * 3. 벤 다이어그램에 적절한 텍스트(eval_U, eval_FMD, Wrong, Right)와 수치(TP, FN, TN, FP) 넣기
        plt.plot(box_x, box_y, 'ro')
        # 집합 표시를 위한 텍스트 넣기
        plt.text(x=73, y=40, s="eval_U", fontdict={'color': 'green','size': 16})
        plt.text(x=13, y=20, s="eval_FMD", fontdict={'color': 'black','size': 16})
        plt.text(x=50, y=57, s="Wrong", fontdict={'color': 'red','size': 16})
        plt.text(x=-50, y=57, s="Right", fontdict={'color': 'blue','size': 16})
        # TP, FN, TN, FP 에 대한 숫자 넣기
        plt.text(x=-75, y=40, s=f"TP: {self.INSTs[INST_name]['TP'][eval_name]}", fontdict={'color': 'purple','size': 16})
        plt.text(x=-50, y=0, s=f"FN: {self.INSTs[INST_name]['FN'][eval_name]}", fontdict={'color': 'purple','size': 16})
        plt.text(x=25, y=0, s=f"TN: {self.INSTs[INST_name]['TN'][eval_name]}", fontdict={'color': 'purple','size': 16})
        plt.text(x=25, y=40, s=f"FP: {self.INSTs[INST_name]['FP'][eval_name]}", fontdict={'color': 'purple','size': 16})

        plt.axis('off')
        plt.show()

    def show_efficience_and_FMD_ratio(self, INST_name, eval_name, show_category=True):
        if show_category == True:
            # * 1. 이 show 메소드를 실행하는 범주 출력
            root_dir_splited = self.root_dir.split('/')
            data_set_name = root_dir_splited[-2]; class_name = root_dir_splited[-1]
            # * 특정 INST가 있으니 데이터셋, 클래스, INST_name을 출력한다.
            print(f'[{data_set_name}, {class_name}, [{INST_name}], {eval_name}]')
            print()
        # * 2. efficience 출력
        print(f"[wrong ratio U, N]:\t\t\t{self.INSTs[INST_name]['N'][eval_name]: 0.4f}") # 오분류 비율 U
        print(f"[wrong ratio FMD, NPV]:\t\t\t{self.INSTs[INST_name]['NPV'][eval_name]: 0.4f}") # 오분류 비율 FMD
        print(f"[recall, TPR]:\t\t\t\t{self.INSTs[INST_name]['TPR'][eval_name]: 0.4f}") # 재현율(TPR)
        print(f"[specificity, TNR]:\t\t\t{self.INSTs[INST_name]['TNR'][eval_name]: 0.4f}") # 특이도(TNR)
        print()
        # * 3. FMD ratio 출력
        eval_U_size = len(self.eval_U[eval_name])
        eval_FMD_size = len(self.eval_U[eval_name][self.INSTs[INST_name]['is_eval_FMD'][eval_name]])
        print(f"[FMD ratio(|eval_FMD|/|eval_U|)]:\t{eval_FMD_size/eval_U_size: 0.4f}")

    def show_eval_U_predicted_matched(self, INST_name, eval_name, show_category=True):
        if show_category == True:
            # * 1. 이 show 메소드를 실행하는 범주 출력
            root_dir_splited = self.root_dir.split('/')
            data_set_name = root_dir_splited[-2]; class_name = root_dir_splited[-1]
            # * 특정 INST가 있으니 데이터셋, 클래스, INST_name을 출력한다.
            print(f'[{data_set_name}, {class_name}, [{INST_name}], {eval_name}]')
            print()
        # * 2. eval_U, eval_U_predicted, eval_U_matched 그래프 그리기
        # eval_U: 실제 정분류, 오분류 그래프, eval_U_predicted: 예측한 정분류, 오분류 그래프
        # eval_U_matched: 실제와 예측한 값이 같다면 True, 아니라면 False
        self.set_square_NPs_infos([20,10])
        eval_U = self.eval_U[eval_name]; eval_U_predicted = np.logical_not(self.INSTs[INST_name]['is_eval_FMD'][eval_name])
        eval_U_matched = np.logical_not(np.logical_xor(eval_U, eval_U_predicted))
        self.show_square_NPs([eval_U, eval_U_predicted, eval_U_matched], ['eval_U', 'eval_U_predicted', 'eval_U_matched'])
        # * 3. accuracy 출력하기
        # TP, FN, TN, FP 초기화
        TP = self.INSTs[INST_name]['TP'][eval_name]; FN = self.INSTs[INST_name]['FN'][eval_name]; TN = self.INSTs[INST_name]['TN'][eval_name]; FP = self.INSTs[INST_name]['FP'][eval_name]
        accuracy = (TP + TN) / (TP + FN + TN + FP)
        print(f"accuracy:\t{accuracy}")

    def show_eval_fmd_right_ratio(self, INST_name, eval_name, show_category=True):
        if show_category == True:
            # * 1. 이 show 메소드를 실행하는 범주 출력
            root_dir_splited = self.root_dir.split('/')
            data_set_name = root_dir_splited[-2]; class_name = root_dir_splited[-1]
            # * 특정 INST가 있으니 데이터셋, 클래스, INST_name을 출력한다.
            print(f'[{data_set_name}, {class_name}, [{INST_name}], {eval_name}]')
            print()
        # * 2 eval_fmd_right_ratio 그래프 그리기
        # figure 크기를 지정함
        plt.figure(figsize=[6,4])
        # eval_fmd는 그래프에서 x축에 해당하는 부분, right_ratio는 그래프에서 y축에 해당하는 부분
        eval_fmd = []; right_ratio = []
        eval_fmds = copy.deepcopy(self.INSTs[INST_name]['eval_fmds'][eval_name])
        eval_fmd_min = eval_fmds.min(); eval_fmd_max = eval_fmds.max() # eval_fmds 최소값, 최대값 찾음.
        eval_fmd_slice = (self.eval_K[eval_name] // 10) + 1
        eval_fmd_interval_length = (eval_fmd_max - eval_fmd_min) / eval_fmd_slice

        # 각 interval을 순회하며 eval_fmd_value(interval의 중앙값)과 right_ratio을 구함
        for interval_offset in range(eval_fmd_slice):
            # 각 interval의 중앙값으로 eval_fmd에 할당
            interval_min = eval_fmd_min + interval_offset*eval_fmd_interval_length; interval_max = eval_fmd_min + (interval_offset+1)*eval_fmd_interval_length
            eval_fmd_value = (interval_min + interval_max) / 2
            # 각 interval의 정분류 비율을 RR에 할당
            upper_than_interval_min = 0
            lower_than_interval_max = 0
            if interval_offset != eval_fmd_slice-1:
                upper_than_interval_min = eval_fmds >= interval_min
                lower_than_interval_max = eval_fmds < interval_max
            else:
                upper_than_interval_min = eval_fmds >= interval_min
                lower_than_interval_max = eval_fmds <= interval_max

            # eval_U에서 interval_value에 해당하는 마커를 찾음
            interval_values_maker = np.logical_and(upper_than_interval_min, lower_than_interval_max)
            is_right_interval_values = self.eval_U[eval_name][interval_values_maker]
            R = len(np.nonzero(is_right_interval_values)[0]); W = len(is_right_interval_values) - R

            # interval에 아무것도 없다면 right_ratio_value에 -1을 할당
            if R+W == 0:
                right_ratio_value = -1
            else:
                right_ratio_value = R / (R + W)

            eval_fmd.append(eval_fmd_value)
            right_ratio.append(right_ratio_value)

        ones = np.ones(eval_fmd_slice)
        plt.scatter(x=eval_fmd, y=right_ratio, s=ones)

        # * 3. 최소제곱법을 이용해서 그래프에 가까운 1차함수 그리고 기울기 표시하기
        # eval_fmd, right_ratio를 모두 넘파이 배열로 바꾸기
        eval_fmd = np.array(eval_fmd); right_ratio = np.array(right_ratio)
        # 음이 아닌 eval_fmd, right_ratio로 각각 x, y를 고름
        right_ratio_non_negative_mask = right_ratio >= 0
        y = right_ratio[right_ratio_non_negative_mask]
        x = eval_fmd[right_ratio_non_negative_mask]
        # Ac = B, c = [a, b], a는 1차 함수 기울기, b는 1차 함수 y절편
        A = np.array([[(x**2).sum(), x.sum()],
                      [x.sum(), len(x)]])
        B = np.array([(x*y).sum(), y.sum()]).T
        c = np.linalg.inv(A)@B; a = c[0]; b = c[1]

        # line의 최소값과 최대값에 해당하는 x값 구하기
        x_line_min = (y.min()-b) / a; x_line_max = (y.max()-b) / a
        # line의 최소값과 최대값에 해당하는 x값 중 작은 것은 x_min, 큰 것은 x_max으로 할당
        x_min = 0; x_max = 0
        if x_line_min < x_line_max:
            x_min = x_line_min; x_max = x_line_max
        else:
            x_min = x_line_max; x_max = x_line_min
        # 최소제곱법 1차 함수 그리기
        plt.plot([x_line_min, x_line_max], [a*x_line_min+b, a*x_line_max+b])
        # line의 1/2 지점에 기울기 표시하기
        plt.text(x=(x_min + x_max)/2, y=(a*x_min+b + a*x_max+b)/2, s=f'slope = {a: 0.4f}')

        # * 4. 피어슨 상관 계수 표시하기
        x_mean = x.mean(); y_mean = y.mean() # x, y 평균 구하기
        cov =  ((x - x_mean)*(y - y_mean)).sum() / (len(x) - 1)  # x, y 공분산 구하기, len(x) 대신 len(x)-1로
        std_x = np.power(((x - x_mean)**2).sum()/(len(x) - 1), 1/2) # x 표준편차 구하기
        std_y = np.power(((y - y_mean)**2).sum()/(len(y) - 1), 1/2) # y 표준편차 구하기
        r =  cov / (std_x * std_y) # 피어슨 상관 계수 구하기
        # line의 1/3 지점에 피어슨 상관 계수 표시하기
        plt.text(x=(x_min * 2/3 + x_max * 1/3), y=((a*x_min+b)*2/3 + (a*x_max+b)*1/3), s=f'pearson correlation coefficient  = {r: 0.4f}')

        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.ylim(-0.1, 1.1) # interval에 아무것도 없는 것은 표기하지 않음
        plt.xlabel('feature map distance', {'size': '16'})
        plt.ylabel('right ratio', {'size': '16'})
        plt.show()

    def show_fmds(self, INST_name, eval_name, show_category=True):
        if show_category == True:
            # * 1. 이 show 메소드를 실행하는 범주 출력
            root_dir_splited = self.root_dir.split('/')
            data_set_name = root_dir_splited[-2]; class_name = root_dir_splited[-1]
            # * 특정 INST가 있으니 데이터셋, 클래스, INST_name을 출력한다.
            print(f'[{data_set_name}, {class_name}, [{INST_name}], {eval_name}]')
            print()

        # * 2. fmds 그래프 그리기
        eval_fmds = self.INSTs[INST_name]['eval_fmds'][eval_name]
        # reval_fmds: 정분류 평가 데이터에 대한 fmds, weval_fmds: 오분류 평가 데이터에 대한 fmds
        reval_fmds = eval_fmds[self.eval_U[eval_name]]; weval_fmds = eval_fmds[np.logical_not(self.eval_U[eval_name])]
        plt.boxplot([self.INSTs[INST_name]['rvalid_fmds'], self.INSTs[INST_name]['wvalid_fmds'], eval_fmds, reval_fmds, weval_fmds], notch=True)
        plt.xticks([1, 2, 3, 4, 5], ['rvalid_fmds', 'wvalid_fmds', 'eval_fmds', 'reval_fmds', 'weval_fmds'])

        # * 3. HP_fmdcs, weval_fmds.min(), reval_fmds.max() 그리기
        # * 4.1. value_names에 그래프에 그릴 값과 이름을 모두 넣음.
        # HP_fmdcs의 값과 이름을 모두 value_names에 저장함
        HP_fmdcs = self.INSTs[INST_name]['HP_fmdcs']; value_names=[]
        fmdc_HP = self.INSTs[INST_name]['fmdc_HP']
        for key in HP_fmdcs.keys():
            if key == fmdc_HP:
                value_names.append([HP_fmdcs[key], key + '(fmdc)'])
            else:
                value_names.append([HP_fmdcs[key], key])
        # weval_fmds.min(), reval_fmds.max()의 값과 이름을 모두 value_names에 저장함
        reval_fmds = eval_fmds[self.eval_U[eval_name]]; reval_fmds_max = reval_fmds.max(); value_names.append([reval_fmds_max, f'reval_fmds_max_{eval_name}'])
        weval_fmds = eval_fmds[np.logical_not(self.eval_U[eval_name])]; weval_fmds_min = weval_fmds.min(); value_names.append([weval_fmds_min, f'weval_fmds_min_{eval_name}'])

        # * 3.2. 실선 그리기
        # value_names로 HP_fmdcs의 값과 이름을 fmds를 가로질러 표시함
        for value, name in value_names:
            if name == f'reval_fmds_max_{eval_name}' or name == f'weval_fmds_min_{eval_name}':
                plt.plot([1, 5], [value, value], label=name, linestyle='--', linewidth=4, alpha=0.4)
            elif name == fmdc_HP + '(fmdc)':
                plt.plot([1, 5], [value, value], label=name, linestyle=':', linewidth=4, alpha=0.4)
            else:
                plt.plot([1, 5], [value, value], label=name)
        # legend로 값의 label 표시
        plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', ncol=1)

        plt.xticks(fontsize = 10)
        plt.yticks(fontsize = 14)
        plt.ylabel('feature map distance', {'size': '16'})
        plt.show()

        # * 4. HP_fmdcs, weval_fmds.min(), reval_fmds.max() 값 출력하기
        for value, name in value_names:
            if name == 'rvalid_fmds_max_wvalid_fmds_min_average' or\
               name == 'rvalid_fmds_max_wvalid_fmds_min_average(fmdc)':
                print(f"{name}:\t{value: .4f}")
            elif name.split('_')[-1] == 'average(fmdc)' or name.split('_')[-1] == 'middle(fmdc)':
                    print(f"{name}:\t\t\t{value: .4f}")
            else:
                print(f"{name}:\t\t\t\t{value: .4f}")

    def show_fmdc_TNR_TPR(self, INST_name, eval_name, show_category=True):
        if show_category == True:
            # * 1. 이 show 메소드를 실행하는 범주 출력
            root_dir_splited = self.root_dir.split('/')
            data_set_name = root_dir_splited[-2]; class_name = root_dir_splited[-1]
            # * 특정 INST가 있으니 데이터셋, 클래스, INST_name을 출력한다.
            print(f'[{data_set_name}, {class_name}, [{INST_name}], {eval_name}]')
            print()

        # * 2. fmdc_TNR_TPR 그래프 그리기
        plt.plot(self.INSTs[INST_name]['fmdcs'][eval_name], self.INSTs[INST_name]['TPRs'][eval_name], 'bo', label='TPR')
        plt.plot(self.INSTs[INST_name]['fmdcs'][eval_name], self.INSTs[INST_name]['TNRs'][eval_name], 'ro', label='TNR')

        # * 3. HP_fmdcs, weval_fmds.min(), reval_fmds.max() 그리기
        # * 3.1. value_names에 그래프에 그릴 값과 이름을 모두 넣음.
        # HP_fmdcs의 값과 이름을 모두 넣음
        value_names = []; HP_fmdcs = self.INSTs[INST_name]['HP_fmdcs']
        fmdc_HP = self.INSTs[INST_name]['fmdc_HP']
        for key in HP_fmdcs.keys():
            if key == fmdc_HP:
                value_names.append([HP_fmdcs[key], key+'(fmdc)'])
            else:
                value_names.append([HP_fmdcs[key], key])

        # weval_fmds.min(), reval_fmds.max()에 대한 값과 이을 모두 value_names에 넣음
        # reval_fmds: 정분류 평가 데이터에 대한 fmds, weval_fmds: 오분류 평가 데이터에 대한 fmds
        eval_fmds = self.INSTs[INST_name]['eval_fmds'][eval_name]
        reval_fmds = eval_fmds[self.eval_U[eval_name]]; reval_fmds_max = reval_fmds.max(); value_names.append([reval_fmds_max, f'reval_fmds_max_{eval_name}'])
        weval_fmds = eval_fmds[np.logical_not(self.eval_U[eval_name])]; weval_fmds_min = weval_fmds.min(); value_names.append([weval_fmds_min, f'weval_fmds_min_{eval_name}'])

        # * 3.2. 실선 그리기
        for value, name in value_names:
            if name == f'reval_fmds_max_{eval_name}' or name == f'weval_fmds_min_{eval_name}':
                plt.plot([value, value], [0, 1], label=name, linestyle='--', linewidth=4, alpha=0.4)
            elif name ==  fmdc_HP + '(fmdc)':
                plt.plot([value, value], [0, 1], label=name, linestyle=':', linewidth=4, alpha=0.4)
            else:
                plt.plot([value, value], [0, 1], label=name)
        # legend로 값의 label 표시
        plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', ncol=1)

        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.xlabel('feature map distance criteria(threshold, fmdc)', {'size': '16'})
        plt.ylabel('TPR / TNR', {'size': '16'})
        plt.show()

        # * 4. HP_fmdcs, weval_fmds.min(), reval_fmds.max() 값 출력하기
        for value, name in value_names:
            if name == 'rvalid_fmds_max_wvalid_fmds_min_average' or\
               name == 'rvalid_fmds_max_wvalid_fmds_min_average(fmdc)':
                print(f"{name}:\t{value: 0.4f}")
            elif name.split('_')[-1] == 'average(fmdc)' or name.split('_')[-1] == 'middle(fmdc)':
                    print(f"{name}:\t\t\t{value: 0.4f}")
            else:
                print(f"{name}:\t\t\t\t{value: 0.4f}")

    def show_roc_curve(self, INST_name, eval_name, show_category=True):
        if show_category == True:
            # * 1. 이 show 메소드를 실행하는 범주 출력
            root_dir_splited = self.root_dir.split('/')
            data_set_name = root_dir_splited[-2]; class_name = root_dir_splited[-1]
            # * 특정 INST가 있으니 데이터셋, 클래스, INST_name을 출력한다.
            print(f'[{data_set_name}, {class_name}, [{INST_name}], {eval_name}]')
            print()

        # * 2. ROC, AUC 그리기
        TPRs = self.INSTs[INST_name]['TPRs'][eval_name]; TNRs = self.INSTs[INST_name]['TNRs'][eval_name]; AUC = self.INSTs[INST_name]['AUC'][eval_name]
        # ROC 그리기
        plt.title('ROC curve')
        plt.plot(1 - TNRs, TPRs)
        # AUC 그리기
        AUC_plot_x = []; AUC_plot_y = []
        for i in range(len(TPRs)):
            AUC_plot_x.append(1-TNRs[i]); AUC_plot_x.append(1-TNRs[i])
            AUC_plot_y.append(0); AUC_plot_y.append(TPRs[i])
        plt.plot(AUC_plot_x, AUC_plot_y, label='AUC', color='red', alpha=0.4, linewidth=8)

        plt.xlabel('FPR(= 1 - TNR)')
        plt.ylabel('TPR')
        plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
        plt.show()

        print(f'AUC: {AUC: 0.4f}')

    def show_eval_infos(self, INST_name, eval_name):
        # * 1. 이 show 메소드를 실행하는 범주 출력
        root_dir_splited = self.root_dir.split('/')
        data_set_name = root_dir_splited[-2]; class_name = root_dir_splited[-1]
        # * 특정 INST가 있으니 데이터셋, 클래스, INST_name을 출력한다.
        print(f'[{data_set_name}, {class_name}, [{INST_name}], {eval_name}]')
        print()

        # * 2. eval_infos 출력
        # eval 벤 다이어그램 그리기
        self.show_eval_venn_diagram(INST_name, eval_name, show_category=False)
        # 효과성 및 FMD 비율 그리기
        self.show_efficience_and_FMD_ratio(INST_name, eval_name, show_category=False)
        # eval_U, eval_U_predicted, eval_U_matched 출력
        self.show_eval_U_predicted_matched(INST_name, eval_name, show_category=False)
        # eval_fmd-정분류 비율 그래프 그리기
        self.show_eval_fmd_right_ratio(INST_name, eval_name, show_category=False)
        # rvalid_fmds, wvalid_fmds, eval_fmds, reval_fmds, weval_fmds 출력
        self.show_fmds(INST_name, eval_name, show_category=False)
        # fmdc_TNR_TPR 그래프 그리기
        self.show_fmdc_TNR_TPR(INST_name, eval_name, show_category=False)
        # show_roc_curve 그래프 그리기
        self.show_roc_curve(INST_name, eval_name, show_category=False)

    def show_all(self, show_all_mask):
        # * 1. 이 show 메소드를 실행하는 범주 출력
        root_dir_splited = self.root_dir.split('/')
        data_set_name = root_dir_splited[-2]; class_name = root_dir_splited[-1]
        print(f'[{data_set_name}, {class_name}]')
        print()

        if show_all_mask['show_dir_infos']:
            print('show_dir_infos()'); self.show_dir_infos()

        if show_all_mask['show_data_infos']:
            print('show_data_infos()'); self.show_data_infos()

        if show_all_mask['show_HP']:
            print('show_HP()'); self.show_HP()

        if show_all_mask['show_FM_repres']:
            self.set_square_NPs_infos(figsize=[60,60], column=7)
            print('show_FM_repres()'); self.show_FM_repres()

        if show_all_mask['show_AMs_and_related']:
            self.set_square_NPs_infos(figsize=[60,60], column=7)
            print('show_AMs_and_related()'); self.show_AMs_and_related()

        if show_all_mask['show_layer_infos']:
            print('show_layer_infos()'); self.show_layer_infos()

        if show_all_mask['show_fmds_as_box_plot']:
            print('show_fmds_as_box_plot()'); self.show_rvalid_fmds_and_wvalid_fmds()

        if show_all_mask['show_fmdc_infos']:
            print('show_fmdc_infos()'); self.show_fmdc_infos()

        if show_all_mask['show_eval_infos']:
            print('show_eval_infos()'); self.show_eval_infos()

        if show_all_mask['show_fmd_right_ratio_graph']:
            print('show_fmd_right_ratio_graph()'); self.show_fmd_right_ratio_graph()

        if show_all_mask['show_eval_venn_diagrams']:
            print('show_eval_venn_diagrams()'); self.show_eval_venn_diagrams()

        if show_all_mask['show_efficience_and_FMD_ratio']:
            print('show_efficience_and_FMD_ratio()'); self.show_efficience_and_FMD_ratio()

    def save(self, model_name):
        # os.system(f'touch ./instances/{model_name}.pickle')
        # 인자로 지정된 경로와 이름으로 파일 저장
        with open(f"{self.root_dir}/instances/{model_name}.pickle", "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self, model_name):
        # 인자로 지정된 경로와 이름으로 파일 불러오기
        with open(f"{self.root_dir}/instances/{model_name}.pickle", "rb" ) as f:
            return copy.deepcopy(pickle.load(f))