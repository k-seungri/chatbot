# 데이터 불러오기, 데이터 전처리
from configs import DEFINES
import re
import pandas as pd
from konlpy.tag impot Okt


FILTERS = "([~.,!?\"':;)(])"
PAD = "<PADDING>"
STD = "<START>"
END = "<END>"
UNK = "<UNKNOWN>"

PAD_INDEX=0
STD_INDEX = 1
END_INDEX = 2
UNK_INDEX = 3

MARKER = [PAD,STD,END,UNK]
CHANGE_FILTER = re.compile(FILTERS)

# 판다스를 통해 데이터를 불러와 학습 데이터와 검증 데이터로 분리하는 함수
def load_data():
    data_df = pd.read_csv(DEFINES.data_path, header=0)
    question, answer = list(data_df['Q']), list(data_df['A'])
    train_input, eval_input, train_label, eval_label = train_test_split(question, answer, test_size=0.33, random_state=42)
    return train_input, train_label, eval_input, eval_label

# 한글 텍스트를 토크나이징하기 위해 형태소로 분리하는 작업
def prepro_like_morphlized(data):
    morph_analyzer = Okt()
    result_data = list()
    for seq in tqdm(data):
        morphlized_seq = " ".join(morph_analyzer.morphs(seq.replace(' ', '')))
        result_data.append(morphlized_seq)

    return result_data

# 인코딩 데이터를 만드는 함수
# value : 전처리 할 데이터
# dictionary : 단어 사전
def enc_processing(value, dictionary):
    sequences_input_index = []
    sequences_length = []
    # 형태소 기준 토크나이징
    if DEFINES.tokenize_as_morph:
        value = prepro_like_morphlized(value)

    for sequence in value:
        # 정규 표현식 라이브러리를 이용해 특수문자 모두 제거
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        sequence_index = []
        for word in sequence.split():
            # 각 단어를 단어 사전을 이용해 단어 인덱스에 추가
            if dictionary.get(word) is not None:
                sequence_index.extend([dictionary[word]])
            # 단어사전에 포함돼 잇지 않다면 UNK (UNK 토큰의 인덱스 값은 2) 토큰을 넣는다
            else:
                sequence_index.extend([dictionary[UNK]])

        # 최대 길이보다 긴 문장의 경우 뒤에 토큰 자르기
        if len(sequence_index) > DEFINES.max_sequence_length:
            sequence_index = sequence_index[:DEFINES.max_sequence_length]

        sequences_length.append(len(sequence_index))
        # max_sequence_length보다 문장 길이가 작다면 빈 부분에 PAD(0)를 넣어준다.
        sequence_index += (DEFINES.max_sequence_length - len(sequence_index)) * [dictionary[PAD]]
        sequences_input_index.append(sequence_index)

    # sequences_input_index : 전처리한 데이터
    # sequences_length : 패딩 처리하기 전의 각 문장의 실제 길이를 담고 있는 리스
    # 넘파이 배열로 변경하는 이유는 텐서플로우 dataset에 넣어 주기 위한 사전 작업
    return np.asarray(sequences_input_index), sequences_length

# 디코딩 입력 데이터를 만드는 함수
def dec_input_processing(value, dictionary):
    sequences_output_index = []
    sequence_length = []

    if DEFINES.tokenize_as_morph:
        value = prepro_like_morphlized(value)

    for sequence in value :
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        sequence_index = []
        # 각 문장의 처음에 시작 토큰을 넣어 줌
        sequence_index = [dictionary[STD]] + [dictionary[word] for word in sequence.split()]

        if len(sequence_index) > DEFINES.max_sequence_length:
            sequence_index = sequence_index[:DEFINES.max_sequence_length]
        sequence_length.append(len(sequence_index))
        sequence_index += (DEFINES.max_sequence_length - len(sequence_index)) * [dictionary[PAD]]
        sequences_output_index.append(sequence_index)

    return np.asarray(sequences_output_index), sequence_length

# 디코더 타킷값을 만드는 전처리 함수
def dec_target_processing(value, dictionary):
    sequences_target_index = []

    if DEFINES.tokenize_as_morph:
        value = prepro_like_morphlized(value)
    for sequence in value:
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        sequence_index = [dictionary[word] for word in sequence.split()]
        # 문장 제한 길이보다 길이가 길 경우 토큰을 자름
        # 그리고 END 토큰을 넣어 줌
        if len(sequence_index) >= DEFINES.max_sequence_length:
            sequence_index = sequence_index[:DEFINES.max_sequence_length-1] + [dictionary[END]]
        else:
            sequence_index += [dictionary[END]]

        sequence_index += (DEFINES.max_sequence_length - len(sequence_index)) * [dictionary[PAD]]
        sequences_target_index.append(sequence_index)

    return np.asarray(sequences_target_index)

# 학습한 모델을 통해 예측할 때 사용하는 함수
# 인덱스를 스트링을 변경하는 함수
def pred2stirng(value, dictionary):
    sentence_string = []
    for v in value:
        sentence_string = [dictionary[index] for index in v['indexs']]

    print(sentence_string)
    answer = ""

    for word in sentence_string:
        if word not in PAD and word not in END:
            answer += word
            answer += " "

    print(answer)
    return answer

# 데이터 각 요소에 대해서 rearrange 함수를 통해서 요소를 변환하여 맵으로 구성
def rearrange(input, output, target):
    features = {"input":input, "output":ouput}
    return features, target

# 정규 표현식을 사용해 특수기호를 모두 제거
# 단어들을 기준으로 나눠서 전체 데이터의 모든 단어를 포함하는 단어 리스트로 만든다
def load_vocabulary():
    vocabulary_list = []
    if (not (os.path.exists(DEFINES.vocabulary_path))):
        if (os.path.exists(DEFINES.data_path)):
            data_df = pd.read_csv(DEFINES.data_path, encoding='utf-8')
            question, answer = list(data_df['Q']), list(data_df['A'])
            if DEFINES.tokenize_as_morph:
                question = prepro_like_morphlized(question)
                answer = prepro_like_morphlized(answer)
            data = []
            data.extend(question)
            data.extend(answer)
            words = data_tokenizer(data)
            words= list(set(words))
            words[:0] = MARKER

        # 사전 리스트를 사전 파일로 만들어 넣는다
        with open(DEFINES.vocabulary_path, 'w', encoding='utf-8') as vocabulary_file:
            for word in words:
                vocabulary_file.write(word + '\n')
    # 사전 파일이 조냊하면 여기에서 그 파일을 불러서 배열에 넣어준다
    with open(DEFINES.vocabulary_path, 'r', encoding='utf-8') as vocabulary_file:
        for line in vocabulary_file:
            vocabulary_list.append(line.strip())

    word2idx, idx2word = make_vocabulary(vocabulary_list)
    return word2idx, idx2word, len(word2idx)

def make_vocabulary(vocabulary_list):
    word2idx = {word: idx for idx, word in enumerate(vocabulary_list)}
    idx2word = {idx: word for idx, word in enumerate(vocabulary_list)}
    return word2idx, idx2word

# 텐서플로 모델에 데이터를 적용하기 위한 데이터 입력 함수
# 학습을 위한 입력 함수
def train_input_fn(train_input_enc, train_output_dec, train_target_dec, batch_size):
    dataset = tf.data.Dataser.from_tensor_slices((train_input_enc, train_output_dec, train_target_dec))
    dataset = dataset.shuffle(buffer_size=len(train_input_enc))
    assert batch_size is not None, "train batchSize must not be None"
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(rearrange)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()

# 평가를 위한 입력 함수
def eval_input_fn(eval_input_enc, eval_output_dec, eval_target_dec, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((eval_input_enc, eval_output_dec, eval_target_dec))
    # 전체 데이터를 섞는다.
    dataset = dataset.shuffle(buffer_size=len(eval_input_enc))
    assert batch_size is not None, "eval batchSize must not be None"
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(rearrange)
    # 평가이므로 1회만 동작 시킨다.
    dataset = dataset.repeat(1)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()