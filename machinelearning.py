'''
    BUGSE ERDOGAN - 150112082
    FAHRIYE GUN - 150112025

'''

''' READ TRAIN.TXT FILE'''
with open("train.txt") as trainFile:
    lines = [line.split() for line in trainFile]

    array_dic = []
    array_class = []

    word_1 = 'B'
    pos_1 = 'B'
    ''' CREATE DICTONARY FOR EVERY SET OF FEATURES'''
    for i in range(0,len(lines)):

        '''
        # 1. token[i]
        if (lines[i] != []):
            dic = dict()
            dic['word'] = lines[i][0]

            array_dic.append(dic)
            array_class.append(lines[i][2])
        '''

        #'''
        # 2. token[i].pos[i]
        if (lines[i] != []):
            dic = dict()
            dic['word'] = lines[i][0]
            dic['pos'] = lines[i][1]
            array_dic.append(dic)
            array_class.append(lines[i][2])
         #'''
        '''
        # 3. token[i].token[i-1].pos[i].pos[i-1]
        if (lines[i] != []):
            dic = dict()

            dic['word'] = lines[i][0]
            dic['pos'] = lines[i][1]
            dic['word-1'] = word_1
            dic['pos-1'] = pos_1
            array_dic.append(dic)
            array_class.append(lines[i][2])
            word_1 = lines[i][0]
            pos_1 = lines[i][1]

        else:
            word_1 = 'B'
            pos_1 = 'B'

        '''


        '''
        # 4. token[i].token[i-1]
        if (lines[i] != []):
            dic = dict()
            dic['word'] = lines[i][0]
            dic['word-1'] = word_1
            array_dic.append(dic)
            array_class.append(lines[i][2])
            word_1 = lines[i][0]

        else:
             word_1 = 'B'

        '''
        '''
        # 5. token[i].token[i-1].pos[i]
        if (lines[i] != []):
            dic = dict()
            dic['word'] = lines[i][0]
            dic['pos'] = lines[i][1]
            dic['word-1'] = word_1
            array_dic.append(dic)
            array_class.append(lines[i][2])
            word_1 = lines[i][0]
        else:
            word_1 = 'B'

        #'''

    '''MAKE VECTORIZATION'''
    from sklearn.feature_extraction import DictVectorizer
    vec = DictVectorizer()
    train_vector = vec.fit_transform(array_dic)

    #'''
    #1. METHOD: LOGISTIC REGRESSION
    from sklearn.linear_model.logistic import LogisticRegression
    train_model = LogisticRegression()
    train_model.fit(train_vector,array_class)
    #'''

    '''
    #2. METHOD: MULTINOMIAL NAIVE BAYES
    from sklearn.naive_bayes import MultinomialNB
    train_model = MultinomialNB()
    train_model.fit(train_vector, array_class)
    '''

''' READ TEST.TXT AND MAKE DICTONARY FOR TEST FILE'''
with open("test.txt") as testFile:
    lines_test = [line.split() for line in testFile]

    array_dic_test = []
    test_class = []
    for i in range(0,len(lines_test)):

        if (lines_test[i] != []):
            dic_test = dict()
            dic_test['word'] = lines_test[i][0]
            dic_test['pos'] = lines_test[i][1]
            array_dic_test.append(dic_test)
            test_class.append(lines_test[i][2])

    test_vector = vec.transform(array_dic_test)

    #PREDICT CLASS
    predicted = []
    predicted =train_model.predict(test_vector)

'''CALCULATE PRECISION/RECALL/F1 SCORE'''
from sklearn.metrics import classification_report
print(classification_report(test_class, predicted, target_names=list(set(test_class))))
