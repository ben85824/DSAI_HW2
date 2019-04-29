import numpy as np
from keras.models import Sequential, Model
from keras.layers import LSTM, RepeatVector, Dense, BatchNormalization, Input
import matplotlib.pyplot as plt

class CharacterTable(object):
    def __init__(self):
        self.chars = [' ','+','-','*','1','2','3','4','5','6','7','8','9','0']
        self.c2i = dict((c, i) for i, c in enumerate(self.chars))
        self.i2c = dict((i, c) for i, c in enumerate(self.chars))
        self.len = len(self.chars)
    
    def encoder(self, C):
        x = np.zeros((len(C), self.len))
        for i, c in enumerate(C):
            x[i, self.c2i[c]] = 1
        return x
    
    def decoder(self, X, max_len = 4):
        answer = ''
        for x in X:
            answer += self.i2c[np.argmax(x)]
        return '{string:' '{digits}}'.format(string = str(answer), digits = max_len)
        
def generator(operator = ['+','-'], digits = 3, training_size = 80000):
    max_num = np.power(10,digits) - 1
    opt_len = len(operator)
    Q, A = [], []
    for i in range(training_size):
        rand_int = np.random.randint(0,max_num,2)
        rand_int.sort()
        a,b = rand_int[1],rand_int[0] # a > b
        if opt_len == 1:
            opt = operator[0]
        else:
            opt = operator[i%opt_len]
        if opt == '+':
            if np.random.randint(0,2,1) == 0:
                a,b = rand_int[0],rand_int[1]
            answer = a + b
        else:
            answer = a - b
        q = '{0:7}'.format(str(a)+str(opt)+str(b))
        ans = '{string:' '{digits}}'.format(string = str(answer), digits = digits+1)
        Q.append(q)
        A.append(ans)
    return Q, A

def mul_generator(operator = ['*'], digits = 3, training_size = 80000):
    max_num = np.power(10,digits) - 1
    Q, A = [], []
    for i in range(training_size):
        rand_int = np.random.randint(0,max_num,2)
        a,b = rand_int[0],rand_int[1]
        opt = '*'
        answer = a * b
        q = '{0:7}'.format(str(a)+str(opt)+str(b))
        ans = '{string:' '{digits}}'.format(string = str(answer), digits = digits*2)
        Q.append(q)
        A.append(ans)
    return Q, A
  
def seq_model(hidden_size = 64,feature_size = 14, input_shape = (7,14), output_shape = (4,14)):
    lstm0 = LSTM(hidden_size, return_sequences=True)
    lstm1 = LSTM(hidden_size, return_sequences=True)
    lstm2 = LSTM(hidden_size, return_state=True)

    lstm3 = LSTM(hidden_size, return_sequences=True)
    #lstm4 = LSTM(64, return_sequences=True)
    #lstm5 = LSTM(64, return_sequences=True)
    fc = Dense(feature_size, activation = 'softmax')
    encoder_inputs = Input(shape = input_shape)
    decoder_inputs = Input(shape = output_shape)
    encoder = lstm0(encoder_inputs)
    encoder = BatchNormalization()(encoder)
    encoder = lstm1(encoder)
    encoder = BatchNormalization()(encoder)
    encoder_outputs_and_states = lstm2(encoder)
    encoder_states =  encoder_outputs_and_states[1:]
    decoder = lstm3(decoder_inputs, initial_state=encoder_states)
    decoder = BatchNormalization()(decoder)
    #decoder = lstm4(decoder)
    #decoder = BatchNormalization()(decoder)
    #decoder = lstm5(decoder)
    #decoder = BatchNormalization()(decoder)
    decoder_outputs = fc(decoder)
    model = Model(inputs = [encoder_inputs, decoder_inputs], outputs = decoder_outputs)
    return model
    
def print_result(q,a,pred_a):
    print('##########################')
    for i in range(len(q)):
        if a[i] == pred_a[i]:
            print('Q %s T %s V %s'%(q[i],a[i],pred_a[i]))
        else:
            print('Q %s T %s X %s'%(q[i],a[i],pred_a[i]))
    print('##########################')
            
def cal_acc(a,pred_a):
    t,f = 0,0
    for i in range(len(a)):
        if a[i] == pred_a[i]:
            t+=1
        else:
            f+=1
    return t/(t+f)

if __name__ == '__main__':
    ctable = CharacterTable()
    
    # Adder
    print('Start train Adder')
    model = seq_model()
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')
    # generate validation data
    val_Q, val_A = generator(operator=['+'], training_size=30000)
    val_x = np.array([ctable.encoder(x) for x in val_Q])
    val_y = np.array([ctable.encoder(x) for x in val_A])
    val_y_zero = np.zeros_like(val_y)
    loss_list, val_loss_list = [],[]
    for i in range(5):
        # generate train data
        Q, A = generator(operator=['+'])
        train_x = np.array([ctable.encoder(x) for x in Q])
        train_y = np.array([ctable.encoder(x) for x in A])
        train_y_zero = np.zeros_like(train_y)
        # tain
        hist = model.fit([train_x, train_y_zero], train_y, batch_size=128, epochs = 10, validation_data = ([val_x, val_y_zero],val_y))
        loss_list.extend(hist.history['loss'])
        val_loss_list.extend(hist.history['val_loss'])
        # test
        pred_A = model.predict([val_x, val_y_zero])
        pred_A = [ctable.decoder(x, 4) for x in pred_A]
        print('Epoch:%d/%d'%((i+1)*10,50))
        print('acc: %f'%(cal_acc(val_A,pred_A)))
        print_result(val_Q[:10],val_A[:10],pred_A[:10])
    plt.figure()
    plt.plot([x for x in range(len(loss_list))],loss_list)
    plt.plot([x for x in range(len(loss_list))],val_loss_list)
    plt.legend(['loss', 'val_loss'], loc='upper left')
    plt.savefig('adder.png')

    # Subtractor
    print('Start train Subtractor')
    model = seq_model()
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')
    # generate validation data
    val_Q, val_A = generator(operator=['-'], training_size=30000)
    val_x = np.array([ctable.encoder(x) for x in val_Q])
    val_y = np.array([ctable.encoder(x) for x in val_A])
    val_y_zero = np.zeros_like(val_y)
    loss_list, val_loss_list = [],[]
    for i in range(5):
        # generate train data
        Q, A = generator(operator=['-'])
        train_x = np.array([ctable.encoder(x) for x in Q])
        train_y = np.array([ctable.encoder(x) for x in A])
        train_y_zero = np.zeros_like(train_y)
        # tain
        hist = model.fit([train_x, train_y_zero], train_y, batch_size=128, epochs = 10, validation_data = ([val_x, val_y_zero],val_y))
        loss_list.extend(hist.history['loss'])
        val_loss_list.extend(hist.history['val_loss'])
        # test
        pred_A = model.predict([val_x, val_y_zero])
        pred_A = [ctable.decoder(x, 4) for x in pred_A]
        print('Epoch:%d/%d'%((i+1)*10,50))
        print('acc: %f'%(cal_acc(val_A,pred_A)))
        print_result(val_Q[:10],val_A[:10],pred_A[:10])
    plt.figure()
    plt.plot([x for x in range(len(loss_list))],loss_list)
    plt.plot([x for x in range(len(loss_list))],val_loss_list)
    plt.legend(['loss', 'val_loss'], loc='upper left')
    plt.savefig('subtractor.png')

    # Subtractor
    print('Start train adder and subtractor')
    model = seq_model()
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')
    # generate validation data
    val_Q, val_A = generator(operator=['+','-'], training_size=30000)
    val_x = np.array([ctable.encoder(x) for x in val_Q])
    val_y = np.array([ctable.encoder(x) for x in val_A])
    val_y_zero = np.zeros_like(val_y)
    loss_list, val_loss_list = [],[]
    for i in range(5):
        # generate train data
        Q, A = generator(operator=['+','-'])
        train_x = np.array([ctable.encoder(x) for x in Q])
        train_y = np.array([ctable.encoder(x) for x in A])
        train_y_zero = np.zeros_like(train_y)
        # tain
        hist = model.fit([train_x, train_y_zero], train_y, batch_size=128, epochs = 10, validation_data = ([val_x, val_y_zero],val_y))
        loss_list.extend(hist.history['loss'])
        val_loss_list.extend(hist.history['val_loss'])
        # test
        pred_A = model.predict([val_x, val_y_zero])
        pred_A = [ctable.decoder(x, 4) for x in pred_A]
        print('Epoch:%d/%d'%((i+1)*10,50))
        print('acc: %f'%(cal_acc(val_A,pred_A)))
        print_result(val_Q[:10],val_A[:10],pred_A[:10])
    plt.figure()
    plt.plot([x for x in range(len(loss_list))],loss_list)
    plt.plot([x for x in range(len(loss_list))],val_loss_list)
    plt.legend(['loss', 'val_loss'], loc='upper left')
    plt.savefig('merge.png')

    # multiplication
    print('Start train multiplication')
    model = seq_model(hidden_size=64, output_shape=(6,14))
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')
    # generate validation data
    val_Q, val_A = mul_generator(training_size=30000)
    val_x = np.array([ctable.encoder(x) for x in val_Q])
    val_y = np.array([ctable.encoder(x) for x in val_A])
    val_y_zero = np.zeros_like(val_y)
    loss_list, val_loss_list = [],[]
    for i in range(5):
        # generate train data
        Q, A = mul_generator()
        train_x = np.array([ctable.encoder(x) for x in Q])
        train_y = np.array([ctable.encoder(x) for x in A])
        train_y_zero = np.zeros_like(train_y)
        # tain
        hist = model.fit([train_x, train_y_zero], train_y, batch_size=128, epochs = 10, validation_data = ([val_x, val_y_zero],val_y))
        loss_list.extend(hist.history['loss'])
        val_loss_list.extend(hist.history['val_loss'])
        # test
        pred_A = model.predict([val_x, val_y_zero])
        pred_A = [ctable.decoder(x, 4) for x in pred_A]
        print('Epoch:%d/%d'%((i+1)*10,50))
        print('acc: %f'%(cal_acc(val_A,pred_A)))
        print_result(val_Q[:10],val_A[:10],pred_A[:10])
    plt.figure()
    plt.plot([x for x in range(len(loss_list))],loss_list)
    plt.plot([x for x in range(len(loss_list))],val_loss_list)
    plt.legend(['loss', 'val_loss'], loc='upper left')
    plt.savefig('multiplication.png')

