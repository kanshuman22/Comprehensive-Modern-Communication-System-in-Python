import scipy
import scipy.interpolate
import numpy as np
import matplotlib.pyplot as plt
import random
from viterbi import Viterbi
from reedsolo import RSCodec, ReedSolomonError




bits=6  #bits per symbol (64 QAM)

noofcarriers = 256

carriers = [i for i in range(noofcarriers)]  # Carriers
print(carriers)

blocksize=16

nblocks=noofcarriers//blocksize

cyclicprefix = noofcarriers // 4

# Choosing pilot carriers

pilots_per_block=4    #can be changed


pilotval=1+5j   #choosen pilot value 

pilotCarriers = np.empty(0,dtype=int)

for i in range (0,noofcarriers,blocksize):      #Selecting the pilot carriers ,starting carriers in each block
    for j in range (0,pilots_per_block):
        pilotCarriers=np.append(pilotCarriers,i+j)




print('pilot carriers',pilotCarriers)

#Carriers which carry the data
datacar = np.delete(carriers, pilotCarriers)

print('data carriers',datacar)

seed_value = 42
np.random.seed(seed_value)


randombits = np.random.randint(2, size=(len(datacar) * bits,))

sequencetobesent=randombits #original sequence to be sent is stored

def randomize(data, seed):      #randomization
    random.seed(seed)
    key = [random.randint(0, 1) for _ in range(len(data))]
    randomized_data = [data[i] ^ key[i] for i in range(len(data))]
    randomized_data=np.array(randomized_data)
    return randomized_data, key

randombits,keyr=randomize(randombits,42)


randomsp=randombits.reshape((len(datacar),bits))

datatobesent=randombits




def int_to_binary_tuple(n):
    binary_str = format(n, '06b')  # Convert the integer to a 6-bit binary string
    binary_tuple = tuple(int(bit) for bit in binary_str)  # Convert the binary string to a tuple of integers
    return binary_tuple 

values=[-7,-5,-3,-1,1,3,5,7]

mapping_table={}
mapkeys=0

for i in range(0,len(values)):
    for j in range(0,len(values)):
        mapping_table[int_to_binary_tuple(mapkeys)]=values[i]+1j*values[j]
        mapkeys+=1



demapping_table = {}  
for key, value in mapping_table.items():
    demapping_table[value] = key

for bit5 in [0, 1]:
    for bit4 in [0, 1]:
        for bit3 in [0, 1]:
            for bit2 in [0, 1]:
                for bit1 in [0, 1]:
                    for bit0 in [0, 1]:
                        bits6 = (bit5,bit4,bit3, bit2, bit1, bit0)
                        sym1 = mapping_table[bits6]
                        plt.plot(sym1.real, sym1.imag, 'bo')
                        bit_sequence = "".join(str(x) for x in bits6)
                        plt.annotate(bit_sequence, (sym1.real, sym1.imag + 0.2), ha='center', fontsize=8, color='blue')                        
plt.plot(sym1.real, sym1.imag)
plt.title('Constellation to be used')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.grid(True)
plt.ylim(-11)    
        
SNRdb=20  #adjustable




def block_interleave(data, block_size):
    interleaved_data = []
    num_blocks = len(data) // block_size

    for i in range(block_size):
        for j in range(num_blocks):
            interleaved_data.append(data[j * block_size + i])

    return interleaved_data

interleaved_data = block_interleave(datatobesent, 6)

interleaved_data=np.array(interleaved_data)

randomsp=interleaved_data.reshape((len(datacar),bits))

print(randomsp)

def mapping(randombits):
    return np.array([mapping_table[tuple(seq)] for seq in randombits])

qam=mapping(randomsp)
print (randomsp[:16,:])
print (qam[:16])


def ofdm_data(qam):
    ofdmsym=np.zeros(noofcarriers,dtype=complex)
    ofdmsym[pilotCarriers]=pilotval         #pilot carriers
    #print('pilot values',ofdmsym[pilotCarriers])
    ofdmsym[datacar]=qam
    return ofdmsym

ofdmdata=ofdm_data(qam)
print(ofdmdata)
print('ofdmdata  1',ofdmdata[datacar])


def ifft(ofdmdata):
    return np.fft.ifft(ofdmdata)

ofdmt=ifft(ofdmdata)

print('size after ifft',ofdmt.shape)
print(ofdmt)


#cyclic prefix addition 



def cyclicprefixadd(ofdmt):
    prefix = ofdmt[-cyclicprefix:]
    return np.array(list(prefix) + list(ofdmt))


ofdmcp=cyclicprefixadd(ofdmt)

print('size after adding cp',ofdmcp.shape)

def channel(signal, SNRdb):
    h=1+ 1j
    signal=np.convolve(h,signal)
    signalpower = np.mean(abs(signal**2))  #signal power
    No = signalpower / (10**(SNRdb / 10))  # noise power
    #generating noise
    n = np.sqrt(No / 2) * (np.random.normal(size=signal.shape) + 1j * np.random.normal(size=signal.shape))
    signal=signal + n # adding noise
    return signal

ofdmsend=ofdmcp 
ofdmrec=channel(ofdmsend,SNRdb) #received signal

print('size after receiving from channel',ofdmrec.shape)

def prefixremover(signal):
    return signal[cyclicprefix:(cyclicprefix+noofcarriers)]

ofdmdem=prefixremover(ofdmrec)

print('size after removing cp',ofdmdem.shape)

def fft(signal):                #fft
    return np.fft.fft(signal)
ofdmdem=fft(ofdmdem)


print('size after fft',ofdmdem.shape)


plt.figure(figsize=(15,4))
plt.plot(abs(ofdmdata), label='Data  signal', color='blue')
plt.plot(abs(ofdmsend), label='Transmitted signal', color='black')

plt.plot(abs(ofdmrec), label='Received signal from channel',color='green')
plt.plot(abs(ofdmdem), label='Demodulated ofdm signal before channel estimation',color='red')
plt.title(' Signals Comparison')



def channelestimate(signal1):
    #estimating block wise
    hest=np.empty(0,dtype=complex)
    pilots=signal1[pilotCarriers]
    print('observed pilots',pilots)
    print('pilots size',pilots.size)
    knormsq=pilots_per_block*pilotval**2
    for block in range(0,pilots.size,pilots_per_block):        
        normsq=0
        obsv=pilots[block:block+pilots_per_block]
        for i in range (0,pilots_per_block):
            normsq+=np.abs(pilots[block+i])**2
        arr = np.full((pilots_per_block,), pilotval)
        hblock=np.dot((arr/knormsq),obsv)               #Applying linear regression
        print('hblock ',hblock)
        hblock1=np.full((blocksize,),hblock)
        hest=np.append(hest,hblock1)        #different estimated h for different slots
        print('hest size',hest.size)
        print('norm in  block',block/4,'is',normsq) 
    return hest
hest=channelestimate(ofdmdem)

print('hest size',hest.shape)
print('ofdmdem size',ofdmdem.shape)

def equalizer(hest,ofdm):
    return ofdm/hest

ofdmdem=equalizer(hest,ofdmdem)

ofdmdata1=ofdmdem[datacar]
print('data ofdm symbols',ofdmdata1)


def demapping(ofdmdata1):
    points=np.empty(0,dtype=tuple)
    ofdmdata1=ofdmdata1[datacar]
    estsymbol=np.empty(0,dtype=complex)
    for i in range (0,ofdmdata1.size):
        btup=int_to_binary_tuple(0)
        dist=abs(ofdmdata1[i]-mapping_table[btup])
        symbol=mapping_table[btup]
        for j in range(0,64):
            btj=int_to_binary_tuple(j)
            if (abs(ofdmdata1[i]-mapping_table[btj])<dist):
                dist=np.abs(ofdmdata1[i]-mapping_table[btj])
                #print(' op1',ofdmdata1[i],' op2 ',mapping_table[btj],' = ',dist)
                symbol=mapping_table[btj]
                btup=btj
        #print('btup',btup)
        points=np.append(points,btup)
        estsymbol=np.append(estsymbol,symbol)
    return points,estsymbol

sequence,estsymbol=demapping(ofdmdem)
sequence=sequence.reshape((-1,))

#convolution encoding using vitterbi algorithm
dot11a_codec = Viterbi(7, [0o133, 0o171])

sequence = np.array(dot11a_codec.encode(sequence),dtype=int)


sequence=np.array(dot11a_codec.decode(sequence),dtype=int)
print('sequence first few samples',sequence[0:64])

print('randombits first few samples',randombits[0:64])



plt.plot(abs(ofdmdem), label='Demodulated ofdm signal after equalization',color='grey')
plt.legend(fontsize=10)
plt.xlabel('Time'); plt.ylabel('Absolute Value');
plt.grid(True);
plt.show()

plt.figure()  # Create a new figure
for i in range(len(ofdmdata1)):  # Use len(estsymbol) instead of estsymbol.size
    plt.plot(ofdmdata1[i].real, ofdmdata1[i].imag, 'ro')  # Plot complex number as a point
    plt.grid(True)
    #plt.ylim(-8)  # Set the y-axis limit as needed

plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.title('Received Symbols Complex Plane Plot')
plt.show()


plt.plot(randombits, label='Sequence transmitted',color='red')
plt.plot(sequence, label='Sequence obtained after demapping',color='blue')
plt.legend(fontsize=10)
plt.title('Sequence ')
plt.xlabel('Time'); plt.ylabel('Value');
plt.xlim([0,100])
plt.grid(True);
plt.show()

print('size transmitted data',randombits.size)
print('size  demapped data',sequence.size)

for i in range (0,qam.size):
    print('QAM sent',qam[i])
    print('OFDMdata1',ofdmdata1[i])
    print('Estimated constellation',estsymbol[i])


afterbuffer=[]

#buffer 
class Buffer:
    def __init__(self):
        self.my_list = []

    def add_element(self, element):
        self.my_list.append(element)
        self.make_tuple_if_size()

    def make_tuple_if_size(self):
        if len(self.my_list) == 6:
            my_tuple = self.convert_to_tuple()
            self.clear_list()
            afterbuffer.append(my_tuple)
            return None
        

    def convert_to_tuple(self):
        return tuple(self.my_list)

    def clear_list(self):
        self.my_list.clear()

bufferobj= Buffer()  

for i in range (0,sequence.size):
   bufferobj.add_element(sequence[i])
print(' After buffer ',afterbuffer[:24])

afterbuffer=np.array(afterbuffer)
afterbuffer=afterbuffer.flatten()

def block_deinterleave(interleaved_data, block_size):
    deinterleaved_data = []
    num_blocks = len(interleaved_data) // block_size

    for i in range(num_blocks):
        for j in range(block_size):
            deinterleaved_data.append(interleaved_data[j * num_blocks + i])

    return deinterleaved_data


deinterleaved=block_deinterleave(afterbuffer,block_size=6)

deinterleaved=np.array(deinterleaved)

def derandomize(data, key):
    derandomized_data = [data[i] ^ key[i] for i in range(len(data))]
    return np.array(derandomized_data)

received_data = derandomize(deinterleaved, keyr)

deinterleaved=received_data

deinterleaved = deinterleaved.reshape((len(datacar),bits))

print(" deinter ",deinterleaved[0:20])



rsc = RSCodec(10)

afterrsc = []


for i in range(0, len(deinterleaved)):
    # Data to be encoded
    encoded = rsc.encode(tuple(deinterleaved[i]))

    # Simulating distortion by setting the last element to 1
    distorted = encoded
    distorted[-1] = 1

    # Decoding the distorted data
    decoded_msg, decoded_msgecc, errata_pos = rsc.decode(distorted)

    # Appending the result as a tuple to afterrsc
    afterrsc.append(tuple(decoded_msg[0:6]))
    #print(' encoded ',deinterleaved[i],' decoded ',decoded_msg)

print('afterrsc',afterrsc[0:20])

afterrsc=np.array(afterrsc)

afterrsc=afterrsc.flatten()


print(' original size ',randombits.size,' received size ',afterrsc.size)

wrongbits=0
for i in range(0,randombits.size):
    if(sequencetobesent[i]!=afterrsc[i]):
        wrongbits+=1

print("BER final :",wrongbits/randombits.size)
    



