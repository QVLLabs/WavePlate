import math
import cmath
import numpy as np

def apply_to_one(state,sp_no,multfunc,matrix,ratiomaintain=False):
  states = []
  for i in state:
    for idx, j in enumerate(i):
      states.append([idx,j])

  all_states = []
  operation = True
  n_ops = 0
  for id,i in enumerate(states):
    if operation:
      idx,j = i[0],i[1]
      n_ops += 1
      if j in [1,-1]:
        ini = np.array([np.zeros(8)])
        if j > 0: ini[0][idx] = 1
        else: ini[0][idx] = -1
        all_states.append(ini)
      if round(j,5) not in [0,1,-1]:
        ini = np.array([np.zeros(8)])
        try:
          if round(states[id+1][1],5) not in [0,1,-1]:
            ini[0][idx] = j
            ini[0][idx + 1] = states[id + 1][1]
            all_states.append(ini)
            operation = False
          else:
            if j > 0: ini[0][idx] = 1
            else: ini[0][idx] = -1
            all_states.append(ini)
        except:
          if round(states[id][1],5) not in [0,1,-1]:
            ini[0][idx] = j
            ini[0][idx] = states[id][1]
            all_states.append(ini)
          else:
            if j > 0: ini[0][idx] = 1
            else: ini[0][idx] = -1
            all_states.append(ini)

      if round(j,5) not in [0,1,-1] and id % 2 != 0: operation = False

      if n_ops == 8: n_ops = 0
    else: operation = True
  all_new_states = []
  for idx, i in enumerate(all_states):
    if idx == sp_no: all_new_states.append([multfunc(all_states[idx], matrix)])
    else: all_new_states.append([all_states[idx]])
  new_state = np.array([np.zeros(8)])
  n_states,idxs = 0,[]
  for i in all_new_states:
    for j in i:
      for t in j:
        for idx,k in enumerate(t):
          if round(k, 5) != 0:
            if idx not in idxs:
              n_states += 1
              idxs.append(idx)
  #if show:print(all_states)
  #if show: print(all_new_states)
  no = math.sqrt(1 / n_states)
  ratios = []
  for i in all_new_states:
    for j in i:
      for t in j:
        for idx, k in enumerate(t):
          if idx%2 == 0 and k not in [1,-1]:
            ratios.append( [idx,t[idx]/t[idx+1]] )
          if round(k, 5) != 0:
            if k > 0: new_state[0][idx] = no
            else: new_state[0][idx] = -no
  #if show:print(new_state)
  if ratiomaintain:
    for idx,i in enumerate(new_state[0]):
      if idx % 2 == 0:
        for id,ra in ratios:
          if idx == id:
            if not math.isnan(ra) and round(ra,2) not in [0,1] :
              ra = abs(ra)
              sum = abs(new_state[0][idx]) + abs(new_state[0][idx+1])
              if round(ra,2) >= 1:
                new_state[0][idx] = sum * ra
                new_state[0][idx+1] = sum * (1-ra)
              else:
                new_state[0][idx] = sum * ra
                new_state[0][idx+1] = sum * (1-ra)
  return new_state

initial_state = np.array([ [1,0,0,0,0,0,0,0] ]) # |RH>
VS = ['|RH>','|RV>','|UH>','|UV>','|LH>','|LV>','|DH>','|DV>']
HVS = ['|H>','|V>']
#decode = {'|RH>': '000', '|RV>': '001' ,'|UH>': '010', '|UV>': '011', '|LH>': '100', '|LV>': '101', '|DH>': '110', '|DV>': '111'}


class PhtotonicCircuit:
  def __init__(self,n_beams=1,initial_state=initial_state):
    self.state = initial_state
    self.circuit = []
  def multiply(self,matrix1,matrix2):
    return np.dot(matrix1,matrix2)

  def HWP(self,theta,sp_no=None):
    """ If you wanna apply the Have wave plate to only one path from different paths you can define the sp_no, but if you're willing to apply to
        all possible paths please leave the sp_no 'None' to avoid any errors
        Note: HWP has a phase shift of e^(iπ) = -1 """
    self.circuit.append(['HWP',theta,sp_no])
    check = theta % 2
    theta = math.radians(theta)
    first = math.cos(theta)**2 - math.sin(theta)**2
    second = 2 * math.cos(theta) * math.sin(theta)
    third = math.sin(theta)**2 - math.cos(theta)**2
    fourth = 2 * (-math.sin(theta)) * math.cos(theta)
    self.hwpmatrix = np.array([ [first,second,0,0,0,0,0,0],[second,third,0,0,0,0,0,0],[0,0,third,fourth,0,0,0,0],[0,0,fourth,third,0,0,0,0],[0,0,0,0,first,fourth,0,0],[0,0,0,0,fourth,third,0,0],[0,0,0,0,0,0,third,second],[0,0,0,0,0,0,second,first] ])
    result = self.multiply(self.state , self.hwpmatrix)
    if sp_no == None: self.state = result
    if sp_no != None:
          #print(check)
          if check in [0,0.5,1]:self.state = apply_to_one(self.state,sp_no,self.multiply,self.hwpmatrix)
          else:self.state = apply_to_one(self.state,sp_no,self.multiply,self.hwpmatrix,ratiomaintain=True)
          return self.state
    return result

  def BS(self,theta,sp_no=None):
    """ Beam splitter with angle θ """
    self.circuit.append(['BS',theta,sp_no])
    rtheta = math.radians(theta)
    def first():
      if theta > 0 and theta < 45 or theta > 135 and theta < 180 or theta > 315 and theta < 360: return abs(math.sin(rtheta))
      else: return 1/math.sqrt(2)
    def third():
      if theta > 45 and theta < 135 or theta > 225 and theta < 315: return abs(math.cos(rtheta))
      else: return 1/math.sqrt(2)
    def second():
      if theta > 0 and theta < 45 or theta > 135 and theta < 225 or theta > 315 and theta < 360: return ( (1/math.sqrt(2)) * math.cos(rtheta) ) - ( (1/math.sqrt(2)) * math.sin(rtheta) )
      else: return 0
    def asecond():
      if theta > 0 and theta < 45 or theta > 135 and theta < 225 or theta > 315 and theta < 360: return ( (1/math.sqrt(2)) * math.sin(rtheta) ) - ( (1/math.sqrt(2)) * math.cos(rtheta) )
      else: return 0
    def fourth():
      if theta == 45 or theta == 225: return -math.sin(rtheta)
      else: return 0
    def afourth():
      if theta == 45 or theta == 225: return math.sin(rtheta)
      else: return 0
    def fifth():
      if theta > 45 and theta < 135 or theta > 225 and theta < 315: return ( (1/math.sqrt(2)) * math.cos(rtheta) ) - ( (1/math.sqrt(2)) * math.sin(rtheta) )
      else: return 0
    def afifth():
      if theta > 45 and theta < 135 or theta > 225 and theta < 315: return ( (1/math.sqrt(2)) * math.sin(rtheta) ) - ( (1/math.sqrt(2)) * math.cos(rtheta) )
      else: return 0
    def sixth():
      if theta == 135 or theta == 315: return -math.sin(rtheta)
      else: return 0
    def asixth():
      if theta == 135 or theta == 315: return math.sin(rtheta)
      else: return 0
    self.bsmatrix = np.array([ [first(),0,fourth(),0,fifth(),0,sixth(),0],[0,first(),0,afourth(),0,afifth(),0,asixth()],[afourth(),0,third(),0,sixth(),0,second(),0],[0,fourth(),0,third(),0,asixth(),0,asecond()],[afifth(),0,asixth(),0,first(),0,afourth(),0],[0,fifth(),0,sixth(),0,first(),0,fourth()],[asixth(),0,asecond(),0,fourth(),0,third(),0],[0,sixth(),0,second(),0,afourth(),0,third()] ])
    result = self.multiply(self.state,self.bsmatrix)
    self.state = result
    return result

  def mirror(self,theta,sp_no = None):
    """ In mirrors if you're willing to apply the same mirror with the same θ please use qc.mirror(135) "for example" to all possible pathes
        don't use qc.mirror(135,0) qc.mirror(135,1) "for example" to avoid any error """
    self.circuit.append(['Mirror',theta,sp_no])
    rtheta = math.radians(theta)
    def first():
      if theta >= 0 and theta <= 90: return 2 * math.sin(rtheta) * math.cos(rtheta)
      elif theta >= 180 and theta <= 270: return 2 * math.sin(rtheta) * math.cos(rtheta)
      else: return 0
    def second():
      if theta >= 0 and theta <= 90: return 2 * (-math.sin(rtheta)) * math.cos(rtheta)
      elif theta >= 180 and theta <= 270: return 2 * (-math.sin(rtheta)) * math.cos(rtheta)
      else: return 0
    def third():
      if theta >= 45 and theta <= 135: return  math.sin(rtheta)**2 - math.cos(rtheta)**2
      elif theta >= 225 and theta <= 315: return  math.sin(rtheta)**2 - math.cos(rtheta)**2
      else: return 0
    def fourth():
      if theta > 45 and theta < 135: return  -(math.sin(rtheta)**2) - math.cos(rtheta)**2
      elif theta > 225 and theta < 315: return  -(math.sin(rtheta)**2) - math.cos(rtheta)**2
      else: return 0
    def fifth():
      if theta > 315 and theta <= 360 or theta >= 0 and theta <= 45: return  math.cos(rtheta)**2 - math.sin(rtheta)**2
      elif theta > 135 and theta <= 225: return  math.cos(rtheta)**2 - math.sin(rtheta)**2
      else: return 0
    def sixth():
      if theta > 315 and theta <= 360: return  -(math.cos(rtheta)**2) - math.sin(rtheta)**2
      if theta >= 0 and theta < 45: return  -(math.cos(rtheta)**2) - math.sin(rtheta)**2
      elif theta > 135 and theta < 225: return  -(math.cos(rtheta)**2) - math.sin(rtheta)**2
      else: return 0
    def seventh():
      if theta >= 90 and theta <= 180: return 2 * -(math.sin(rtheta)) * math.cos(rtheta)
      elif theta >= 270 and theta <= 360: return 2 * -(math.sin(rtheta)) * math.cos(rtheta)
      else: return 0
    def eighth():
      if theta >= 90 and theta <= 180: return 2 * math.sin(rtheta) * math.cos(rtheta)
      elif theta >= 270 and theta <= 360: return 2 * math.sin(rtheta) * math.cos(rtheta)
      else: return 0
    self.mirrormatrix = np.array([ [0,0,first(),0,third(),0,seventh(),0],[0,0,0,second(),0,fourth(),0,eighth()],[first(),0,0,0,seventh(),0,fifth(),0],[0,second(),0,0,0,eighth(),0,sixth()],[third(),0,seventh(),0,0,0,first(),0],[0,fourth(),0,eighth(),0,0,0,second()],[seventh(),0,fifth(),0,first(),0,0,0],[0,eighth(),0,sixth(),0,second(),0,0] ])
    if sp_no != None:
      self.state = apply_to_one(self.state,sp_no,self.multiply,self.mirrormatrix)
      return self.state
    result = self.multiply(self.state,self.mirrormatrix)
    self.state = result
    return result

  def PBS(self,theta,sp_no=None):
    """ the polarizing beam splitter have a phase shift of e^(i*2π) = 1 and e^(i*0.5π) = i """
    """ the PBS is a cube so in order to get the best results you can use angels of (0,90,180,270) """
    phase = (np.round(cmath.exp(1j * 0.5 * np.pi).real,2) + np.round(cmath.exp(1j * 0.5 * np.pi).imag,2)) * 1j
    rtheta = math.radians(theta)
    first = math.cos(rtheta) * phase
    second = math.sin(rtheta) * phase

    self.pbsmatrix = [ [1,0,0,0,0,0,0,0] , [0,0,0,first,0,0,0,second] , [0,0,1,0,0,0,0,0] , [0,first,0,0,0,second,0,0] , [0,0,0,0,1,0,0,0] , [0,0,0,second,0,0,0,first] , [0,0,0,0,0,0,1,0] , [0,second,0,0,0,first,0,0] ]

    if sp_no != None: self.state = apply_to_one(self.state,sp_no,self.multiply,self.pbsmatrix)
    else: self.state = self.multiply(self.state,self.pbsmatrix)
    return self.state

  def show_state(self,state=None,show=True,type='ALL'):
    if state == None: state = self.state
    res = ''
    for idx,i in enumerate(state[0]):
      if i != 0:
        if idx != len(state[0]):
          res += '+ ' if i > 0 else ''
          res += '- ' if i < 0 else ''
        res += str(abs(np.round(i,2)))
        res += VS[idx] + ' ' if type == 'ALL' else HVS[idx] + ' '

    if show: print('\n',res,'\n')
    return res

  def measure_probs(self,state=None,shots=100,typ='ALL'):
    if state == None: state = self.state
    probs = {}
    prob = []
    for idx, i in enumerate(state[0]):
        if i.real != 0: prob.append(i.real**2)
        else: prob.append(i.imag**2)
    n_states = 0
    for i in prob:
      if round(i,5) != 0: n_states += 1
    total = sum(prob)
    prob = [p / total for p in prob]
    for _ in range(shots + n_states ):
        choice = np.random.choice(np.arange(0,len(state[0])), p=prob)
        key = VS[choice] if typ == 'ALL' else HVS[choice]
        if key not in probs: probs[key] = 0
        else: probs[key] += 1
    return probs

  def Z_expectation(self,shots):
    probs = self.measure_probs(shots=shots)
    decode = {}
    d_states = ['000','001','010','011','100','101','110','111']
    for idx,i in enumerate(probs.keys()): decode[i] = d_states[idx]
    expects = np.zeros(3)
    decoded = {}
    for state in decode.keys():
      for key in probs.keys():
        if state == key: decoded[decode[state]] = probs[key]
    for key in decoded.keys():
        perc = decoded[key]/shots
        check = np.array([( float(key[i]) - 1/2 )*2*perc for i in range(3)])
        expects += check
    return expects

  def compute(self,direction,show=True):
    doubles  = [ [i,self.state[0][idx+1]]  for idx,i in enumerate(self.state[0]) if idx % 2 == 0 ]
    doubles = [ [doubles[direction]] ]
    h,v = np.sqrt(abs(doubles[0][0][0])),np.sqrt(abs(doubles[0][0][1]))
    sum = h + v
    hv_state = [ [np.sqrt(h/sum) , np.sqrt(v/sum)] ]
    if show == True: self.show_state(state=hv_state,type='hv')
    return hv_state

  def draw(self):
    # still thinking how to implment this
    print(self.circuit)