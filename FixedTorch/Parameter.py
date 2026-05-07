class Parameter:
    wl = 64
    bitlength = 63
    fl = 16
    trunc_type = 1 # 0 faithful 1 stochastic 2 local 3 TruncXpert
    softmax_type = 0  # 0 piranha-submax  1 piranha-relu  2 sigma
    pp = 0.5
    pv = 1
    np = 0.5
    nv = -1
  #定义类方法

    @classmethod
    def setfl(cls, fl) :
        cls.fl = fl

    @classmethod
    def setwl(cls, wl) :
        cls.wl = wl

    @classmethod
    def settrunc_type(cls, trunc_type) :
        cls.trunc_type = trunc_type

    @classmethod
    def setsoftmax_type(cls, softmax_type) :
        cls.softmax_type = softmax_type
